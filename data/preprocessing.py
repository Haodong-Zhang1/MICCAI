# kits23_segmentation/data/preprocessing.py
import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
import logging
import multiprocessing as mp
from functools import partial
import json
import time


class KidneyTumorPreprocessor:
    """Kidney tumor data preprocessor with memory optimizations and enhanced flexibility"""

    def __init__(self, output_dir="E:/MICCAI/kits23_segmentation/output/preprocessed_data",
                 preprocessing_steps=None, log_level=logging.INFO):
        """
        Initialize the preprocessor with configurable options.

        Args:
            output_dir: Directory to save preprocessed data
            preprocessing_steps: List of preprocessing steps and their parameters
            log_level: Logging level
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

        # Setup logging
        self.logger = self._setup_logger(log_level)

        # Default preprocessing pipeline
        self.preprocessing_steps = preprocessing_steps or [
            {'name': 'hu_window', 'params': {'window_width': 500, 'window_level': 50}},
            {'name': 'normalize', 'params': {'min_bound': -100, 'max_bound': 400}},
            {'name': 'resample', 'params': {'target_spacing': (1.0, 1.0, 1.0)}},
            {'name': 'kidney_region_crop', 'params': {'margin': 30}}
        ]

        # Memory management settings
        self.memory_threshold = 100000000  # ~100M voxels
        self.default_chunk_size = 50

        self.logger.info("Initialized KidneyTumorPreprocessor")
        self.logger.info(f"Preprocessing steps: {self.preprocessing_steps}")

    def _setup_logger(self, log_level):
        """Set up and configure logger."""
        logger = logging.getLogger("KidneyTumorPreprocessor")
        logger.setLevel(log_level)

        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()

        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Add file handler
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.output_dir, "logs", "preprocessing.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def hu_window(self, image, window_width=500, window_level=50):
        """
        Apply HU windowing to CT image with memory efficiency.

        Args:
            image: Input image array
            window_width: Width of HU window
            window_level: Center level of HU window

        Returns:
            Windowed image
        """
        min_hu = window_level - window_width // 2
        max_hu = window_level + window_width // 2

        # Use float32 and in-place operations
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        np.clip(image, min_hu, max_hu, out=image)
        return image

    def normalize(self, image, min_bound=-100, max_bound=400):
        """
        Memory-efficient normalization to [-1, 1] range.

        Args:
            image: Input image array
            min_bound: Minimum intensity bound
            max_bound: Maximum intensity bound

        Returns:
            Normalized image
        """
        # Ensure we're using float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Calculate normalization factor once
        norm_factor = 2.0 / (max_bound - min_bound)

        # Process in chunks to reduce memory usage
        chunk_size = self.default_chunk_size
        for start_idx in range(0, image.shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, image.shape[0])

            # Get slice and process it with optimized operations
            chunk = image[start_idx:end_idx]
            np.clip(chunk, min_bound, max_bound, out=chunk)
            np.subtract(chunk, min_bound, out=chunk)
            np.multiply(chunk, norm_factor, out=chunk)
            np.subtract(chunk, 1.0, out=chunk)

        return image

    def kidney_region_crop(self, image, segmentation, margin=30):
        """
        Adaptive kidney region cropping with configurable margin.

        Args:
            image: Input image array
            segmentation: Segmentation mask
            margin: Margin around kidney region in voxels

        Returns:
            Tuple of cropped image and segmentation
        """
        # Get region containing kidneys and tumors
        kidney_tumor_mask = (segmentation > 0)

        if not np.any(kidney_tumor_mask):
            self.logger.warning("No kidney or tumor found in segmentation mask")
            return image, segmentation  # No kidney or tumor, return original

        # Find bounding box of non-zero regions
        indices = np.where(kidney_tumor_mask)
        min_z, max_z = np.min(indices[0]), np.max(indices[0])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])
        min_x, max_x = np.min(indices[2]), np.max(indices[2])

        # Add margin
        min_z = max(0, min_z - margin)
        max_z = min(image.shape[0], max_z + margin)
        min_y = max(0, min_y - margin)
        max_y = min(image.shape[1], max_y + margin)
        min_x = max(0, min_x - margin)
        max_x = min(image.shape[2], max_x + margin)

        # Crop image and segmentation
        cropped_image = image[min_z:max_z, min_y:max_y, min_x:max_x].copy()
        cropped_segmentation = segmentation[min_z:max_z, min_y:max_y, min_x:max_x].copy()

        # Log cropping information
        original_size = np.prod(image.shape) * 4 / (1024 ** 2)  # Size in MB
        cropped_size = np.prod(cropped_image.shape) * 4 / (1024 ** 2)  # Size in MB
        reduction = (1 - cropped_size / original_size) * 100

        self.logger.debug(f"Cropped from {image.shape} to {cropped_image.shape}")
        self.logger.debug(f"Memory reduction: {original_size:.1f}MB to {cropped_size:.1f}MB ({reduction:.1f}%)")

        return cropped_image, cropped_segmentation

    def resample(self, image, segmentation, spacing, target_spacing=(1.0, 1.0, 1.0)):
        """
        Memory-efficient resampling to target voxel spacing.

        Args:
            image: Input image array
            segmentation: Segmentation mask
            spacing: Original voxel spacing
            target_spacing: Target voxel spacing

        Returns:
            Tuple of resampled image and segmentation
        """
        # Skip if current spacing is already close to target
        if np.allclose(spacing, target_spacing, rtol=0.05):
            self.logger.debug(f"Skipping resampling as spacing {spacing} is already close to target {target_spacing}")
            return image, segmentation

        # Calculate scale factors
        scale_factors = np.array(spacing) / np.array(target_spacing)

        # Calculate new dimensions
        new_shape = np.round(np.array(image.shape) * scale_factors).astype(int)

        # Check if volume is too large for a direct resize
        total_voxels = np.prod(new_shape)
        is_large_volume = total_voxels > self.memory_threshold

        self.logger.info(f"Resampling volume from {image.shape} to {new_shape}")
        if is_large_volume:
            self.logger.info(f"Large volume detected: {new_shape}, using chunked processing")

            # Process image in chunks
            result_img = np.zeros(new_shape, dtype=np.float32)
            chunk_size = self.default_chunk_size

            for start_idx in range(0, image.shape[0], chunk_size):
                end_idx = min(start_idx + chunk_size, image.shape[0])

                # Calculate corresponding region in resampled image
                new_start = int(start_idx * scale_factors[0])
                new_end = int(end_idx * scale_factors[0])

                # Resize this chunk
                chunk = image[start_idx:end_idx]
                chunk_shape = (new_end - new_start, new_shape[1], new_shape[2])
                resized_chunk = resize(
                    chunk,
                    chunk_shape,
                    order=3,
                    mode='constant',
                    anti_aliasing=True,
                    preserve_range=True
                ).astype(np.float32)

                # Insert into result
                result_img[new_start:new_end] = resized_chunk

            resampled_image = result_img

            # Handle segmentation (use nearest-neighbor for labels)
            resampled_segmentation = resize(
                segmentation,
                new_shape,
                order=0,  # Nearest neighbor for labels
                mode='constant',
                preserve_range=True
            ).astype(segmentation.dtype)

        else:
            # For smaller volumes, resize in one go
            resampled_image = resize(
                image,
                new_shape,
                order=3,  # Cubic interpolation for image
                mode='constant',
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)

            resampled_segmentation = resize(
                segmentation,
                new_shape,
                order=0,  # Nearest neighbor for labels
                mode='constant',
                preserve_range=True
            ).astype(segmentation.dtype)

        return resampled_image, resampled_segmentation

    def preprocess_case(self, case_id, case_dir, save=True):
        """
        Preprocess a single case with memory optimization.

        Args:
            case_id: Case identifier
            case_dir: Directory containing case data
            save: Whether to save preprocessed results

        Returns:
            Dictionary with preprocessed data or None if failed
        """
        start_time = time.time()
        self.logger.info(f"Starting preprocessing for {case_id}")

        # Check if already processed
        case_output_dir = os.path.join(self.output_dir, case_id)
        if save and os.path.exists(case_output_dir):
            img_path = os.path.join(case_output_dir, "imaging_preprocessed.nii.gz")
            seg_path = os.path.join(case_output_dir, "segmentation_preprocessed.nii.gz")
            if os.path.exists(img_path) and os.path.exists(seg_path):
                self.logger.info(f"Case {case_id} already preprocessed, skipping")
                try:
                    # Load the preprocessed data to return
                    img_nii = nib.load(img_path)
                    seg_nii = nib.load(seg_path)

                    # Get original data for metadata
                    original_img_path = self._find_imaging_file(case_dir)
                    if original_img_path:
                        original_nii = nib.load(original_img_path)
                        original_shape = original_nii.shape
                        original_spacing = original_nii.header.get_zooms()
                    else:
                        # Default values if original can't be found
                        original_shape = img_nii.shape
                        original_spacing = img_nii.header.get_zooms()

                    return {
                        'case_id': case_id,
                        'image': img_nii.get_fdata().astype(np.float32),
                        'segmentation': seg_nii.get_fdata().astype(np.int16),
                        'original_shape': original_shape,
                        'original_spacing': original_spacing
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to load preprocessed data for {case_id}, reprocessing: {e}")
                    # Continue with normal processing

        # Read segmentation label
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")

        try:
            seg_nii = nib.load(seg_path)
            seg_data = seg_nii.get_fdata().astype(np.int16)  # Use int16 for segmentation
        except FileNotFoundError:
            self.logger.error(f"Segmentation file not found for {case_id}: {seg_path}")
            return None
        except Exception as e:
            self.logger.error(f"Cannot load segmentation file for {case_id}: {str(e)}")
            return None

        # Look for imaging file
        img_path = self._find_imaging_file(case_dir)

        if img_path is None:
            self.logger.error(f"Cannot find suitable imaging file for {case_id}")
            return None

        try:
            img_nii = nib.load(img_path)
            # Load as float32 directly to save memory
            img_data = img_nii.get_fdata().astype(np.float32)

            # Check for excessive size
            if np.prod(img_data.shape) > self.memory_threshold:
                self.logger.warning(f"Large volume detected in {case_id}: {img_data.shape}")
        except Exception as e:
            self.logger.error(f"Cannot load imaging file {img_path} for {case_id}: {str(e)}")
            return None

        # Get spacing information
        spacing = img_nii.header.get_zooms()
        original_shape = img_data.shape

        # Execute preprocessing steps dynamically based on configuration
        try:
            for step in self.preprocessing_steps:
                step_name = step['name']
                step_params = step['params']

                step_start_time = time.time()
                self.logger.debug(f"Applying {step_name} with params {step_params}")

                if step_name == 'hu_window':
                    img_data = self.hu_window(img_data, **step_params)
                elif step_name == 'normalize':
                    img_data = self.normalize(img_data, **step_params)
                elif step_name == 'resample':
                    img_data, seg_data = self.resample(img_data, seg_data, spacing, **step_params)
                elif step_name == 'kidney_region_crop':
                    img_data, seg_data = self.kidney_region_crop(img_data, seg_data, **step_params)
                else:
                    self.logger.warning(f"Unknown preprocessing step: {step_name}")

                step_time = time.time() - step_start_time
                self.logger.debug(f"Step {step_name} completed in {step_time:.2f}s")

        except MemoryError:
            self.logger.error(f"Memory error during preprocessing of {case_id}. Try reducing chunk size.")
            return None
        except Exception as e:
            self.logger.error(f"Error during preprocessing of {case_id}: {str(e)}")
            return None

        # Save preprocessed data if required
        if save:
            case_output_dir = os.path.join(self.output_dir, case_id)
            os.makedirs(case_output_dir, exist_ok=True)

            try:
                # Create new NIfTI files
                preprocessed_img = nib.Nifti1Image(img_data, img_nii.affine)
                preprocessed_seg = nib.Nifti1Image(seg_data, seg_nii.affine)

                # Save preprocessed files
                nib.save(preprocessed_img, os.path.join(case_output_dir, "imaging_preprocessed.nii.gz"))
                nib.save(preprocessed_seg, os.path.join(case_output_dir, "segmentation_preprocessed.nii.gz"))

                # Save metadata
                metadata = {
                    'case_id': case_id,
                    'original_shape': list(map(int, original_shape)),
                    'preprocessed_shape': list(map(int, img_data.shape)),
                    'original_spacing': list(map(float, spacing)),
                    'preprocessing_steps': self.preprocessing_steps,
                    'memory_usage_mb': float(np.prod(img_data.shape) * 4 / (1024 ** 2)),
                    'processing_time_sec': float(time.time() - start_time)
                }

                with open(os.path.join(case_output_dir, "metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)

                self.logger.info(f"Saved preprocessed files for {case_id}")
            except Exception as e:
                self.logger.error(f"Error saving preprocessed files for {case_id}: {str(e)}")

        processing_time = time.time() - start_time
        self.logger.info(f"Completed preprocessing for {case_id} in {processing_time:.2f}s")

        return {
            'case_id': case_id,
            'image': img_data,
            'segmentation': seg_data,
            'original_shape': original_shape,
            'original_spacing': spacing,
            'processing_time': processing_time
        }

    def _find_imaging_file(self, case_dir):
        """Find the appropriate imaging file for a case."""
        # First try the main imaging file
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        if os.path.exists(img_path):
            return img_path

        # If not found, look in instances folder
        instances_dir = os.path.join(case_dir, "instances")
        if os.path.exists(instances_dir):
            # First try to find kidney instance annotations
            kidney_files = [f for f in os.listdir(instances_dir)
                            if f.startswith("kidney_instance-1_annotation-1")]
            if kidney_files:
                return os.path.join(instances_dir, kidney_files[0])

            # If no kidney annotations, try to use any available file
            files = os.listdir(instances_dir)
            if files:
                return os.path.join(instances_dir, files[0])

        return None

    def preprocess_dataset(self, dataset_root, num_workers=None, case_ids=None):
        """
        Preprocess the entire dataset with optional parallel processing.

        Args:
            dataset_root: Root directory containing cases
            num_workers: Number of parallel workers (None=no parallelization, 0=auto)
            case_ids: Optional list of specific case IDs to process

        Returns:
            List of processed case data
        """
        # Determine available cores if auto mode is selected
        if num_workers == 0:
            num_workers = max(1, mp.cpu_count() - 1)

        if case_ids is None:
            # Find all case directories
            case_ids = []
            try:
                for item in os.listdir(dataset_root):
                    item_path = os.path.join(dataset_root, item)
                    if os.path.isdir(item_path) and item.startswith("case_"):
                        case_ids.append(item)
                case_ids.sort()
            except Exception as e:
                self.logger.error(f"Error scanning dataset directory {dataset_root}: {e}")
                return []

        if not case_ids:
            self.logger.error(f"No valid case directories found in {dataset_root}")
            return []

        self.logger.info(f"Starting preprocessing for {len(case_ids)} cases")

        # Save preprocessing configuration
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'preprocessing_steps': self.preprocessing_steps,
            'memory_threshold': self.memory_threshold,
            'default_chunk_size': self.default_chunk_size,
            'num_workers': num_workers,
            'dataset_path': dataset_root,
            'cases': case_ids
        }

        with open(os.path.join(self.output_dir, "preprocessing_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        processed_cases = []
        successful = 0
        total_start_time = time.time()

        # Define process function
        def process_case_wrapper(case_id):
            case_dir = os.path.join(dataset_root, case_id)
            try:
                result = self.preprocess_case(case_id, case_dir)
                if result:
                    return result
            except Exception as e:
                self.logger.error(f"Failed to process {case_id}: {str(e)}")
            return None

        # Parallel processing if requested
        if num_workers and num_workers > 1:
            self.logger.info(f"Using parallel processing with {num_workers} workers")
            try:
                with mp.Pool(processes=num_workers) as pool:
                    results = pool.map(process_case_wrapper, case_ids)
                processed_cases = [result for result in results if result is not None]
                successful = len(processed_cases)
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {str(e)}")
                # Fall back to sequential processing
                self.logger.info("Falling back to sequential processing")
                for case_id in case_ids:
                    result = process_case_wrapper(case_id)
                    if result:
                        processed_cases.append(result)
                        successful += 1
        else:
            # Sequential processing
            self.logger.info("Using sequential processing")
            for case_id in case_ids:
                result = process_case_wrapper(case_id)
                if result:
                    processed_cases.append(result)
                    successful += 1

        total_time = time.time() - total_start_time
        self.logger.info(
            f"Completed preprocessing {successful}/{len(case_ids)} cases successfully in {total_time:.2f}s")

        # Calculate and save dataset statistics
        if processed_cases:
            stats = self.get_dataset_statistics(processed_cases)
            with open(os.path.join(self.output_dir, "dataset_statistics.json"), 'w') as f:
                json.dump(stats, f, indent=2)

        return processed_cases

    def get_dataset_statistics(self, cases):
        """计算数据集的统计信息"""
        print("计算数据集统计信息...")
        
        # 初始化统计信息
        stats = {
            'image': {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            },
            'kidney': {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            },
            'tumor': {
                'mean': 0.0,
                'std': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            }
        }
        
        # 使用更小的采样大小
        sample_size = 10000
        all_samples = []
        
        # 收集样本
        for case in cases:
            # 对图像进行随机采样
            image = case['image']
            mask = case['segmentation']
            
            # 计算采样步长
            step = max(1, image.size // sample_size)
            
            # 使用更高效的方式采样
            image_samples = image[::step, ::step, ::step].flatten()
            mask_samples = mask[::step, ::step, ::step].flatten()
            
            # 分别处理肾脏和肿瘤区域
            kidney_mask = (mask_samples == 1)
            tumor_mask = (mask_samples == 2)
            
            # 收集样本
            all_samples.append({
                'image': image_samples,
                'kidney': image_samples[kidney_mask],
                'tumor': image_samples[tumor_mask]
            })
            
            # 更新全局最大最小值
            stats['image']['min'] = min(stats['image']['min'], float(image_samples.min()))
            stats['image']['max'] = max(stats['image']['max'], float(image_samples.max()))
            
            if len(image_samples[kidney_mask]) > 0:
                stats['kidney']['min'] = min(stats['kidney']['min'], float(image_samples[kidney_mask].min()))
                stats['kidney']['max'] = max(stats['kidney']['max'], float(image_samples[kidney_mask].max()))
            
            if len(image_samples[tumor_mask]) > 0:
                stats['tumor']['min'] = min(stats['tumor']['min'], float(image_samples[tumor_mask].min()))
                stats['tumor']['max'] = max(stats['tumor']['max'], float(image_samples[tumor_mask].max()))
        
        # 计算统计信息
        for region in ['image', 'kidney', 'tumor']:
            # 合并所有样本
            combined_samples = np.concatenate([s[region] for s in all_samples])
            
            if len(combined_samples) > 0:
                stats[region]['mean'] = float(np.mean(combined_samples))
                stats[region]['std'] = float(np.std(combined_samples))
        
        print("数据集统计信息计算完成")
        return stats