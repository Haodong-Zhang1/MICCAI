# kits23_segmentation/data/dataset.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import json
import random
from pathlib import Path


class KidneyTumorDataset(Dataset):
    """
    Dataset class for kidney tumor segmentation with support for preprocessed data.
    Handles patch extraction and on-the-fly augmentation.
    """

    def __init__(self, data_dir, patch_size=(128, 128, 64), split='train', transforms=None,
                 cache_rate=0.0, use_preprocessed=True, seed=42):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the dataset
            patch_size: Size of patches to extract (D, H, W)
            split: Dataset split ('train', 'val', or 'test')
            transforms: Data augmentation transformations
            cache_rate: Percentage of data to cache in memory (0.0-1.0)
            use_preprocessed: Whether to use preprocessed data
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.split = split
        self.transforms = transforms
        self.cache_rate = cache_rate
        self.use_preprocessed = use_preprocessed
        self.seed = seed

        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Get case IDs from split file
        splits_file = os.path.join(data_dir, 'cv_splits.json')
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)

            if split in splits:
                self.case_ids = splits[split]
            else:
                raise ValueError(f"Split '{split}' not found in splits file")
        else:
            # If splits file doesn't exist, create a default split
            print(f"Splits file not found at {splits_file}. Creating default split.")
            all_cases = self._get_all_case_ids()

            # Shuffle cases with fixed seed for reproducibility
            random.Random(seed).shuffle(all_cases)

            # Create splits (70% train, 15% val, 15% test by default)
            n_train = int(0.7 * len(all_cases))
            n_val = int(0.15 * len(all_cases))

            splits = {
                'train': all_cases[:n_train],
                'val': all_cases[n_train:n_train + n_val],
                'test': all_cases[n_train + n_val:]
            }

            # Save splits for future use
            os.makedirs(os.path.dirname(splits_file), exist_ok=True)
            with open(splits_file, 'w') as f:
                json.dump(splits, f, indent=2)

            self.case_ids = splits[split]

        # Setup data loading
        self.data_list = []
        self.cached_data = {}

        # Prepare data list for actual loading
        for case_id in self.case_ids:
            self._add_case_to_list(case_id)

        # Cache data according to cache_rate
        if self.cache_rate > 0:
            num_to_cache = int(len(self.data_list) * self.cache_rate)
            for idx in range(num_to_cache):
                case_id = self.data_list[idx]['case_id']
                image, label = self._load_case_data(case_id)
                self.cached_data[case_id] = (image, label)

            print(f"Cached {num_to_cache} cases in memory")

    def _get_all_case_ids(self):
        """Get all available case IDs from the dataset directory."""
        if self.use_preprocessed:
            preprocessed_dir = os.path.join(self.data_dir, 'output', 'preprocessed_data')
            if os.path.exists(preprocessed_dir):
                return [d for d in os.listdir(preprocessed_dir)
                        if os.path.isdir(os.path.join(preprocessed_dir, d)) and d.startswith('case_')]

        # Fall back to raw data directory
        return [d for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('case_')]

    def _add_case_to_list(self, case_id):
        """Add a case to the data list with metadata."""
        case_data = {'case_id': case_id}

        # Get metadata if available
        if self.use_preprocessed:
            metadata_path = os.path.join(self.data_dir, 'output', 'preprocessed_data',
                                         case_id, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                case_data.update(metadata)

        self.data_list.append(case_data)

    def _load_case_data(self, case_id):
        """Load image and label data for a case."""
        if self.use_preprocessed:
            # Load preprocessed data
            preprocessed_dir = os.path.join(self.data_dir, 'output', 'preprocessed_data', case_id)
            img_path = os.path.join(preprocessed_dir, 'imaging_preprocessed.nii.gz')
            seg_path = os.path.join(preprocessed_dir, 'segmentation_preprocessed.nii.gz')
        else:
            # Load raw data
            case_dir = os.path.join(self.data_dir, case_id)
            img_path = self._find_imaging_file(case_dir)
            seg_path = os.path.join(case_dir, 'segmentation.nii.gz')

        # Load image and segmentation
        try:
            image_nii = nib.load(img_path)
            image = image_nii.get_fdata().astype(np.float32)

            seg_nii = nib.load(seg_path)
            label = seg_nii.get_fdata().astype(np.int8)

            return image, label
        except Exception as e:
            print(f"Error loading data for {case_id}: {e}")
            return None, None

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

    def _extract_random_patch(self, image, label):
        """Extract a random patch from the image and label."""
        # Get image dimensions
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # Handle case where image is smaller than patch size
        if d < pd or h < ph or w < pw:
            # Pad image and label if needed
            pad_d = max(0, pd - d)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pad_width = ((pad_d // 2, pad_d - pad_d // 2),
                             (pad_h // 2, pad_h - pad_h // 2),
                             (pad_w // 2, pad_w - pad_w // 2))
                image = np.pad(image, pad_width, mode='constant', constant_values=image.min())
                label = np.pad(label, pad_width, mode='constant', constant_values=0)

                # Update dimensions
                d, h, w = image.shape

        # Get random center point for the patch
        # Prioritize regions with kidney/tumor
        if np.any(label > 0) and random.random() < 0.8:  # 80% chance to center on kidney/tumor
            # Get indices of non-zero elements in the label
            indices = np.argwhere(label > 0)
            # Randomly select one of those indices as the center point
            center_idx = random.choice(indices)
            center_d, center_h, center_w = center_idx
        else:
            # Random center point
            center_d = random.randint(pd // 2, d - pd // 2) if d > pd else d // 2
            center_h = random.randint(ph // 2, h - ph // 2) if h > ph else h // 2
            center_w = random.randint(pw // 2, w - pw // 2) if w > pw else w // 2

        # Extract patch using the center point
        d_min = max(0, center_d - pd // 2)
        d_max = min(d, center_d + pd // 2 + pd % 2)
        h_min = max(0, center_h - ph // 2)
        h_max = min(h, center_h + ph // 2 + ph % 2)
        w_min = max(0, center_w - pw // 2)
        w_max = min(w, center_w + pw // 2 + pw % 2)

        patch_image = image[d_min:d_max, h_min:h_max, w_min:w_max]
        patch_label = label[d_min:d_max, h_min:h_max, w_min:w_max]

        # Ensure patch size is correct (in case of edge patches)
        if patch_image.shape != self.patch_size:
            # Additional padding if needed
            pad_d = max(0, pd - patch_image.shape[0])
            pad_h = max(0, ph - patch_image.shape[1])
            pad_w = max(0, pw - patch_image.shape[2])

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
                patch_image = np.pad(patch_image, pad_width, mode='constant', constant_values=patch_image.min())
                patch_label = np.pad(patch_label, pad_width, mode='constant', constant_values=0)

        return patch_image, patch_label

    def __len__(self):
        """Get the number of samples in the dataset."""
        if self.split == 'train':
            # Return a larger number for training to ensure sufficient random patches
            return len(self.data_list) * 50
        else:
            return len(self.data_list)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        if self.split == 'train':
            # For training, get a random case and extract a random patch
            case_idx = idx % len(self.data_list)
        else:
            # For validation/testing, iterate through cases
            case_idx = idx

        case_data = self.data_list[case_idx]
        case_id = case_data['case_id']

        # Try to get cached data first
        if case_id in self.cached_data:
            image, label = self.cached_data[case_id]
        else:
            image, label = self._load_case_data(case_id)

        if image is None or label is None:
            # If loading failed, return a zero patch
            image = np.zeros(self.patch_size, dtype=np.float32)
            label = np.zeros(self.patch_size, dtype=np.int8)
        else:
            # Extract patch
            image, label = self._extract_random_patch(image, label)

        # Convert to torch tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        label = torch.from_numpy(label).unsqueeze(0)  # Add channel dimension

        # Apply transforms if any
        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': label, 'case_id': case_id}


def get_data_loaders(data_dir, batch_size=2, patch_size=(128, 128, 64),
                    num_workers=4, transforms=None, use_preprocessed=True):
    """
    Create data loaders for training, validation, and testing.

    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        patch_size: Size of patches to extract (D, H, W)
        num_workers: Number of data loading workers
        transforms: Data augmentation transformations
        use_preprocessed: Whether to use preprocessed data

    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for each split
    """
    # Create datasets
    train_dataset = KidneyTumorDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        split='train',
        transforms=transforms,
        use_preprocessed=use_preprocessed
    )

    val_dataset = KidneyTumorDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        split='val',
        transforms=None,  # No transforms for validation
        use_preprocessed=use_preprocessed
    )

    test_dataset = KidneyTumorDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        split='test',
        transforms=None,  # No transforms for testing
        use_preprocessed=use_preprocessed
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader