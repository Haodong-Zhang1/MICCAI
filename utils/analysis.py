# kits23_segmentation/utils/analysis.py
import os
import pandas as pd
import numpy as np
import json
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split
from collections import defaultdict


def numpy_json_encoder(obj):
    """处理NumPy类型的JSON编码器"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def analyze_dataset(dataset_root, output_dir=None):
    """
    Analyze the KiTS23 dataset and collect metadata.

    Args:
        dataset_root: Path to the dataset root directory
        output_dir: Directory to save analysis results (optional)

    Returns:
        DataFrame containing case metadata
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Find all cases
    case_dirs = [d for d in os.listdir(dataset_root)
                 if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("case_")]
    case_dirs.sort()

    # Initialize metadata storage
    metadata = []

    # Analyze each case
    for case_id in case_dirs:
        print(f"Analyzing {case_id}...")
        case_dir = os.path.join(dataset_root, case_id)

        # Initialize case metadata
        case_metadata = {
            'case_id': case_id,
            'has_segmentation': False,
            'has_imaging': False,
            'kidney_volume': 0,
            'tumor_volume': 0,
            'image_shape': None,
            'spacing': None
        }

        # Check for segmentation file
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        if os.path.exists(seg_path):
            case_metadata['has_segmentation'] = True

            try:
                # Load segmentation
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()

                # Calculate volumes
                kidney_mask = (seg_data == 1)
                tumor_mask = (seg_data == 2)
                case_metadata['kidney_volume'] = np.sum(kidney_mask)
                case_metadata['tumor_volume'] = np.sum(tumor_mask)

                # Store spacing
                spacing = seg_nii.header.get_zooms()
                case_metadata['spacing'] = spacing
            except Exception as e:
                print(f"Error analyzing segmentation for {case_id}: {e}")

        # Check for imaging file
        img_path = os.path.join(case_dir, "imaging.nii.gz")
        if not os.path.exists(img_path):
            # Try to find in instances folder
            instances_dir = os.path.join(case_dir, "instances")
            if os.path.exists(instances_dir):
                kidney_files = [f for f in os.listdir(instances_dir)
                                if f.startswith("kidney_instance-1_annotation-1")]
                if kidney_files:
                    img_path = os.path.join(instances_dir, kidney_files[0])

        # Load imaging data if found
        if os.path.exists(img_path):
            case_metadata['has_imaging'] = True

            try:
                # Load image metadata
                img_nii = nib.load(img_path)
                case_metadata['image_shape'] = img_nii.shape

                # Add spacing if not already set
                if case_metadata['spacing'] is None:
                    case_metadata['spacing'] = img_nii.header.get_zooms()
            except Exception as e:
                print(f"Error analyzing imaging for {case_id}: {e}")

        # Add to metadata list
        metadata.append(case_metadata)

    # Convert to DataFrame
    metadata_df = pd.DataFrame(metadata)

    # Calculate derived metrics if possible
    if output_dir is not None:
        # Save metadata to CSV
        metadata_csv_path = os.path.join(output_dir, "dataset_metadata.csv")
        metadata_df.to_csv(metadata_csv_path, index=False)

        # Calculate and save dataset statistics
        stats = calculate_dataset_statistics(metadata_df)
        stats_path = os.path.join(output_dir, "dataset_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=numpy_json_encoder)

    return metadata_df


def calculate_dataset_statistics(metadata_df):
    """
    Calculate statistics from dataset metadata.

    Args:
        metadata_df: DataFrame containing dataset metadata

    Returns:
        Dictionary of dataset statistics
    """
    stats = {}

    # Count cases
    stats['total_cases'] = len(metadata_df)
    stats['cases_with_segmentation'] = metadata_df['has_segmentation'].sum()
    stats['cases_with_imaging'] = metadata_df['has_imaging'].sum()
    stats['cases_with_tumor'] = (metadata_df['tumor_volume'] > 0).sum()

    # Volume statistics (in voxels)
    stats['kidney_volume_stats'] = {
        'mean': float(metadata_df['kidney_volume'].mean()),
        'std': float(metadata_df['kidney_volume'].std()),
        'min': float(metadata_df['kidney_volume'].min()),
        'max': float(metadata_df['kidney_volume'].max()),
        'median': float(metadata_df['kidney_volume'].median())
    }

    # Tumor statistics for cases with tumor
    tumor_cases = metadata_df[metadata_df['tumor_volume'] > 0]
    if len(tumor_cases) > 0:
        stats['tumor_volume_stats'] = {
            'mean': float(tumor_cases['tumor_volume'].mean()),
            'std': float(tumor_cases['tumor_volume'].std()),
            'min': float(tumor_cases['tumor_volume'].min()),
            'max': float(tumor_cases['tumor_volume'].max()),
            'median': float(tumor_cases['tumor_volume'].median())
        }

        # Calculate tumor-to-kidney ratio
        tumor_cases['tumor_kidney_ratio'] = tumor_cases['tumor_volume'] / tumor_cases['kidney_volume']
        stats['tumor_kidney_ratio_stats'] = {
            'mean': float(tumor_cases['tumor_kidney_ratio'].mean()),
            'std': float(tumor_cases['tumor_kidney_ratio'].std()),
            'min': float(tumor_cases['tumor_kidney_ratio'].min()),
            'max': float(tumor_cases['tumor_kidney_ratio'].max()),
            'median': float(tumor_cases['tumor_kidney_ratio'].median())
        }

    # Image shape statistics
    if 'image_shape' in metadata_df.columns and metadata_df['image_shape'].notna().any():
        # Extract shape dimensions
        shapes = np.array([s for s in metadata_df['image_shape'].dropna()])
        if len(shapes) > 0:
            stats['image_shape_stats'] = {
                'mean': [float(s) for s in shapes.mean(axis=0)],
                'std': [float(s) for s in shapes.std(axis=0)],
                'min': [float(s) for s in shapes.min(axis=0)],
                'max': [float(s) for s in shapes.max(axis=0)]
            }

    # Spacing statistics
    if 'spacing' in metadata_df.columns and metadata_df['spacing'].notna().any():
        # Extract spacing values
        spacings = np.array([s for s in metadata_df['spacing'].dropna()])
        if len(spacings) > 0:
            stats['spacing_stats'] = {
                'mean': [float(s) for s in spacings.mean(axis=0)],
                'std': [float(s) for s in spacings.std(axis=0)],
                'min': [float(s) for s in spacings.min(axis=0)],
                'max': [float(s) for s in spacings.max(axis=0)]
            }

    return stats


def create_cv_splits(metadata_df, n_splits=5, test_size=0.15, stratify_by_tumor=True,
                     output_dir=None, random_state=42):
    """
    Create cross-validation splits for the dataset.

    Args:
        metadata_df: DataFrame containing dataset metadata
        n_splits: Number of cross-validation folds
        test_size: Proportion of data to reserve for testing
        stratify_by_tumor: Whether to stratify splits by tumor presence
        output_dir: Directory to save splits (optional)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing case IDs for each split
    """
    # Filter cases with both segmentation and imaging
    valid_cases = metadata_df[metadata_df['has_segmentation'] & metadata_df['has_imaging']]

    # Prepare stratification if requested
    if stratify_by_tumor:
        # Create tumor size categories for better stratification
        valid_cases['tumor_category'] = pd.cut(
            valid_cases['tumor_volume'],
            bins=[-1, 0, 100, 1000, 10000, float('inf')],
            labels=['no_tumor', 'very_small', 'small', 'medium', 'large']
        )
        stratify = valid_cases['tumor_category']
    else:
        stratify = None

    # Extract case IDs
    case_ids = valid_cases['case_id'].values

    # Split into training/validation and test sets
    train_val_ids, test_ids = train_test_split(
        case_ids,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )

    # Create K-fold cross-validation splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize splits
    cv_splits = {
        'test': test_ids.tolist()
    }

    # If stratifying by tumor, need to join back with metadata for each fold
    if stratify_by_tumor:
        train_val_df = valid_cases[valid_cases['case_id'].isin(train_val_ids)]
        fold_stratify = train_val_df['tumor_category'].values

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_ids, fold_stratify)):
            train_fold_ids = train_val_df.iloc[train_idx]['case_id'].values
            val_fold_ids = train_val_df.iloc[val_idx]['case_id'].values

            cv_splits[f'fold_{fold}_train'] = train_fold_ids.tolist()
            cv_splits[f'fold_{fold}_val'] = val_fold_ids.tolist()
    else:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_ids)):
            train_fold_ids = train_val_ids[train_idx]
            val_fold_ids = train_val_ids[val_idx]

            cv_splits[f'fold_{fold}_train'] = train_fold_ids.tolist()
            cv_splits[f'fold_{fold}_val'] = val_fold_ids.tolist()

    # Add combined train set for final testing
    cv_splits['train'] = train_val_ids.tolist()

    # Save splits if output directory provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        splits_path = os.path.join(output_dir, "cv_splits.json")
        with open(splits_path, 'w') as f:
            json.dump(cv_splits, f, indent=2)

        print(f"Cross-validation splits saved to {splits_path}")

    return cv_splits


def analyze_predictions(predictions_dir, ground_truth_dir=None, output_dir=None):
    """
    Analyze model predictions and compare with ground truth if available.

    Args:
        predictions_dir: Directory containing prediction files
        ground_truth_dir: Directory containing ground truth files (optional)
        output_dir: Directory to save analysis results (optional)

    Returns:
        DataFrame containing prediction analysis
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Find all prediction files
    case_dirs = [d for d in os.listdir(predictions_dir)
                 if os.path.isdir(os.path.join(predictions_dir, d))]
    case_dirs.sort()

    # Initialize results storage
    results = []

    for case_id in case_dirs:
        print(f"Analyzing predictions for {case_id}...")

        # Check for metrics file (precomputed metrics)
        metrics_path = os.path.join(predictions_dir, case_id, "metrics.json")

        case_result = {'case_id': case_id}

        if os.path.exists(metrics_path):
            # Load precomputed metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            # Add metrics to results
            case_result.update(metrics)
        else:
            # Load prediction
            pred_path = os.path.join(predictions_dir, case_id, "prediction.nii.gz")
            if not os.path.exists(pred_path):
                # Try numpy format
                pred_path = os.path.join(predictions_dir, case_id, "prediction.npy")

            if os.path.exists(pred_path):
                try:
                    # Load prediction
                    if pred_path.endswith('.npy'):
                        pred = np.load(pred_path)
                    else:
                        pred_nii = nib.load(pred_path)
                        pred = pred_nii.get_fdata()

                    # Calculate prediction statistics
                    case_result['kidney_volume'] = np.sum(pred == 1)
                    case_result['tumor_volume'] = np.sum(pred == 2)

                    # Compare with ground truth if available
                    if ground_truth_dir is not None:
                        gt_path = os.path.join(ground_truth_dir, case_id, "segmentation.nii.gz")

                        if os.path.exists(gt_path):
                            gt_nii = nib.load(gt_path)
                            gt = gt_nii.get_fdata()

                            # Calculate overlap metrics
                            from kits23_segmentation.utils.metrics import dice_coefficient, hausdorff_distance_95, \
                                normalized_surface_distance

                            # Kidney metrics
                            kidney_pred = (pred == 1)
                            kidney_gt = (gt == 1)
                            case_result['dice_kidney'] = dice_coefficient(kidney_pred, kidney_gt)
                            case_result['hausdorff95_kidney'] = hausdorff_distance_95(kidney_pred, kidney_gt)
                            case_result['surface_dice_kidney'] = normalized_surface_distance(kidney_pred, kidney_gt)

                            # Tumor metrics
                            tumor_pred = (pred == 2)
                            tumor_gt = (gt == 2)
                            case_result['dice_tumor'] = dice_coefficient(tumor_pred, tumor_gt)
                            case_result['hausdorff95_tumor'] = hausdorff_distance_95(tumor_pred, tumor_gt)
                            case_result['surface_dice_tumor'] = normalized_surface_distance(tumor_pred, tumor_gt)
                except Exception as e:
                    print(f"Error analyzing predictions for {case_id}: {e}")

        # Add to results
        results.append(case_result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate overall statistics
    if len(results_df) > 0:
        stats = defaultdict(dict)

        # Calculate statistics for each metric
        for col in results_df.columns:
            if col != 'case_id' and results_df[col].dtype in (float, int):
                stats[col] = {
                    'mean': float(results_df[col].mean()),
                    'std': float(results_df[col].std()),
                    'min': float(results_df[col].min()),
                    'max': float(results_df[col].max()),
                    'median': float(results_df[col].median())
                }

        # Save statistics if output directory provided
        if output_dir is not None:
            stats_path = os.path.join(output_dir, "prediction_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(dict(stats), f, indent=2)

            # Save results DataFrame
            results_path = os.path.join(output_dir, "prediction_results.csv")
            results_df.to_csv(results_path, index=False)

    return results_df