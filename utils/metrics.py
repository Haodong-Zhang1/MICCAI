# kits23_segmentation/utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_coefficient(y_pred, y_true, smooth=1e-5):
    """
    Calculate Dice coefficient.

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient value
    """
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice


def hausdorff_distance_95(y_pred, y_true):
    """
    Calculate 95th percentile Hausdorff distance.
    Implementation using distance transforms for efficiency.

    Args:
        y_pred: Predicted segmentation (binary)
        y_true: Ground truth segmentation (binary)

    Returns:
        95th percentile Hausdorff distance
    """
    # Convert tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    # Ensure binary masks
    y_pred = (y_pred > 0.5).astype(np.bool)
    y_true = (y_true > 0.5).astype(np.bool)

    # If either mask is empty, return maximum distance
    if not np.any(y_pred) or not np.any(y_true):
        return float('inf')

    # Calculate distance transforms
    dt_pred = distance_transform_edt(~y_pred)
    dt_true = distance_transform_edt(~y_true)

    # Get distances
    hausdorff_pred = dt_true[y_pred]
    hausdorff_true = dt_pred[y_true]

    # Calculate 95th percentile
    return max(
        np.percentile(hausdorff_pred, 95) if hausdorff_pred.size > 0 else 0,
        np.percentile(hausdorff_true, 95) if hausdorff_true.size > 0 else 0
    )


def normalized_surface_distance(y_pred, y_true, threshold=1.0):
    """
    Calculate Normalized Surface Distance.

    Args:
        y_pred: Predicted segmentation (binary)
        y_true: Ground truth segmentation (binary)
        threshold: Distance threshold

    Returns:
        Normalized Surface Distance value
    """
    # Convert tensors to numpy if needed
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    # Ensure binary masks
    y_pred = (y_pred > 0.5).astype(np.bool)
    y_true = (y_true > 0.5).astype(np.bool)

    # Handle empty masks
    if not np.any(y_pred) and not np.any(y_true):
        return 1.0  # Perfect match if both are empty
    elif not np.any(y_pred) or not np.any(y_true):
        return 0.0  # No match if only one is empty

    # Calculate distance transforms for boundaries
    dt_pred = distance_transform_edt(~y_pred)
    dt_true = distance_transform_edt(~y_true)

    # Get boundary voxels
    boundary_pred = np.logical_and(y_pred, np.logical_not(
        binary_erosion(y_pred, structure=np.ones((3, 3, 3)))
    ))
    boundary_true = np.logical_and(y_true, np.logical_not(
        binary_erosion(y_true, structure=np.ones((3, 3, 3)))
    ))

    # Get surface distances
    surface_dist_pred = dt_true[boundary_pred]
    surface_dist_true = dt_pred[boundary_true]

    # Calculate normalized distances
    pred_overlap = np.sum(surface_dist_pred <= threshold) / max(np.sum(boundary_pred), 1)
    true_overlap = np.sum(surface_dist_true <= threshold) / max(np.sum(boundary_true), 1)

    # Return average of both directions
    return (pred_overlap + true_overlap) / 2


def binary_erosion(binary_mask, structure=None):
    """
    Simple binary erosion implementation.

    Args:
        binary_mask: Binary mask to erode
        structure: Structuring element

    Returns:
        Eroded binary mask
    """
    from scipy.ndimage import binary_erosion as scipy_binary_erosion
    return scipy_binary_erosion(binary_mask, structure=structure)


def calculate_metrics(logits, targets, include_background=False):
    """
    Calculate multiple segmentation metrics for kidney tumor segmentation.

    Args:
        logits: Model predictions (B, C, D, H, W)
        targets: Ground truth labels (B, D, H, W)
        include_background: Whether to include background class in metrics

    Returns:
        Dictionary of metrics
    """
    # Determine number of classes and starting index (skip background if not included)
    num_classes = logits.shape[1]
    start_class = 0 if include_background else 1

    # Convert logits to probabilities and get segmentation maps
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    # Initialize metrics dictionary
    metrics = {}

    # Calculate metrics for each class
    for class_idx in range(start_class, num_classes):
        # Get binary masks for current class
        pred_mask = (preds == class_idx).float()
        target_mask = (targets == class_idx).float()

        # Class name for metrics
        if class_idx == 0:
            class_name = 'background'
        elif class_idx == 1:
            class_name = 'kidney'
        elif class_idx == 2:
            class_name = 'tumor'
        else:
            class_name = f'class_{class_idx}'

        # Calculate Dice coefficient
        dice = dice_coefficient(pred_mask, target_mask).item()
        metrics[f'dice_{class_name}'] = dice

        # Calculate additional metrics if any mask has positive values
        if torch.any(pred_mask) or torch.any(target_mask):
            # Move to CPU for more complex metrics
            pred_mask_np = pred_mask.cpu().numpy()
            target_mask_np = target_mask.cpu().numpy()

            # Calculate Hausdorff distance (95th percentile)
            hausdorff = hausdorff_distance_95(pred_mask_np, target_mask_np)
            metrics[f'hausdorff95_{class_name}'] = hausdorff

            # Calculate normalized surface distance
            nsd = normalized_surface_distance(pred_mask_np, target_mask_np)
            metrics[f'surface_dice_{class_name}'] = nsd

    # Calculate composite metrics
    if 'dice_kidney' in metrics and 'dice_tumor' in metrics:
        metrics['mean_dice'] = (metrics['dice_kidney'] + metrics['dice_tumor']) / 2.0

    if 'surface_dice_kidney' in metrics and 'surface_dice_tumor' in metrics:
        metrics['mean_surface_dice'] = (metrics['surface_dice_kidney'] + metrics['surface_dice_tumor']) / 2.0

    return metrics