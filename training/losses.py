# kits23_segmentation/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """

    def __init__(self, weight=None, smooth=1e-5):
        super().__init__()
        self.weight = weight  # Class weights
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W)

        Returns:
            Dice loss value
        """
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]

        # Convert to one-hot encoding
        target_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        # Apply softmax to get class probabilities
        probs = F.softmax(logits, dim=1)

        # Calculate dice coefficient for each class
        intersection = (probs * target_one_hot).sum(dim=(0, 2, 3, 4))
        cardinality = (probs + target_one_hot).sum(dim=(0, 2, 3, 4))

        dice_coef = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Apply class weights if provided
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                self.weight = torch.tensor(self.weight, device=logits.device)
            dice_coef = dice_coef * self.weight
            dice_loss = 1.0 - dice_coef.sum() / self.weight.sum()
        else:
            dice_loss = 1.0 - dice_coef.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W)

        Returns:
            Focal loss value
        """
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]

        # Convert to one-hot encoding
        target_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        # Apply softmax to get class probabilities
        probs = F.softmax(logits, dim=1)

        # Calculate focal loss
        pt = (target_one_hot * probs) + ((1 - target_one_hot) * (1 - probs))
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha balancing
        if self.alpha is not None:
            focal_weight = focal_weight * (self.alpha * target_one_hot + (1 - self.alpha) * (1 - target_one_hot))

        # Calculate cross entropy loss
        ce_loss = -torch.log(pt + 1e-7)

        # Apply focal weighting
        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.weight is not None:
            if not isinstance(self.weight, torch.Tensor):
                self.weight = torch.tensor(self.weight, device=logits.device)
            # Expand weights to match loss dimensions
            weight_expanded = self.weight.view(1, num_classes, 1, 1, 1).expand_as(loss)
            loss = loss * weight_expanded

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary loss to emphasize segmentation at object boundaries.
    """

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.laplacian_kernel = torch.tensor([
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W)

        Returns:
            Boundary loss value
        """
        num_classes = logits.shape[1]
        batch_size = logits.shape[0]
        device = logits.device

        # Create Laplacian kernel for boundary detection
        if self.laplacian_kernel.device != device:
            self.laplacian_kernel = self.laplacian_kernel.to(device)

        # Initialize loss
        boundary_loss = 0.0

        # Process each class separately
        for class_idx in range(1, num_classes):  # Skip background class
            # Get binary mask for current class
            target_mask = (targets == class_idx).float()
            pred_mask = F.softmax(logits, dim=1)[:, class_idx]

            # Detect boundaries using Laplacian filtering
            if batch_size > 1:
                # Process each batch item separately for 3D convolution
                target_boundaries = torch.zeros_like(target_mask)
                pred_boundaries = torch.zeros_like(pred_mask)

                for b in range(batch_size):
                    for d in range(target_mask.shape[1]):
                        # Process each depth slice
                        target_slice = target_mask[b, d:d + 1].unsqueeze(0)  # Add channel dim
                        pred_slice = pred_mask[b, d:d + 1].unsqueeze(0)  # Add channel dim

                        # Apply Laplacian filter to detect boundaries
                        target_boundary = F.conv2d(target_slice, self.laplacian_kernel, padding=1).abs()
                        pred_boundary = F.conv2d(pred_slice, self.laplacian_kernel, padding=1).abs()

                        target_boundaries[b, d] = target_boundary.squeeze()
                        pred_boundaries[b, d] = pred_boundary.squeeze()
            else:
                # Process single batch for efficiency
                target_boundaries = torch.zeros_like(target_mask)
                pred_boundaries = torch.zeros_like(pred_mask)

                for d in range(target_mask.shape[1]):
                    # Process each depth slice
                    target_slice = target_mask[0, d:d + 1].unsqueeze(0)  # Add batch and channel dims
                    pred_slice = pred_mask[0, d:d + 1].unsqueeze(0)  # Add batch and channel dims

                    # Apply Laplacian filter to detect boundaries
                    target_boundary = F.conv2d(target_slice, self.laplacian_kernel, padding=1).abs()
                    pred_boundary = F.conv2d(pred_slice, self.laplacian_kernel, padding=1).abs()

                    target_boundaries[0, d] = target_boundary.squeeze()
                    pred_boundaries[0, d] = pred_boundary.squeeze()

            # Binary boundary loss (MSE)
            class_boundary_loss = F.mse_loss(pred_boundaries, target_boundaries)
            boundary_loss += class_boundary_loss

        return self.weight * boundary_loss / (num_classes - 1)  # Average over classes


class CombinedLoss(nn.Module):
    """
    Combined loss function with weighted components for kidney tumor segmentation.
    """

    def __init__(self, dice_weight=1.0, focal_weight=0.5, boundary_weight=0.2,
                 class_weights=None, smooth=1e-5, focal_gamma=2.0):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        # Component losses
        self.dice_loss = DiceLoss(weight=class_weights, smooth=smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, weight=class_weights)
        self.boundary_loss = BoundaryLoss(weight=1.0)  # Internal weight set to 1.0

    def forward(self, logits, targets):
        """
        Args:
            logits: Model predictions (B, C, D, H, W)
            targets: Ground truth labels (B, D, H, W)

        Returns:
            Combined weighted loss value
        """
        # Calculate component losses
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        # Calculate boundary loss if weight > 0
        if self.boundary_weight > 0:
            boundary = self.boundary_loss(logits, targets)
        else:
            boundary = 0.0

        # Combine losses with weights
        combined_loss = (
                self.dice_weight * dice +
                self.focal_weight * focal +
                self.boundary_weight * boundary
        )

        return combined_loss, {
            'dice_loss': dice.item(),
            'focal_loss': focal.item(),
            'boundary_loss': boundary if isinstance(boundary, float) else boundary.item(),
            'total_loss': combined_loss.item()
        }


def get_loss_function(config=None):
    """
    Factory function to create a loss function based on configuration.

    Args:
        config: Dictionary containing loss configuration parameters

    Returns:
        Configured loss function
    """
    default_config = {
        'dice_weight': 1.0,
        'focal_weight': 0.5,
        'boundary_weight': 0.2,
        'class_weights': [0.1, 0.3, 0.6],  # Background, kidney, tumor
        'focal_gamma': 2.0,
        'smooth': 1e-5
    }

    if config is not None:
        default_config.update(config)

    return CombinedLoss(
        dice_weight=default_config['dice_weight'],
        focal_weight=default_config['focal_weight'],
        boundary_weight=default_config['boundary_weight'],
        class_weights=default_config['class_weights'],
        focal_gamma=default_config['focal_gamma'],
        smooth=default_config['smooth']
    )