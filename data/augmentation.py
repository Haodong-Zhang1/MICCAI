# kits23_segmentation/data/augmentation.py
import numpy as np
import random
import torch
from monai.transforms import (
    Compose, RandRotate90, RandShiftIntensity, RandGaussianNoise,
    RandGaussianSmooth, RandAdjustContrast, RandFlip, Spacing,
    RandScaleIntensity, SpatialPad, RandSpatialCrop, RandAffine
)


class AugmentationPipeline:
    """
    Configurable augmentation pipeline for kidney tumor segmentation.
    Implements a combination of spatial and intensity augmentations.
    """

    def __init__(self, spatial_transforms=True, intensity_transforms=True,
                 deformation_transforms=True, prob=0.5):
        """
        Initialize the augmentation pipeline.

        Args:
            spatial_transforms: Whether to include spatial transforms
            intensity_transforms: Whether to include intensity transforms
            deformation_transforms: Whether to include deformation transforms
            prob: Probability of applying each transform
        """
        self.spatial_transforms = spatial_transforms
        self.intensity_transforms = intensity_transforms
        self.deformation_transforms = deformation_transforms
        self.prob = prob

        # Create transform pipeline
        self.transform = self._create_transform_pipeline()

    def _create_transform_pipeline(self):
        """Create a compose of all the transforms."""
        transforms = []

        # Spatial transforms
        if self.spatial_transforms:
            transforms.extend([
                RandFlip(prob=self.prob, spatial_axis=0),
                RandFlip(prob=self.prob, spatial_axis=1),
                RandFlip(prob=self.prob, spatial_axis=2),
                RandRotate90(prob=self.prob, max_k=3, spatial_axes=(0, 1)),
                RandRotate90(prob=self.prob, max_k=3, spatial_axes=(1, 2)),
                RandRotate90(prob=self.prob, max_k=3, spatial_axes=(0, 2))
            ])

        # Intensity transforms
        if self.intensity_transforms:
            transforms.extend([
                RandScaleIntensity(prob=self.prob, factors=0.15),
                RandShiftIntensity(prob=self.prob, offsets=0.1),
                RandGaussianNoise(prob=self.prob, mean=0.0, std=0.1),
                RandGaussianSmooth(prob=self.prob, sigma_x=(0.5, 1.15)),
                RandAdjustContrast(prob=self.prob, gamma=(0.9, 1.1))
            ])

        # Deformation transforms
        if self.deformation_transforms:
            transforms.append(
                RandAffine(
                    prob=self.prob,
                    scale_range=(0.85, 1.15),
                    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                    translate_range=(5, 5, 5),
                    padding_mode='zeros'
                )
            )

        return Compose(transforms)

    def __call__(self, image, mask):
        """Apply the transform to image and mask."""
        # Convert inputs to expected format
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension

        # Apply transforms
        transformed = self.transform({'image': image, 'mask': mask})

        # Convert back to numpy if necessary
        image_out = transformed['image'].squeeze(0).numpy() if isinstance(image, np.ndarray) else transformed['image']
        mask_out = transformed['mask'].squeeze(0).numpy() if isinstance(mask, np.ndarray) else transformed['mask']

        return {'image': image_out, 'mask': mask_out}


class TumorCentricAugmentation:
    """
    Specialized augmentation that focuses on tumor regions,
    providing more aggressive augmentation for small tumors.
    """

    def __init__(self, patch_size=(128, 128, 64), tumor_prob=0.8,
                 tumor_min_size=10, standard_pipeline=None):
        """
        Initialize the tumor-centric augmentation pipeline.

        Args:
            patch_size: Size of patches to extract
            tumor_prob: Probability of centering on a tumor
            tumor_min_size: Minimum size (voxels) of tumors to consider
            standard_pipeline: Standard augmentation pipeline to use
        """
        self.patch_size = patch_size
        self.tumor_prob = tumor_prob
        self.tumor_min_size = tumor_min_size
        self.standard_pipeline = standard_pipeline or AugmentationPipeline()

        # Create specialized tumor-centric transforms
        self.tumor_transforms = self._create_tumor_transforms()

    def _create_tumor_transforms(self):
        """Create specialized transforms for tumor regions."""
        return Compose([
            # More aggressive intensity transforms for tumor enhancement
            RandScaleIntensity(prob=0.7, factors=0.2),
            RandShiftIntensity(prob=0.7, offsets=0.15),
            RandGaussianNoise(prob=0.6, mean=0.0, std=0.15),
            RandGaussianSmooth(prob=0.5, sigma_x=(0.5, 1.5)),
            RandAdjustContrast(prob=0.7, gamma=(0.85, 1.15))
        ])

    def _extract_tumor_patch(self, image, mask):
        """Extract a patch centered on a tumor region."""
        # Find tumor voxels (label == 2 for tumors)
        tumor_voxels = np.argwhere(mask == 2)

        if len(tumor_voxels) < self.tumor_min_size:
            # If tumor is too small, fall back to standard random patch
            return self._extract_random_patch(image, mask)

        # Select a random tumor voxel as center
        center_idx = random.choice(tumor_voxels)
        d, h, w = center_idx

        # Extract patch using the tumor voxel as center
        pd, ph, pw = self.patch_size
        d_min = max(0, d - pd // 2)
        d_max = min(image.shape[0], d + pd // 2)
        h_min = max(0, h - ph // 2)
        h_max = min(image.shape[1], h + ph // 2)
        w_min = max(0, w - pw // 2)
        w_max = min(image.shape[2], w + pw // 2)

        patch_image = image[d_min:d_max, h_min:h_max, w_min:w_max]
        patch_mask = mask[d_min:d_max, h_min:h_max, w_min:w_max]

        # Pad if necessary
        if patch_image.shape != self.patch_size:
            pad_d = max(0, pd - patch_image.shape[0])
            pad_h = max(0, ph - patch_image.shape[1])
            pad_w = max(0, pw - patch_image.shape[2])

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
                patch_image = np.pad(patch_image, pad_width, mode='constant', constant_values=patch_image.min())
                patch_mask = np.pad(patch_mask, pad_width, mode='constant', constant_values=0)

        return patch_image, patch_mask

    def _extract_random_patch(self, image, mask):
        """Extract a random patch."""
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
                mask = np.pad(mask, pad_width, mode='constant', constant_values=0)

                # Update dimensions
                d, h, w = image.shape

        # Get random center point for the patch
        center_d = random.randint(pd // 2, d - pd // 2) if d > pd else d // 2
        center_h = random.randint(ph // 2, h - ph // 2) if h > ph else h // 2
        center_w = random.randint(pw // 2, w - pw // 2) if w > pw else w // 2

        # Extract patch
        d_min = max(0, center_d - pd // 2)
        d_max = min(d, center_d + pd // 2)
        h_min = max(0, center_h - ph // 2)
        h_max = min(h, center_h + ph // 2)
        w_min = max(0, center_w - pw // 2)
        w_max = min(w, center_w + pw // 2)

        patch_image = image[d_min:d_max, h_min:h_max, w_min:w_max]
        patch_mask = mask[d_min:d_max, h_min:h_max, w_min:w_max]

        # Ensure patch size is correct
        if patch_image.shape != self.patch_size:
            pad_d = max(0, pd - patch_image.shape[0])
            pad_h = max(0, ph - patch_image.shape[1])
            pad_w = max(0, pw - patch_image.shape[2])

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                pad_width = ((0, pad_d), (0, pad_h), (0, pad_w))
                patch_image = np.pad(patch_image, pad_width, mode='constant', constant_values=patch_image.min())
                patch_mask = np.pad(patch_mask, pad_width, mode='constant', constant_values=0)

        return patch_image, patch_mask

    def __call__(self, image, mask):
        """Apply tumor-centric augmentation."""
        # Determine whether to focus on tumor
        if random.random() < self.tumor_prob and np.any(mask == 2):
            # Extract tumor-centric patch
            patch_image, patch_mask = self._extract_tumor_patch(image, mask)

            # Apply tumor-specific transforms
            transformed = self.tumor_transforms({'image': patch_image, 'mask': patch_mask})
            patch_image = transformed['image']
            patch_mask = transformed['mask']
        else:
            # Extract random patch and apply standard augmentation
            patch_image, patch_mask = self._extract_random_patch(image, mask)
            transformed = self.standard_pipeline(patch_image, patch_mask)
            patch_image = transformed['image']
            patch_mask = transformed['mask']

        return {'image': patch_image, 'mask': patch_mask}


def get_augmentation_pipeline(config=None):
    """
    Create an augmentation pipeline based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured augmentation pipeline
    """
    if config is None:
        config = {
            'use_tumor_centric': True,
            'patch_size': (128, 128, 64),
            'tumor_prob': 0.8,
            'spatial_transforms': True,
            'intensity_transforms': True,
            'deformation_transforms': True,
            'transform_prob': 0.5
        }

    # Create standard pipeline
    standard_pipeline = AugmentationPipeline(
        spatial_transforms=config.get('spatial_transforms', True),
        intensity_transforms=config.get('intensity_transforms', True),
        deformation_transforms=config.get('deformation_transforms', True),
        prob=config.get('transform_prob', 0.5)
    )

    # Create tumor-centric pipeline if requested
    if config.get('use_tumor_centric', True):
        return TumorCentricAugmentation(
            patch_size=config.get('patch_size', (128, 128, 64)),
            tumor_prob=config.get('tumor_prob', 0.8),
            standard_pipeline=standard_pipeline
        )

    return standard_pipeline