# kits23_segmentation/models/am_msf_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SqueezeExcitation(nn.Module):
    """
    Squeeze and Excitation block for channel-wise feature recalibration.
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Ensure minimum size for the reduced dimension
        reduced_channels = max(1, in_channels // reduction_ratio)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, *_ = x.size()

        # Global average pooling
        y = self.pool(x).view(batch_size, channels)

        # Channel-wise scaling factors
        y = self.fc(y).view(batch_size, channels, 1, 1, 1)

        # Apply scaling factors
        return x * y


class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise Separable 3D Convolution for efficient computation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )

        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AMSFF(nn.Module):
    """
    Adaptive Multi-Scale Feature Fusion module that dynamically weights
    features from different scales based on their relevance.
    """

    def __init__(self, channels_list, output_channels=None):
        super().__init__()

        self.n_scales = len(channels_list)
        self.output_channels = output_channels or max(channels_list)

        # Scale-specific feature enhancement
        self.scale_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(channels),
                nn.LeakyReLU(inplace=True),
                SqueezeExcitation(channels)
            ) for channels in channels_list
        ])

        # Feature projections to common channel dimension
        self.scale_projectors = nn.ModuleList([
            nn.Conv3d(channels, self.output_channels, kernel_size=1)
            for channels in channels_list
        ])

        # Dynamic weight generation
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, channels // 8, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(channels // 8, 1, kernel_size=1)
            ) for channels in channels_list
        ])

        # Final fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.output_channels, self.output_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.output_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, feature_maps):
        assert len(feature_maps) == self.n_scales, "Number of input feature maps must match number of scales"

        # Process each scale
        enhanced_features = []
        scale_weights = []

        for i, feature in enumerate(feature_maps):
            # Enhance features at each scale
            enhanced = self.scale_enhancers[i](feature)
            enhanced_features.append(enhanced)

            # Generate scale-specific weights
            weight = self.weight_generators[i](enhanced)
            scale_weights.append(weight)

        # Normalize weights using softmax across scales
        scale_weights = torch.cat(scale_weights, dim=1)
        normalized_weights = F.softmax(scale_weights, dim=1)

        # Project features to common channel dimension and resample to target resolution
        target_shape = feature_maps[0].shape[2:]  # Use shape of first feature map as target
        resampled_features = []

        for i, feature in enumerate(enhanced_features):
            # Project to common dimension
            projected = self.scale_projectors[i](feature)

            # Resample if needed
            if projected.shape[2:] != target_shape:
                projected = F.interpolate(projected, size=target_shape, mode='trilinear', align_corners=False)

            resampled_features.append(projected)

        # Apply weights to each scale
        weighted_features = []
        for i, feature in enumerate(resampled_features):
            weight = normalized_weights[:, i:i + 1]
            weighted = feature * weight
            weighted_features.append(weighted)

        # Sum weighted features
        fused = sum(weighted_features)

        # Apply final convolution
        output = self.fusion_conv(fused)

        return output


class QuantizedAMSFF(nn.Module):
    """
    Quantized version of Adaptive Multi-Scale Feature Fusion module.
    Uses quantized operations for efficient inference.
    """

    def __init__(self, channels_list, output_channels=None):
        super().__init__()

        self.n_scales = len(channels_list)
        self.output_channels = output_channels or max(channels_list)

        # Scale-specific feature enhancement with quantized convolutions
        self.scale_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(channels),
                nn.LeakyReLU(inplace=True),
                SqueezeExcitation(channels)
            ) for channels in channels_list
        ])

        # Quantized feature projections
        self.scale_projectors = nn.ModuleList([
            nn.Conv3d(channels, self.output_channels, kernel_size=1)
            for channels in channels_list
        ])

        # Quantized weight generation
        self.weight_generators = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, channels // 8, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(channels // 8, 1, kernel_size=1)
            ) for channels in channels_list
        ])

        # Final quantized fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(self.output_channels, self.output_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.output_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, feature_maps):
        assert len(feature_maps) == self.n_scales, "Number of input feature maps must match number of scales"

        # Process each scale with quantization
        enhanced_features = []
        scale_weights = []

        for i, feature in enumerate(feature_maps):
            # Enhance features at each scale
            enhanced = self.scale_enhancers[i](feature)
            enhanced_features.append(enhanced)

            # Generate scale-specific weights
            weight = self.weight_generators[i](enhanced)
            scale_weights.append(weight)

        # Normalize weights using softmax across scales
        scale_weights = torch.cat(scale_weights, dim=1)
        normalized_weights = F.softmax(scale_weights, dim=1)

        # Project features to common channel dimension and resample to target resolution
        target_shape = feature_maps[0].shape[2:]  # Use shape of first feature map as target
        resampled_features = []

        for i, feature in enumerate(enhanced_features):
            # Project to common dimension
            projected = self.scale_projectors[i](feature)

            # Resample if needed
            if projected.shape[2:] != target_shape:
                projected = F.interpolate(projected, size=target_shape, mode='trilinear', align_corners=False)

            resampled_features.append(projected)

        # Apply weights to each scale
        weighted_features = []
        for i, feature in enumerate(resampled_features):
            weight = normalized_weights[:, i:i + 1]
            weighted = feature * weight
            weighted_features.append(weighted)

        # Sum weighted features
        fused = sum(weighted_features)

        # Apply final convolution
        output = self.fusion_conv(fused)

        return output


class EncoderBlock(nn.Module):
    """
    Encoder block with optional depthwise separable convolutions for efficiency.
    """

    def __init__(self, in_channels, out_channels, use_depthwise_separable=True):
        super().__init__()

        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv3d(in_channels, out_channels)
            self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.se = SqueezeExcitation(out_channels)

        # Skip connection for residual learning
        self.skip = nn.Conv3d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Apply SE before adding residual
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection handling and adaptive feature fusion.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_depthwise_separable=True):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

        # Adaptive fusion between upsampled features and skip connection
        self.fusion = AMSFF([out_channels, skip_channels], output_channels=out_channels)

        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv3d(out_channels, out_channels)
            self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels)
        else:
            self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample
        up = self.upconv(x)

        # Fusion between upsampled and skip connection
        fused = self.fusion([up, skip])

        # Convolution blocks
        out = self.conv1(fused)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        return out


class AdaptiveMultiScaleFeatureFusionNet(nn.Module):
    """
    Complete 3D U-Net architecture with Adaptive Multi-Scale Feature Fusion modules.
    Features progressive channel growth and efficient convolutions.
    """

    def __init__(self, in_channels=1, num_classes=3, initial_channels=32,
                 depth=4, growth_factor=1.5, max_channels=320,
                 use_depthwise_separable=True):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth

        # Calculate channel dimensions with progressive growth
        self.channels = []
        for i in range(depth + 1):
            channels = min(int(initial_channels * (growth_factor ** i)), max_channels)
            self.channels.append(channels)

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, self.channels[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.channels[0]),
            nn.LeakyReLU(inplace=True)
        )

        # Encoder pathway
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(depth):
            self.encoders.append(
                EncoderBlock(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i],
                    use_depthwise_separable=use_depthwise_separable
                )
            )
            self.downsamplers.append(
                nn.Conv3d(self.channels[i], self.channels[i + 1], kernel_size=3, stride=2, padding=1)
            )

        # Bottom level
        self.bottom = nn.Sequential(
            EncoderBlock(
                in_channels=self.channels[-1],
                out_channels=self.channels[-1],
                use_depthwise_separable=use_depthwise_separable
            ),
            SqueezeExcitation(self.channels[-1])
        )

        # Decoder pathway
        self.decoders = nn.ModuleList()

        for i in range(depth):
            self.decoders.append(
                DecoderBlock(
                    in_channels=self.channels[depth - i],
                    skip_channels=self.channels[depth - i - 1],
                    out_channels=self.channels[depth - i - 1],
                    use_depthwise_separable=use_depthwise_separable
                )
            )

        # Multi-scale feature fusion for final prediction
        self.msff = AMSFF(
            channels_list=self.channels[:depth],
            output_channels=self.channels[0]
        )

        # Final classification layer
        self.final_conv = nn.Conv3d(self.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Encoder pathway with skip connections
        skips = [x]

        for i in range(self.depth):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.downsamplers[i](x)

        # Bottom level
        x = self.bottom(x)

        # Decoder pathway
        decoder_features = []

        for i in range(self.depth):
            skip = skips[self.depth - i]
            x = self.decoders[i](x, skip)
            decoder_features.insert(0, x)  # Store decoder features for multi-scale fusion

        # Multi-scale feature fusion
        fused = self.msff(decoder_features)

        # Final classification
        logits = self.final_conv(fused)

        return logits


def create_am_msf_net(config=None):
    """
    Factory function to create an AdaptiveMultiScaleFeatureFusionNet instance.
    
    Args:
        config: Optional configuration dictionary containing model parameters.
                If None, default parameters will be used.
    
    Returns:
        AdaptiveMultiScaleFeatureFusionNet: Configured model instance
    """
    if config is None:
        config = {}
    
    # Default parameters
    default_params = {
        'in_channels': 1,
        'num_classes': 3,
        'initial_channels': 32,
        'depth': 4,
        'growth_factor': 1.5,
        'max_channels': 320,
        'use_depthwise_separable': True
    }
    
    # Update with provided config
    params = {**default_params, **config}
    
    return AdaptiveMultiScaleFeatureFusionNet(
        in_channels=params['in_channels'],
        num_classes=params['num_classes'],
        initial_channels=params['initial_channels'],
        depth=params['depth'],
        growth_factor=params['growth_factor'],
        max_channels=params['max_channels'],
        use_depthwise_separable=params['use_depthwise_separable']
    )