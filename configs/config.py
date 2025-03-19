# kits23_segmentation/configs/config.py
import os
import json
import argparse
import psutil
import torch
import GPUtil
from collections import defaultdict
import torch.nn as nn
import torch.quantization
import onnx
import onnxruntime
import matplotlib.pyplot as plt
import time
import numpy as np
from kits23_segmentation.utils.config import ConfigManager
from kits23_segmentation.utils.visualization import visualize_performance_metrics, create_performance_report


class ConfigManager:
    """
    Configuration manager for kidney tumor segmentation project.
    Handles loading, saving, and merging configurations from different sources.
    """

    def __init__(self, config_path=None):
        self.default_config = {
            'model': {
                'in_channels': 1,
                'num_classes': 3,
                'initial_channels': 32,
                'depth': 4,
                'growth_factor': 1.5,
                'max_channels': 320,
                'use_depthwise_separable': True
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'max_epochs': 300,
                'weight_decay': 1e-5,
                'lr_scheduler': 'cosine',
                'patience': 30,
                'mixed_precision': True,
                'gradient_clipping': 1.0
            },
            'efficiency': {
                'use_quantization': False,
                'quantization_scheme': 'dynamic',
                'target_device': 'cpu',
                'batch_size': 1,
                'num_workers': 4,
                'pin_memory': True
            },
            'deployment': {
                'export_format': None,
                'optimize_for_inference': True,
                'use_tensorrt': False,
                'tensorrt_precision': 'FP32',
                'onnx_optimization_level': 3
            },
            'monitoring': {
                'log_interval': 100,
                'save_interval': 10,
                'performance_monitoring': True,
                'resource_tracking': True
            }
        }
        
        # 加载用户配置
        if config_path:
            self.load_config(config_path)

    def _load_json(self, file_path):
        """Load JSON configuration file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration from {file_path}: {e}")
            return {}

    def _merge_configs(self, user_config, target=None, path=None):
        """
        Recursively merge user configuration into target configuration.

        Args:
            user_config: User configuration dictionary
            target: Target configuration dictionary (default: self.config)
            path: Current path in the configuration (for logging)
        """
        if target is None:
            target = self.config

        if path is None:
            path = []

        for key, value in user_config.items():
            current_path = path + [key]

            if key not in target:
                # Add new key
                target[key] = value
            elif isinstance(value, dict) and isinstance(target[key], dict):
                # Recursively merge dictionaries
                self._merge_configs(value, target[key], current_path)
            else:
                # Override existing value
                target[key] = value

    def get(self, key, default=None):
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        target = self.config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def update_from_args(self, args):
        """
        Update configuration from command line arguments.

        Args:
            args: Parsed command line arguments
        """
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Update configuration with non-None values
        for key, value in args.items():
            if value is not None:
                self.set(key, value)

    def save(self, file_path=None):
        """
        Save configuration to file.

        Args:
            file_path: Output file path (default: self.config_path)
        """
        file_path = file_path or self.config_path

        if file_path is None:
            print("No configuration file path specified.")
            return False

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving configuration to {file_path}: {e}")
            return False

    def to_dict(self):
        """Get the entire configuration dictionary."""
        return self.config.copy()

    def reset(self):
        """Reset configuration to default values."""
        self.config = self.default_config.copy()

    def get_efficiency_config(self):
        """获取效率相关配置"""
        return self.config.get('efficiency', {})
    
    def get_deployment_config(self):
        """获取部署相关配置"""
        return self.config.get('deployment', {})
    
    def get_monitoring_config(self):
        """获取监控相关配置"""
        return self.config.get('monitoring', {})

class QuantizedAMSFF(nn.Module):
    """量化版本的AMSFF模型"""
    
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """融合模型层以优化量化性能"""
        torch.quantization.fuse_modules(self.model, [['conv', 'bn', 'relu']], inplace=True)

class AMSFF(nn.Module):
    """改进的AMSFF模块"""
    
    def __init__(self, channels_list, output_channels=None):
        super().__init__()
        # 现有初始化代码...
        
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=output_channels,
            num_heads=8,
            dropout=0.1
        )
        
        # 添加残差连接
        self.residual = nn.Conv3d(
            channels_list[0],
            output_channels,
            kernel_size=1
        )
    
    def forward(self, feature_maps):
        # 现有特征融合代码...
        
        # 添加注意力机制
        fused = feature_maps[0].permute(2, 0, 1, 3, 4)  # 调整维度顺序
        fused, _ = self.attention(fused, fused, fused)
        fused = fused.permute(1, 2, 0, 3, 4)  # 恢复维度顺序
        
        # 添加残差连接
        residual = self.residual(feature_maps[0])
        fused = fused + residual
        
        return fused

def export_model(model, format='onnx', output_dir=None):
    """导出模型为不同格式"""
    if output_dir is None:
        output_dir = 'exports'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 128, 128, 64)
    
    if format == 'onnx':
        # 导出ONNX模型
        onnx_path = os.path.join(output_dir, 'model.onnx')
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
    elif format == 'torchscript':
        # 导出TorchScript模型
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, os.path.join(output_dir, 'model.pt'))

# 添加新的评估指标
class EfficiencyMetrics:
    """计算模型效率指标"""
    
    @staticmethod
    def calculate_inference_time(model, test_loader, device):
        """计算推理时间"""
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                start_time = time.time()
                _ = model(images)
                inference_times.append(time.time() - start_time)
        
        return {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times)
        }
    
    @staticmethod
    def calculate_memory_usage(model, test_loader, device):
        """计算内存使用"""
        if torch.cuda.is_available():
            memory_usage = []
            model.eval()
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    _ = model(images)
                    memory_usage.append(torch.cuda.memory_allocated() / (1024**3))
            
            return {
                'mean_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage),
                'min_memory_usage': np.min(memory_usage)
            }
        return None
    
    @staticmethod
    def calculate_model_complexity(model):
        """计算模型复杂度"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024**2)  # 假设每个参数是4字节
        }

def calculate_comprehensive_metrics(model, test_loader, device):
    """计算综合评估指标"""
    # 计算准确性指标
    accuracy_metrics = calculate_metrics(model, test_loader, device)
    
    # 计算效率指标
    efficiency_metrics = EfficiencyMetrics()
    inference_metrics = efficiency_metrics.calculate_inference_time(model, test_loader, device)
    memory_metrics = efficiency_metrics.calculate_memory_usage(model, test_loader, device)
    complexity_metrics = efficiency_metrics.calculate_model_complexity(model)
    
    # 合并所有指标
    comprehensive_metrics = {
        **accuracy_metrics,
        **inference_metrics,
        **memory_metrics,
        **complexity_metrics
    }
    
    return comprehensive_metrics