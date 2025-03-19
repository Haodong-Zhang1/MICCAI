import os
import json
from typing import Dict, Any, Optional

class ConfigManager:
    """配置管理器，用于处理训练和模型配置"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._get_default_config()
        
        if config_path is not None and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据集配置
            'dataset': {
                'data_dir': 'dataset',
                'preprocessed': True,
                'patch_size': [64, 64, 32],
                'batch_size': 1,
                'num_workers': 2
            },
            
            # 模型配置
            'model': {
                'in_channels': 1,
                'initial_channels': 16,
                'num_classes': 3,
                'depth': 3,
                'growth_factor': 1.5,
                'max_channels': 320,
                'use_depthwise_separable': True
            },
            
            # 训练配置
            'training': {
                'output_dir': 'kits23_segmentation/output',
                'learning_rate': 1e-4,
                'max_epochs': 10,
                'weight_decay': 1e-5,
                'lr_scheduler': 'cosine',
                'patience': 30,
                'mixed_precision': True,
                'gradient_clipping': 1.0,
                'save_interval': 10
            },
            
            # 性能监控配置
            'monitoring': {
                'enable_tensorboard': True,
                'log_interval': 10,
                'validation_interval': 1
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
            self.update_config(loaded_config)
    
    def save_config(self, config_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件保存路径
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        def update_recursive(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    update_recursive(d1[k], v)
                else:
                    d1[k] = v
        
        update_recursive(self.config, new_config)
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config['model']
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config['training']
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self.config['dataset']
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取性能监控配置"""
        return self.config['monitoring'] 