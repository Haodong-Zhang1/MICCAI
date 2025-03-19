# kits23_segmentation/main_train.py
import os
import argparse
import logging
import json
import sys
import torch
import numpy as np
import random
import psutil
import GPUtil
import torch.quantization
from datetime import datetime
from pathlib import Path
import torch.nn as nn

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from kits23_segmentation.data.data_loader import get_data_loader, DataTransform
from kits23_segmentation.data.augmentation import get_augmentation_pipeline
from kits23_segmentation.models.am_msf_net import create_am_msf_net, QuantizedAMSFF
from kits23_segmentation.training.trainer import train_amsff_model, AMSFFTrainer
from kits23_segmentation.utils.config import ConfigManager
from kits23_segmentation.utils.visualization import visualize_performance_metrics

def export_model(model, format='torchscript', output_dir=None):
    """导出模型为TorchScript格式"""
    if output_dir is None:
        output_dir = 'exports'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建示例输入
    dummy_input = torch.randn(1, 1, 64, 64, 32)
    
    if format == 'torchscript':
        # 导出TorchScript模型
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, os.path.join(output_dir, 'model.pt'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AM-MSF-Net for KiTS23 Challenge")

    # Dataset and output directories
    parser.add_argument("--dataset_dir", type=str, default="dataset",
                        help="Path to KiTS23 dataset")
    parser.add_argument("--output_dir", type=str, default="kits23_segmentation/output",
                        help="Output directory")
    parser.add_argument("--preprocessed", action="store_true",
                        help="Use preprocessed data for training")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=30,
                        help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 64],
                        help="Training patch size (D, H, W)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Cross-validation fold to use")

    # Model parameters
    parser.add_argument("--initial_channels", type=int, default=32,
                        help="Initial number of channels in the network")
    parser.add_argument("--depth", type=int, default=4,
                        help="Network depth (number of downsampling stages)")
    parser.add_argument("--growth_factor", type=float, default=1.5,
                        help="Channel growth factor between stages")
    parser.add_argument("--max_channels", type=int, default=320,
                        help="Maximum number of channels")
    parser.add_argument("--efficient", action="store_true",
                        help="Use depthwise separable convolutions for efficiency")

    # Loss function parameters
    parser.add_argument("--dice_weight", type=float, default=1.0,
                        help="Weight for Dice loss component")
    parser.add_argument("--focal_weight", type=float, default=0.5,
                        help="Weight for Focal loss component")
    parser.add_argument("--boundary_weight", type=float, default=0.2,
                        help="Weight for boundary loss component")

    # Experiment tracking
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name for logging")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Mixed precision and optimization
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["cosine", "plateau", "none"],
                        help="Learning rate scheduler")

    # 训练过程控制
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="验证集评估间隔（epochs）")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="模型保存间隔（epochs）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="预热训练轮数")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定checkpoint恢复训练")

    # 添加新的参数
    parser.add_argument("--amsff_config", type=str, default=None,
                        help="AMSFF模型配置JSON文件路径")
    parser.add_argument("--use_cloud", action="store_true",
                        help="是否使用云GPU训练")
    parser.add_argument("--quantization", action="store_true",
                        help="是否使用模型量化")
    parser.add_argument("--export_format", type=str, default=None,
                        choices=['onnx', 'torchscript'],
                        help="模型导出格式")
    parser.add_argument("--performance_monitoring", action="store_true",
                        help="是否启用性能监控")

    return parser.parse_args()


def setup_logging(output_dir, experiment_name):
    """Setup logging configuration."""
    log_dir = os.path.join(output_dir, 'logs', experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("KiTS23-Training")


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_environment():
    """检查运行环境和资源"""
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        logging.info(f"使用GPU: {device_name} (设备 {current_device}/{device_count})")
    else:
        logging.warning("未检测到CUDA，将使用CPU训练")

    # 检查系统内存
    memory = psutil.virtual_memory()
    logging.info(f"系统内存: {memory.total / (1024 ** 3):.2f}GB, 可用: {memory.available / (1024 ** 3):.2f}GB")

    # 检查GPU内存
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                logging.info(f"GPU {gpu.id} 内存: {gpu.memoryTotal}MB, 可用: {gpu.memoryFree}MB")
        except:
            logging.warning("无法获取GPU内存信息")


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']


def setup_environment(args):
    """设置训练环境"""
    # 检查CUDA可用性
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.warning("未检测到GPU，使用CPU训练")
    
    # 设置混合精度训练
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    return device, scaler


def monitor_resources():
    """监控系统资源使用情况"""
    metrics = {}
    
    # CPU使用率
    metrics['cpu_percent'] = psutil.cpu_percent()
    
    # 内存使用
    memory = psutil.virtual_memory()
    metrics['memory_percent'] = memory.percent
    metrics['memory_used'] = memory.used / (1024**3)  # GB
    
    # GPU使用情况
    if torch.cuda.is_available():
        metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
        metrics['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
        metrics['gpu_utilization'] = GPUtil.getGPUs()[0].load * 100
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # 设置实验名称
    if args.experiment_name is None:
        args.experiment_name = f"amsff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.experiment_name)
    logger.info("开始训练...")
    
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"设置随机种子: {args.seed}")
    
    # 检查环境
    check_environment()
    device, scaler = setup_environment(args)
    logger.info(f"使用设备: {device}")
    
    # 加载AMSFF配置
    if args.amsff_config:
        config_manager = ConfigManager(args.amsff_config)
    else:
        config_manager = ConfigManager()
    logger.info("加载AMSFF配置完成")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    preprocessed_data_dir = Path("kits23_segmentation/output/preprocessed_data")
    if not preprocessed_data_dir.exists():
        raise FileNotFoundError(f"预处理数据目录不存在: {preprocessed_data_dir}")
        
    transform = DataTransform(
        patch_size=args.patch_size,
        augment=True
    )
    
    # 创建训练、验证和测试数据加载器
    train_loader = get_data_loader(
        data_dir=preprocessed_data_dir,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform
    )
    
    val_loader = get_data_loader(
        data_dir=preprocessed_data_dir,
        split='val',
        batch_size=1,  # 验证时使用批次大小1
        num_workers=args.num_workers,
        transform=DataTransform()  # 验证时不需要数据增强
    )
    
    test_loader = get_data_loader(
        data_dir=preprocessed_data_dir,
        split='test',
        batch_size=1,  # 测试时使用批次大小1
        num_workers=args.num_workers,
        transform=DataTransform()  # 测试时不需要数据增强
    )
    
    # 创建模型
    logger.info("创建AMSFF模型...")
    model_config = config_manager.get_model_config()
    model_config.update({
        'initial_channels': args.initial_channels,
        'depth': args.depth,
        'use_depthwise_separable': args.efficient
    })
    model = create_am_msf_net(model_config)
    model = model.to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 如果启用量化
    if args.quantization:
        logger.info("启用模型量化...")
        model = QuantizedAMSFF(model)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
    
    # 创建优化器
    logger.info("创建优化器...")
    training_config = config_manager.get_training_config()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay if args.weight_decay is not None else training_config['weight_decay']
    )
    
    # 创建学习率调度器
    logger.info("创建学习率调度器...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = AMSFFTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_manager.get_training_config(),
        device=device,
        scaler=scaler,
        logger=logger
    )
    
    # 使用效率监控进行训练
    history, efficiency_metrics = trainer.train_with_efficiency_monitoring()
    
    # 保存最终模型
    logger.info("保存最终模型...")
    final_model_path = os.path.join(args.output_dir, 'models', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # 导出模型
    if args.export_format:
        logger.info(f"导出模型为{args.export_format}格式...")
        export_model(model, args.export_format, args.output_dir)
    
    logger.info("训练完成！")
    return model, history


if __name__ == "__main__":
    main()