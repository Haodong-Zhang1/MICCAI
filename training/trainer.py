# kits23_segmentation/training/trainer.py
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
import json
from tqdm import tqdm
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import torch.nn as nn

from kits23_segmentation.utils.metrics import calculate_metrics


class AMSFFTrainer:
    """
    AMSFF模型训练器
    """
    def __init__(self, model, train_loader, val_loader, config, device, scaler=None, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.scaler = scaler
        self.logger = logger or logging.getLogger(__name__)
        
        # 设置损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        
        # 设置混合精度训练
        self.mixed_precision = scaler is not None
        
        # 设置训练参数
        self.max_epochs = config.get('max_epochs', 300)
        self.eval_interval = config.get('eval_interval', 5)
        self.save_interval = config.get('save_interval', 10)
        self.patience = config.get('patience', 30)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化训练状态
        self.best_val_loss = float('inf')
        self.best_model_path = self.output_dir / 'best_model.pth'
        self.patience_counter = 0
        self.current_epoch = 0

        # Default configuration
        default_config = {
            'output_dir': 'kits23_segmentation/output',
            'learning_rate': 1e-4,
            'max_epochs': 300,
            'weight_decay': 1e-5,
            'lr_scheduler': 'cosine',
            'patience': 30,
            'mixed_precision': True,
            'gradient_clipping': 1.0,
            'experiment_name': f'amsff_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'save_interval': 10
        }

        # Update with provided config
        self.config = default_config.copy()
        if config is not None:
            self.config.update(config)

        # Setup directories
        self.output_dir = os.path.join(self.config['output_dir'], 'models', self.config['experiment_name'])
        self.log_dir = os.path.join(self.config['output_dir'], 'logs', self.config['experiment_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Setup learning rate scheduler
        if self.config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['max_epochs']
            )
        elif self.config['lr_scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        else:
            self.scheduler = None

        # Setup mixed precision training
        self.mixed_precision = self.config['mixed_precision']
        if self.mixed_precision:
            self.scaler = GradScaler()

        # Setup logging
        self.logger = self._setup_logger()

        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.no_improvement_epochs = 0
        self.history = defaultdict(list)

        # Move model to device
        self.model.to(self.device)

        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Model initialized with {total_params:,} parameters "
                         f"({trainable_params:,} trainable)")
        self.logger.info(f"Training configuration: {self.config}")

    def _setup_logger(self):
        """Setup and configure logger."""
        logger = logging.getLogger(f"AMSFFTrainer_{self.config['experiment_name']}")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        if logger.handlers:
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, 'training.log'))
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_format)
        logger.addHandler(console_handler)

        return logger

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        start_time = time.time()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['max_epochs']}")

        for batch_idx, batch in enumerate(pbar):
            # Get data
            images = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss, loss_components = self.loss_fn(logits, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                logits = self.model(images)
                loss, loss_components = self.loss_fn(logits, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )

                self.optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)

            # Calculate batch metrics
            with torch.no_grad():
                batch_metrics = calculate_metrics(
                    logits.detach(),
                    targets.detach(),
                    include_background=False
                )

                # Update epoch metrics
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'dice_tumor': f"{batch_metrics['dice_tumor']:.4f}"
                })

        # Calculate average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)

        # Log epoch results
        elapsed_time = time.time() - start_time
        self.logger.info(f"Epoch {self.current_epoch + 1} training completed in {elapsed_time:.2f}s")
        self.logger.info(f"Train Loss: {avg_loss:.4f}, "
                         f"Dice Kidney: {epoch_metrics['dice_kidney']:.4f}, "
                         f"Dice Tumor: {epoch_metrics['dice_tumor']:.4f}")

        # Save to history
        self.history['train_loss'].append(avg_loss)
        for k, v in epoch_metrics.items():
            self.history[f'train_{k}'].append(v)

        return avg_loss, epoch_metrics

    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        val_metrics = defaultdict(float)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # Get data
                images = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)

                # Forward pass
                logits = self.model(images)
                loss, _ = self.loss_fn(logits, targets)

                # Update loss
                val_loss += loss.item()

                # Calculate metrics
                batch_metrics = calculate_metrics(
                    logits,
                    targets,
                    include_background=False
                )

                # Update epoch metrics
                for k, v in batch_metrics.items():
                    val_metrics[k] += v
                    val_loss /= len(self.val_loader)
                    for k in val_metrics:
                        val_metrics[k] /= len(self.val_loader)

                    # Log validation results
                    self.logger.info(f"Validation Loss: {val_loss:.4f}, "
                                     f"Dice Kidney: {val_metrics['dice_kidney']:.4f}, "
                                     f"Dice Tumor: {val_metrics['dice_tumor']:.4f}")

                    # Save to history
                    self.history['val_loss'].append(val_loss)
                    for k, v in val_metrics.items():
                        self.history[f'val_{k}'].append(v)

                    # Return average dice score (kidney + tumor) as primary validation metric
                    primary_metric = (val_metrics['dice_kidney'] + val_metrics['dice_tumor']) / 2

                    return val_loss, val_metrics, primary_metric

    def train(self):
        """Execute full training process."""
        self.logger.info(f"Starting training for {self.config['max_epochs']} epochs")

        for epoch in range(self.config['max_epochs']):
            self.current_epoch = epoch

            # Training epoch
            train_loss, train_metrics = self.train_epoch()

            # Validation
            val_loss, val_metrics, primary_metric = self.validate()

            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint if improved
            if primary_metric > self.best_val_metric:
                self.best_val_metric = primary_metric
                self.no_improvement_epochs = 0

                # Save best model
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"New best model saved with validation metric: {primary_metric:.4f}")
            else:
                self.no_improvement_epochs += 1
                self.logger.info(f"No improvement for {self.no_improvement_epochs} epochs")

            # Save regular checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # Save training history
            self.save_history()

            # Early stopping
            if self.no_improvement_epochs >= self.config['patience']:
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Current learning rate: {current_lr:.2e}")

        # Load best model after training
        self.load_checkpoint('best_model.pth')
        self.logger.info("Training completed, best model loaded")

        return self.history

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, filename)

        # Create checkpoint
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'history': dict(self.history)
        }

        # Add scheduler state if exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Add scaler state if using mixed precision
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, filename)

        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if exists
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state if using mixed precision
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']

        # Load history if available
        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True

    def save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.output_dir, 'training_history.json')

        # Convert defaultdict to regular dict for JSON serialization
        history_dict = dict(self.history)

        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)

    def test(self, test_loader, save_predictions=True):
        """
        Test the model on a test dataset.

        Args:
            test_loader: Test data loader
            save_predictions: Whether to save predictions

        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        test_metrics = defaultdict(float)
        predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                # Get data
                images = batch['image'].to(self.device)
                targets = batch['label'].to(self.device)
                case_id = batch['case_id'][0]  # Assuming batch size 1 for testing

                # Forward pass
                logits = self.model(images)

                # Calculate metrics
                batch_metrics = calculate_metrics(
                    logits,
                    targets,
                    include_background=False
                )

                # Update metrics
                for k, v in batch_metrics.items():
                    test_metrics[k] += v

                # Save predictions if requested
                if save_predictions:
                    # Convert logits to segmentation mask
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1).cpu().numpy()

                    # Save info for later writing
                    predictions.append({
                        'case_id': case_id,
                        'predictions': preds,
                        'metrics': {k: float(v) for k, v in batch_metrics.items()}
                    })

        # Calculate average metrics
        for k in test_metrics:
            test_metrics[k] /= len(test_loader)

        # Log test results
        self.logger.info(f"Test Results:")
        self.logger.info(f"Dice Kidney: {test_metrics['dice_kidney']:.4f}")
        self.logger.info(f"Dice Tumor: {test_metrics['dice_tumor']:.4f}")
        self.logger.info(f"Surface Dice Kidney: {test_metrics.get('surface_dice_kidney', 0.0):.4f}")
        self.logger.info(f"Surface Dice Tumor: {test_metrics.get('surface_dice_tumor', 0.0):.4f}")

        # Save metrics and predictions
        if save_predictions:
            # Create results directory
            results_dir = os.path.join(self.config['output_dir'], 'results', self.config['experiment_name'])
            os.makedirs(results_dir, exist_ok=True)

            # Save overall metrics
            metrics_path = os.path.join(results_dir, 'test_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(dict(test_metrics), f, indent=2)

            # Save predictions for each case
            for pred_info in predictions:
                case_id = pred_info['case_id']
                pred_mask = pred_info['predictions']
                case_metrics = pred_info['metrics']

                # Create case directory
                case_dir = os.path.join(results_dir, case_id)
                os.makedirs(case_dir, exist_ok=True)

                # Save prediction as NumPy array
                pred_path = os.path.join(case_dir, 'prediction.npy')
                np.save(pred_path, pred_mask)

                # Save case metrics
                case_metrics_path = os.path.join(case_dir, 'metrics.json')
                with open(case_metrics_path, 'w') as f:
                    json.dump(case_metrics, f, indent=2)

        return dict(test_metrics)

    def train_with_efficiency_monitoring(self):
        """
        执行完整的训练过程，同时监控训练效率指标
        """
        self.logger.info(f"开始训练，共{self.config['max_epochs']}轮")
        
        # 初始化效率监控指标
        efficiency_metrics = {
            'epoch_times': [],
            'batch_times': [],
            'memory_usage': [],
            'gpu_usage': [],
            'inference_times': [],
            'training_speed': []  # samples per second
        }
        
        for epoch in range(self.config['max_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_loss, train_metrics = self.train_epoch_with_monitoring(efficiency_metrics)
            
            # 验证
            val_loss, val_metrics, primary_metric = self.validate()
            
            # 记录epoch时间
            epoch_time = time.time() - epoch_start_time
            efficiency_metrics['epoch_times'].append(epoch_time)
            
            # 记录GPU内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
                efficiency_metrics['memory_usage'].append(memory_used)
                torch.cuda.reset_peak_memory_stats()
            
            # 学习率调度器步进
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 保存检查点（如果有改进）
            if primary_metric > self.best_val_metric:
                self.best_val_metric = primary_metric
                self.no_improvement_epochs = 0
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"保存新的最佳模型，验证指标: {primary_metric:.4f}")
            else:
                self.no_improvement_epochs += 1
                self.logger.info(f"已经{self.no_improvement_epochs}轮没有改进")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # 保存训练历史
            self.save_history()
            
            # 记录效率指标
            self.logger.info(f"Epoch {epoch + 1} 效率指标:")
            self.logger.info(f"- 训练时间: {epoch_time:.2f}秒")
            self.logger.info(f"- 平均批处理时间: {np.mean(efficiency_metrics['batch_times']):.4f}秒")
            if torch.cuda.is_available():
                self.logger.info(f"- GPU内存使用: {memory_used:.2f}GB")
            self.logger.info(f"- 训练速度: {np.mean(efficiency_metrics['training_speed']):.2f}样本/秒")
            
            # 提前停止
            if self.no_improvement_epochs >= self.config['patience']:
                self.logger.info(f"触发提前停止，在第{epoch + 1}轮")
                break
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"当前学习率: {current_lr:.2e}")
        
        # 加载最佳模型
        self.load_checkpoint('best_model.pth')
        self.logger.info("训练完成，已加载最佳模型")
        
        # 保存效率指标
        self.save_efficiency_metrics(efficiency_metrics)
        
        return self.history, efficiency_metrics

    def train_epoch_with_monitoring(self, efficiency_metrics):
        """带效率监控的训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        batch_start_time = time.time()
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config['max_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # 记录批处理开始时间
            batch_start_time = time.time()
            
            # 获取数据
            images = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            if self.mixed_precision:
                with autocast():
                    logits = self.model(images)
                    loss = self.model.loss_fn(logits, targets)
                
                # 反向传播（使用梯度缩放）
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config['gradient_clipping'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准前向传播
                logits = self.model(images)
                loss = self.model.loss_fn(logits, targets)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clipping']
                    )
                
                self.optimizer.step()
            
            # 更新进度条
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (batch_idx + 1)
            
            # 计算批处理指标
            with torch.no_grad():
                batch_metrics = calculate_metrics(
                    logits.detach(),
                    targets.detach(),
                    include_background=False
                )
                
                # 更新epoch指标
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'dice_tumor': f"{batch_metrics['dice_tumor']:.4f}"
                })
            
            # 记录效率指标
            batch_time = time.time() - batch_start_time
            efficiency_metrics['batch_times'].append(batch_time)
            efficiency_metrics['training_speed'].append(len(images) / batch_time)
            
            if torch.cuda.is_available():
                efficiency_metrics['gpu_usage'].append(
                    torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                )
        
        # 计算平均指标
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)
        
        return avg_loss, epoch_metrics

    def save_efficiency_metrics(self, metrics):
        """保存效率指标到JSON文件"""
        metrics_path = os.path.join(self.output_dir, 'efficiency_metrics.json')
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (list, np.ndarray)):
                serializable_metrics[k] = [float(x) for x in v]
            else:
                serializable_metrics[k] = v
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

def train_amsff_model(model, train_loader, val_loader, test_loader=None, config=None):
    """
    Factory function to create and train an AM-MSF-Net model.
    
    Args:
        model: The AM-MSF-Net model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader
        config: Optional configuration dictionary
    
    Returns:
        AMSFFTrainer: Configured trainer instance
    """
    # Default configuration
    default_config = {
        'output_dir': 'kits23_segmentation/output',
        'learning_rate': 1e-4,
        'max_epochs': 300,
        'weight_decay': 1e-5,
        'lr_scheduler': 'cosine',
        'patience': 30,
        'mixed_precision': True,
        'gradient_clipping': 1.0,
        'experiment_name': f'amsff_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'save_interval': 10
    }
    
    # Update with provided config
    if config is not None:
        default_config.update(config)
    
    # Create trainer instance
    trainer = AMSFFTrainer(
        model=model,
        loss_fn=model.loss_fn,  # Assuming model has loss_fn attribute
        train_loader=train_loader,
        val_loader=val_loader,
        config=default_config
    )
    
    # Train the model
    trainer.train()
    
    # Test if test_loader is provided
    if test_loader is not None:
        trainer.test(test_loader)
    
    return trainer