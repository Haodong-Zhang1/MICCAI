import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json

def calculate_dice(pred, target, smooth=1.0):
    """计算Dice系数"""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    dice = (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    return dice

def calculate_metrics(pred, target):
    """计算各种评估指标"""
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    # Dice系数
    dice = calculate_dice(pred, target)
    
    # 准确率
    accuracy = np.mean(pred == target)
    
    # 精确率
    precision = np.sum(pred * target) / (np.sum(pred) + 1e-6)
    
    # 召回率
    recall = np.sum(pred * target) / (np.sum(target) + 1e-6)
    
    # F1分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_training_history(history, save_dir):
    """绘制训练历史"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Dice系数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='训练Dice')
    plt.plot(history['val_dice'], label='验证Dice')
    plt.title('训练和验证Dice系数')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(pred, target, save_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(target.flatten(), pred.flatten())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def save_metrics(metrics, save_dir):
    """保存评估指标"""
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    # 保存为JSON格式
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def visualize_predictions(model, test_loader, device, save_dir, num_samples=5):
    """可视化预测结果"""
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            targets = batch['label']
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 选择中间切片进行可视化
            slice_idx = preds.shape[2] // 2
            
            plt.figure(figsize=(15, 5))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(images[0, 0, slice_idx].cpu().numpy(), cmap='gray')
            plt.title('原始图像')
            plt.axis('off')
            
            # 真实标签
            plt.subplot(1, 3, 2)
            plt.imshow(targets[0, slice_idx].numpy(), cmap='jet')
            plt.title('真实标签')
            plt.axis('off')
            
            # 预测结果
            plt.subplot(1, 3, 3)
            plt.imshow(preds[0, slice_idx], cmap='jet')
            plt.title('预测结果')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_sample_{i}.png'))
            plt.close()

def train_with_visualization(model, train_loader, val_loader, test_loader, optimizer, num_epochs, device, save_dir):
    """训练模型并可视化结果"""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_metrics': [], 'val_metrics': []
    }
    
    best_val_dice = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # 计算训练指标
        train_metrics = calculate_metrics(np.array(train_preds), np.array(train_targets))
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_dice'].append(train_metrics['dice'])
        history['train_metrics'].append(train_metrics)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['label'].to(device)
                
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # 计算验证指标
        val_metrics = calculate_metrics(np.array(val_preds), np.array(val_targets))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_dice'].append(val_metrics['dice'])
        history['val_metrics'].append(val_metrics)
        
        # 保存最佳模型
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {history["train_loss"][-1]:.4f}, Train Dice: {train_metrics["dice"]:.4f}')
        print(f'Val Loss: {history["val_loss"][-1]:.4f}, Val Dice: {val_metrics["dice"]:.4f}')
        
        # 每个epoch结束后保存可视化结果
        plot_training_history(history, save_dir)
        save_metrics(history['train_metrics'], os.path.join(save_dir, 'train_metrics'))
        save_metrics(history['val_metrics'], os.path.join(save_dir, 'val_metrics'))
    
    # 训练结束后进行最终评估和可视化
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    visualize_predictions(model, test_loader, device, save_dir)
    
    return history

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建保存目录
    save_dir = os.path.join('output', 'visualization', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和数据
    from kits23_segmentation.models.am_msf_net import create_am_msf_net
    from kits23_segmentation.data.data_loader import get_data_loader
    
    model = create_am_msf_net().to(device)
    train_loader = get_data_loader(split='train')
    val_loader = get_data_loader(split='val')
    test_loader = get_data_loader(split='test')
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 开始训练
    history = train_with_visualization(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        num_epochs=300,
        device=device,
        save_dir=save_dir
    ) 