import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import json

class KiTS23Dataset(Dataset):
    """
    KiTS23数据集加载器
    """
    def __init__(self, data_dir, split='train', transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据集根目录（预处理后的数据目录）
            split: 数据集分割（'train', 'val', 'test'）
            transform: 数据转换函数
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 获取所有可用的case
        self.cases = sorted([d.name for d in self.data_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('case')])
        
        # 确保找到了cases
        if not self.cases:
            raise RuntimeError(f"No cases found in {self.data_dir}")
            
        # 按照8:1:1的比例分割数据集
        n_total = len(self.cases)
        if split == 'train':
            self.cases = self.cases[:int(0.8 * n_total)]
        elif split == 'val':
            self.cases = self.cases[int(0.8 * n_total):int(0.9 * n_total)]
        else:  # test
            self.cases = self.cases[int(0.9 * n_total):]
        
        print(f"Found {len(self.cases)} cases for {split} set")
        print(f"First few cases: {', '.join(self.cases[:5])}")
        
        # 验证第一个case的文件结构
        first_case = self.cases[0]
        first_case_dir = self.data_dir / first_case
        print(f"\nChecking file structure in {first_case}:")
        for item in first_case_dir.iterdir():
            print(f"  - {item.name}")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        # 获取案例目录
        case_id = self.cases[idx]
        case_dir = self.data_dir / case_id
        
        # 检查目录是否存在
        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        
        # 加载预处理后的图像和标签
        image_path = case_dir / 'imaging_preprocessed.nii.gz'
        label_path = case_dir / 'segmentation_preprocessed.nii.gz'
        
        if not image_path.exists():
            raise FileNotFoundError(f"Preprocessed image file not found: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Preprocessed label file not found: {label_path}")
            
        # 加载预处理后的数据
        image_nib = nib.load(str(image_path))
        label_nib = nib.load(str(label_path))
        
        image = image_nib.get_fdata()
        label = label_nib.get_fdata()
        
        # 转换为张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 添加通道维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 应用数据转换
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        # 构建返回字典
        sample = {
            'image': image,
            'label': label,
            'case_id': case_id
        }
        
        return sample

class DataTransform:
    """
    数据预处理和增强
    """
    def __init__(self, patch_size=None, augment=False):
        """
        初始化数据转换
        
        Args:
            patch_size: 图像块大小，如果为None则使用完整图像
            augment: 是否进行数据增强
        """
        self.patch_size = patch_size
        self.augment = augment
    
    def __call__(self, image, label=None):
        """
        应用数据转换
        
        Args:
            image: 输入图像
            label: 分割标签（可选）
        
        Returns:
            转换后的图像和标签
        """
        # 标准化
        image = self.normalize(image)
        
        # 提取图像块
        if self.patch_size is not None:
            image, label = self.extract_patch(image, label)
        
        # 数据增强
        if self.augment:
            image, label = self.augment_data(image, label)
        
        return (image, label) if label is not None else image
    
    @staticmethod
    def normalize(image):
        """图像标准化"""
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    
    def extract_patch(self, image, label=None):
        """提取随机图像块"""
        if self.patch_size is None:
            return image, label
        
        # 获取图像尺寸
        _, D, H, W = image.shape
        
        # 确保patch_size不大于图像尺寸
        pd, ph, pw = [min(p, d) for p, d in zip(self.patch_size, (D, H, W))]
        
        # 随机选择起始位置
        d = np.random.randint(0, D - pd + 1)
        h = np.random.randint(0, H - ph + 1)
        w = np.random.randint(0, W - pw + 1)
        
        # 提取图像块
        image = image[:, d:d+pd, h:h+ph, w:w+pw]
        if label is not None:
            label = label[d:d+pd, h:h+ph, w:w+pw]
        
        return image, label
    
    def augment_data(self, image, label=None):
        """数据增强"""
        if not self.augment:
            return image, label
        
        # 随机翻转
        if np.random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
            if label is not None:
                label = torch.flip(label, dims=[-1])
        
        if np.random.random() > 0.5:
            image = torch.flip(image, dims=[-2])
            if label is not None:
                label = torch.flip(label, dims=[-2])
        
        # 随机旋转
        k = np.random.randint(0, 4)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[-2, -1])
            if label is not None:
                label = torch.rot90(label, k=k, dims=[-2, -1])
        
        return image, label

def collate_fn(batch):
    """
    自定义批处理函数，确保所有数据具有相同的大小
    """
    # 获取批次中的最大尺寸
    max_d = max(item['image'].shape[1] for item in batch)
    max_h = max(item['image'].shape[2] for item in batch)
    max_w = max(item['image'].shape[3] for item in batch)
    
    # 调整所有数据到相同大小
    images = []
    labels = []
    case_ids = []
    
    for item in batch:
        # 获取当前数据尺寸
        _, d, h, w = item['image'].shape
        
        # 创建填充后的张量
        padded_image = torch.zeros((1, max_d, max_h, max_w), dtype=item['image'].dtype)
        padded_label = torch.zeros((max_d, max_h, max_w), dtype=item['label'].dtype)
        
        # 复制数据
        padded_image[:, :d, :h, :w] = item['image']
        padded_label[:d, :h, :w] = item['label']
        
        images.append(padded_image)
        labels.append(padded_label)
        case_ids.append(item['case_id'])
    
    # 堆叠数据
    batch_dict = {
        'image': torch.stack(images),
        'label': torch.stack(labels),
        'case_id': case_ids
    }
    
    return batch_dict

def get_data_loader(data_dir, split='train', batch_size=1, num_workers=4, 
                   transform=None, shuffle=True):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据集根目录
        split: 数据集分割（'train', 'val', 'test'）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        transform: 数据转换函数
        shuffle: 是否打乱数据
    
    Returns:
        DataLoader: PyTorch数据加载器
    """
    # 创建数据集实例
    dataset = KiTS23Dataset(
        data_dir=data_dir,
        split=split,
        transform=transform
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == 'train',  # 只在训练集打乱数据
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 如果使用GPU，启用pin_memory
        collate_fn=collate_fn  # 使用自定义的批处理函数
    )
    
    return loader 