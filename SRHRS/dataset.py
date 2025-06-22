import os
import csv
import numpy as np  # 添加numpy导入
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class CropPositionDataset(Dataset):
    """
    加载裁剪图像块及其位置和尺度标签

    参数:
    root_dir (str): 图像块存储目录
    csv_file (str): 元数据CSV文件路径
    transform (callable, optional): 图像预处理转换
    """

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.metadata = []

        # 从CSV文件读取元数据
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                # 转换数据类型
                row['scale'] = float(row['scale'])
                row['rel_x'] = float(row['rel_x'])
                row['rel_y'] = float(row['rel_y'])
                self.metadata.append(row)

        print(f"数据集加载完成: 共 {len(self.metadata)} 个图像块")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取元数据
        meta = self.metadata[idx]
        img_name = os.path.join(self.root_dir, meta['filename'])

        try:
            # 加载图像
            image = Image.open(img_name).convert('RGB')
            w, h = image.size

            # 计算分割点
            w_mid = w // 2
            h_mid = h // 2

            # 创建四个子区域
            regions = [
                (0, 0, w_mid, h_mid),  # 左上
                (w_mid, 0, w, h_mid),  # 右上
                (0, h_mid, w_mid, h),  # 左下
                (w_mid, h_mid, w, h)  # 右下
            ]

            # 裁剪并插值
            sub_images = []
            for region in regions:
                # 裁剪子图
                cropped = image.crop(region)
                # 插值回原始大小
                resized = cropped.resize((w, h), Image.BILINEAR)
                sub_images.append(resized)

            # 应用图像转换
            if self.transform:
                image = self.transform(image)
                sub_images = [self.transform(img) for img in sub_images]

            # 将子图像列表转换为张量 [4, C, H, W]
            sub_images_tensor = torch.stack(sub_images)

            # 提取标签: 尺度和位置
            scale = torch.tensor(meta['scale'], dtype=torch.float32)
            position = torch.tensor([meta['rel_x'], meta['rel_y']], dtype=torch.float32)

            return image, sub_images_tensor, {'scale': scale, 'position': position}

        except Exception as e:
            print(f"加载图像 {img_name} 时出错: {str(e)}")
            # 返回空图像和零标签，保持结构一致
            dummy_image = torch.zeros(3, 512, 512)
            dummy_sub_images = torch.zeros(4, 3, 512, 512)  # 四个空子图像
            dummy_labels = {
                'scale': torch.tensor(0.0),
                'position': torch.tensor([0.0, 0.0])
            }
            return dummy_image, dummy_sub_images, dummy_labels


# ==================== 数据转换函数 ====================

def get_default_transform():
    """获取默认的图像转换管道"""
    return transforms.Compose([
        transforms.Resize((512, 512)),  # 确保统一尺寸
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_eval_transform():
    """获取评估/推理用的图像转换管道"""
    return transforms.Compose([
        transforms.Resize((512, 512)),  # 确保统一尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ==================== 数据加载工具函数 ====================

def create_data_loaders(root_dir, csv_file, batch_size=32, val_split=0.2, shuffle=True):
    """
    创建训练和验证数据加载器

    参数:
    root_dir (str): 图像块目录
    csv_file (str): 元数据CSV路径
    batch_size (int): 批量大小
    val_split (float): 验证集比例 (0.0-1.0)
    shuffle (bool): 是否打乱数据

    返回:
    train_loader, val_loader: 训练和验证数据加载器
    """
    # 创建完整数据集
    full_dataset = CropPositionDataset(
        root_dir=root_dir,
        csv_file=csv_file,
        transform=None  # 先不设置转换
    )

    # 分割数据集
    num_total = len(full_dataset)
    indices = list(range(num_total))
    split = int(val_split * num_total)

    if shuffle:
        np.random.shuffle(indices)  # 使用numpy的shuffle

    train_indices = indices[split:]
    val_indices = indices[:split]

    # 创建训练集和验证集（使用不同的转换）
    train_dataset = full_dataset[train_indices]

    val_dataset = full_dataset[val_indices]

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"数据加载器创建完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")

    return train_loader, val_loader
