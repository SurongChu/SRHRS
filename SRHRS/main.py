import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import CropPositionDataset, create_data_loaders, get_default_transform
from Model import SRHRS


# 训练配置
class TrainerConfig:
    def __init__(self):
        self.root_dir = 'path_to_your_image_directory'  # 替换为实际路径
        self.csv_file = 'path_to_your_metadata.csv'  # 替换为实际路径
        self.batch_size = 32
        self.val_split = 0.2
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.momentum = 0.996
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dir = 'runs/experiment_' + time.strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_interval = 5


# 训练器类
class Trainer:
    def __init__(self, config):
        self.config = config
        self.writer = SummaryWriter(config.log_dir)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # 初始化模型
        self.model = SRHRS().to(config.device)

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # 初始化数据加载器
        self.train_loader, self.val_loader = create_data_loaders(
            root_dir=config.root_dir,
            csv_file=config.csv_file,
            batch_size=config.batch_size,
            val_split=config.val_split,
            shuffle=True
        )

        # 损失权重
        self.loss_weights = {
            'consistency_loss': 1.0,
            'prediction_loss': 2.0,
            'decomposition_loss': 1.0
        }

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (images, sub_images, labels) in enumerate(self.train_loader):
            # 准备数据
            images = images.to(self.config.device)
            sub_images = sub_images.to(self.config.device)

            # 子图像拆分
            x31, x32, x33, x34 = sub_images[:, 0], sub_images[:, 1], sub_images[:, 2], sub_images[:, 3]

            # 准备标签
            scale_labels = labels['scale'].to(self.config.device)
            pos_labels = labels['position'].to(self.config.device)
            # 合并标签为 [batch_size, 3] 张量: [scale, rel_x, rel_y]
            combined_labels = torch.stack([scale_labels, pos_labels[:, 0], pos_labels[:, 1]], dim=1)

            # 前向传播
            online_proj, target_proj, predictions, q, k = self.model(
                images, images, x31, x32, x33, x34
            )

            # 计算损失
            loss_dict, loss = self.model.compute_losses(
                online_proj, target_proj, predictions, q, k, combined_labels
            )

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 更新目标网络
            self.model._update_target_network(self.config.momentum)

            # 记录统计信息
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 打印训练进度
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

                # 记录到TensorBoard
                for loss_name, loss_value in loss_dict.items():
                    self.writer.add_scalar(f'train/{loss_name}', loss_value, epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / total_samples
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, sub_images, labels) in enumerate(self.val_loader):
                # 准备数据
                images = images.to(self.config.device)
                sub_images = sub_images.to(self.config.device)

                # 子图像拆分
                x31, x32, x33, x34 = sub_images[:, 0], sub_images[:, 1], sub_images[:, 2], sub_images[:, 3]

                # 准备标签
                scale_labels = labels['scale'].to(self.config.device)
                pos_labels = labels['position'].to(self.config.device)
                combined_labels = torch.stack([scale_labels, pos_labels[:, 0], pos_labels[:, 1]], dim=1)

                # 前向传播
                online_proj, target_proj, predictions, q, k = self.model(
                    images, images, x31, x32, x33, x34
                )

                # 计算损失
                loss_dict, loss = self.model.compute_losses(
                    online_proj, target_proj, predictions, q, k, combined_labels
                )

                # 记录统计信息
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # 记录验证损失到TensorBoard
                if batch_idx == 0:
                    for loss_name, loss_value in loss_dict.items():
                        self.writer.add_scalar(f'val/{loss_name}', loss_value, epoch)

        avg_loss = total_loss / total_samples
        return avg_loss

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()

            # 训练一个epoch
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate(epoch)

            # 打印epoch统计信息
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # 记录到TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)

            # 保存checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best')

        self.writer.close()


# 主函数
def main():
    config = TrainerConfig()
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()