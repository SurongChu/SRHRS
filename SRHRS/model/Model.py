import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ProjectionHead(nn.Module):
    """投影网络，输出指定维度的向量"""

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)
        )

    def forward(self, x):
        return self.mlp(x)


class LSPredictionHead(nn.Module):
    """统一预测头，输出三个预测值"""

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=3):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.shared_mlp(x)


class QuadPartitionUpsample(nn.Module):
    """将特征图四等分并上采样回原始尺寸"""

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        batch, channels, h, w = x.size()
        h_mid = h // 2
        w_mid = w // 2

        top_left = x[:, :, :h_mid, :w_mid]
        top_right = x[:, :, :h_mid, w_mid:]
        bottom_left = x[:, :, h_mid:, :w_mid]
        bottom_right = x[:, :, h_mid:, w_mid:]

        top_left_up = F.interpolate(top_left, size=(h, w), mode=self.mode, align_corners=False)
        top_right_up = F.interpolate(top_right, size=(h, w), mode=self.mode, align_corners=False)
        bottom_left_up = F.interpolate(bottom_left, size=(h, w), mode=self.mode, align_corners=False)
        bottom_right_up = F.interpolate(bottom_right, size=(h, w), mode=self.mode, align_corners=False)

        return top_left_up, top_right_up, bottom_left_up, bottom_right_up


class SRHRS(nn.Module):
    """双分支网络架构"""

    def __init__(self, base_encoder=resnet50, feature_dim=1024, prediction_dim=3):
        super().__init__()
        self.online_encoder = base_encoder(pretrained=False)
        self.online_encoder.fc = nn.Identity()
        self.online_projector = ProjectionHead(input_dim=2048, output_dim=feature_dim)
        self.predictor = LSPredictionHead(input_dim=feature_dim, output_dim=prediction_dim)

        self.target_encoder = base_encoder(pretrained=False)
        self.target_encoder.fc = nn.Identity()
        self.target_projector = ProjectionHead(input_dim=2048, output_dim=feature_dim)

        # 添加分区上采样模块
        self.quad_partition = QuadPartitionUpsample(mode='bilinear')

        self._init_target_network()
        self._freeze_target_network()

    def _init_target_network(self):
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())

    def _freeze_target_network(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _update_target_network(self, momentum=0.996):
        for param_o, param_t in zip(self.online_encoder.parameters(),
                                    self.target_encoder.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1. - momentum)
        for param_o, param_t in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            param_t.data = param_t.data * momentum + param_o.data * (1. - momentum)

    def forward(self, x1, x2, x31, x32, x33, x34):
        # 动态分支
        online_feat = self.online_encoder(x1)
        online_proj = self.online_projector(online_feat)
        online_proj = F.normalize(online_proj, dim=1)
        predictions = self.predictor(online_proj)

        # 计算四个查询向量
        q31 = self.online_projector(self.online_encoder(x31))
        q32 = self.online_projector(self.online_encoder(x32))
        q33 = self.online_projector(self.online_encoder(x33))
        q34 = self.online_projector(self.online_encoder(x34))
        q = torch.stack([q31, q32, q33, q34], dim=1)  # [batch, 4, feature_dim]
        q = F.normalize(q, p=2, dim=2)

        # 静态分支
        with torch.no_grad():
            target_feat = self.target_encoder(x2)
            target_proj = self.target_projector(target_feat)
            target_proj = F.normalize(target_proj, dim=1)

            # 分区并计算键向量
            top_left, top_right, bottom_left, bottom_right = self.quad_partition(target_feat)
            k31 = self.target_projector(top_left)
            k32 = self.target_projector(top_right)
            k33 = self.target_projector(bottom_left)
            k34 = self.target_projector(bottom_right)
            k = torch.stack([k31, k32, k33, k34], dim=1)  # [batch, 4, feature_dim]
            k = F.normalize(k, p=2, dim=2)

        return online_proj, target_proj, predictions, q, k

    def compute_losses(self, online_proj, target_proj, predictions, q, k, labels):
        # LCL 损失 (对比损失)
        logits = online_proj @ target_proj.T
        consistency_labels = torch.arange(online_proj.size(0), device=online_proj.device)
        consistency_loss = F.cross_entropy(logits, consistency_labels)

        # LLSP 损失 (预测损失)
        prediction_loss = F.binary_cross_entropy_with_logits(predictions, labels)

        # LDP 损失 (分解损失)
        similarity = torch.einsum('bik,bjk->bij', q, k)  # [batch, 4, 4]

        # 重塑为 [batch*4, 4]
        batch_size = similarity.size(0)
        logits = similarity.view(-1, 4)
        targets = torch.arange(4, device=logits.device).repeat(batch_size)
        decomposition_loss = F.cross_entropy(logits, targets)

        # 总损失
        total_loss = consistency_loss + 2 * prediction_loss + decomposition_loss

        # 返回详细损失
        loss_dict = {
            'consistency_loss': consistency_loss,
            'prediction_loss': prediction_loss,
            'decomposition_loss': decomposition_loss,  # 修复: 添加逗号
            'total_loss': total_loss
        }

        return loss_dict, total_loss