# model_reg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionNetwork(nn.Module):
    """
    支持三输入：
        x_lidar: [B, 360]  - LiDAR 扫描数据
        road_type: [B]     - 道路类型（可视为类别索引或连续值）
        turn_direction: [B] - 转向方向（可视为类别索引或连续值）
    输出：
        [B] - 回归角度（度）
    """
    def __init__(self, use_embedding=True):
        super().__init__()
        self.use_embedding = use_embedding

        # ========== LiDAR 特征提取 CNN ==========
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # ========== 嵌入或直接数值拼接 ==========
        if use_embedding:
            # 如果 road_type / turn_direction 是离散类别索引
            self.emb_road = nn.Embedding(10, 8)    # 假设最多10种道路类型
            self.emb_turn = nn.Embedding(5, 4)     # 假设最多5种转向类型
            in_dim = 64 + 8 + 4
        else:
            # 如果它们是连续数值
            in_dim = 64 + 2

        # ========== 回归层 ==========
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_lidar, road_type, turn_direction):
        # x_lidar: [B, 360]
        if x_lidar.dim() == 2:
            x_lidar = x_lidar.unsqueeze(1)  # -> [B,1,360]
        lidar_feat = self.features(x_lidar) # -> [B,64]

        if self.use_embedding:
            road_emb = self.emb_road(road_type.long())        # [B,8]
            turn_emb = self.emb_turn(turn_direction.long())   # [B,4]
            feat = torch.cat([lidar_feat, road_emb, turn_emb], dim=1)
        else:
            extra = torch.stack([road_type, turn_direction], dim=1).float()  # [B,2]
            feat = torch.cat([lidar_feat, extra], dim=1)

        out = self.regressor(feat)
        return out.squeeze(1)
