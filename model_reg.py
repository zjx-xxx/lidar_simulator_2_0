import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionNetwork(nn.Module):
    """
    LiDAR 360 点一维信号 + (道路类型, 转向方向) 的多模态回归网络
    - 卷积 Backbone 使用 circular padding，显式建模环结构
    - 对道路/转向模态进行增强：FiLM 调制 + 门控残差直通
    - 仍然输出一个标量（你原先是 angle 回归就保持不变；若需要sin/cos再说）
    """
    def __init__(
        self,
        use_embedding: bool = True,
        n_road: int = 10,
        n_turn: int = 5,
        d_road: int = 8,
        d_turn: int = 4,
        conv_channels=(16, 32, 64),
        kernel_size=7,
        dropout=0.3,
        out_dim=1,
    ):
        super().__init__()
        self.use_embedding = use_embedding

        # ===== 1) LiDAR Backbone：Conv1d + BN + GELU + MaxPool，全部 circular padding =====
        k = kernel_size
        p = k // 2  # 对称卷积，padding=k//2
        c1, c2, c3 = conv_channels

        self.backbone = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=k, padding=p, padding_mode='circular'),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(c1, c2, kernel_size=k, padding=p, padding_mode='circular'),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(c2, c3, kernel_size=k, padding=p, padding_mode='circular'),
            nn.BatchNorm1d(c3),
            nn.GELU(),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)   # -> [B, C, 1]
        self.proj = nn.Linear(c3, 64)                # 压到固定 64 维作为 LiDAR 全局表征
        self.proj_act = nn.GELU()

        # ===== 2) 嵌入或直接数值拼接（准备条件向量 cond）=====
        if use_embedding:
            # 如果 road_type / turn_direction 是离散类别索引
            self.emb_road = nn.Embedding(n_road, d_road)
            self.emb_turn = nn.Embedding(n_turn, d_turn)
            cond_dim = d_road + d_turn
        else:
            # 如果它们是连续数值（例如曲率/角度）
            cond_dim = 2

        # 把条件向量提炼一下，作为调制的“控制信号”
        # 这里用一个小 MLP，把 cond_dim -> 64（与 LiDAR 表征同维度，便于广播/融合）
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # ===== 3) FiLM 调制 + 门控残差直通，显式放大道路/转向模态的影响 =====
        # 由 cond 生成对 LiDAR 64 维特征的缩放/平移参数
        self.film_gamma = nn.Linear(64, 64)
        self.film_beta  = nn.Linear(64, 64)
        # 动态门控：学习一个 0~1 的权重，控制“调制后特征”与“原特征”的占比
        self.gate = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        # ===== 4) 回归头 =====
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),   # 缺省输出 1 维
        )

    def forward(self, x_lidar, road_type, turn_direction):
        """
        x_lidar: [B, 360] 或 [B, 1, 360]
        road_type, turn_direction:
            - use_embedding=True  -> int64 index 张量（[B] 或 [B,1]）
            - use_embedding=False -> float 连续值（[B] 或 [B,1]）
        """
        # -- 保障 LiDAR 维度 --
        if x_lidar.dim() == 2:
            x_lidar = x_lidar.unsqueeze(1)  # [B,1,L]
        # 允许 L != 360；环形卷积与自适应池化会兼容任意长度

        # ===== LiDAR 提特征（环形卷积）=====
        feat = self.backbone(x_lidar)        # [B, C, L']
        feat = self.global_pool(feat).squeeze(-1)  # [B, C]
        feat = self.proj_act(self.proj(feat))      # [B, 64]
        lidar_feat = feat                         # 保存一份原始 LiDAR 表征

        # ===== 条件向量 cond =====
        if self.use_embedding:
            road = road_type.long().view(-1)
            turn = turn_direction.long().view(-1)
            cond_raw = torch.cat([self.emb_road(road), self.emb_turn(turn)], dim=1)  # [B, d_road+d_turn]
        else:
            # 连续值：确保形状 [B, 2]
            road = road_type.float().view(-1, 1)
            turn = turn_direction.float().view(-1, 1)
            cond_raw = torch.cat([road, turn], dim=1)  # [B,2]

        cond = self.cond_mlp(cond_raw)  # [B,64]

        # ===== FiLM 调制 + 门控残差直通 =====
        gamma = self.film_gamma(cond)           # [B,64]
        beta  = self.film_beta(cond)            # [B,64]
        mod_lidar = gamma * lidar_feat + beta   # 调制后的 LiDAR 特征

        gate_w = self.gate(cond)                # [B,1] ∈ (0,1)
        fused = gate_w * mod_lidar + (1.0 - gate_w) * lidar_feat  # 门控融合，保证关键模态有“硬影响”

        # ===== 回归输出 =====
        out = self.regressor(fused)             # [B, out_dim]
        if out.shape[1] == 1:
            out = out.squeeze(1)                # [B]
        return out
