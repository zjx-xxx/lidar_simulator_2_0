# model_reg.py
import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        ff: int = 256,
        num_layers: int = 4,
        max_len: int = 360,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len

        # 序列标量投影到 d_model
        self.input_proj = nn.Linear(1, d_model)
        # 可学习位置编码
        self.pos_emb = nn.Embedding(max_len, d_model)

        # 4 层 Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 保留原先的道路与转向嵌入（维度保持 4）
        self.road_emb = nn.Embedding(4, 4)      # 道路类型
        self.towards_emb = nn.Embedding(3, 4)   # 转向方向

        # 与原先 FC 头部规模对齐：d_model + 8 → 64 → 1
        self.fc = nn.Sequential(
            nn.Linear(d_model + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, road_type: torch.Tensor, turn_direction: torch.Tensor) -> torch.Tensor:
        # x: [B, 360]
        x = x.float()
        B, L = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds max_len={self.max_len}")

        h = self.input_proj(x.unsqueeze(-1))               # [B, L, d_model]
        pos = self.pos_emb(torch.arange(L, device=x.device))   # [L, d_model]
        h = h + pos.unsqueeze(0)

        h = self.encoder(h)                                # [B, L, d_model]
        h = self.norm(h)
        h = h.mean(dim=1)                                  # [B, d_model]

        road_vec = self.road_emb(road_type)                # [B, 4]
        towards_vec = self.towards_emb(turn_direction)     # [B, 4]
        h = torch.cat([h, road_vec, towards_vec], dim=1)   # [B, d_model+8]

        out = self.fc(h).squeeze(1)                        # [B]
        return out
