import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)  # [B, 32, 180]
        )

        self.road_emb = nn.Embedding(4, 4)      # 道路类型
        self.towards_emb = nn.Embedding(3, 4)   # 转向方向

        self.fc = nn.Sequential(
            nn.Linear(32 * 180 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x, road_type, turn_direction):
        x = x.unsqueeze(1)  # [B, 1, 360]
        x = self.conv(x)    # [B, 32, 180]
        x = x.flatten(1)    # [B, 32*180]

        road_vec = self.road_emb(road_type)         # [B, 4]
        towards_vec = self.towards_emb(turn_direction)  # [B, 4]

        x = torch.cat([x, road_vec, towards_vec], dim=1)
        x = self.fc(x)  # [B, 1]
        return x.squeeze(1)  # [B]