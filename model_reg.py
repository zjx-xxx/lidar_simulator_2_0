import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),                 # 362 -> 181

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),                 # 181 -> 90

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),         # [B,64,1]
            nn.Flatten()                      # [B,64]
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)                   # [B,1,L]  (L 可变)
        x = self.features(x)                 # [B,64]
        x = self.regressor(x)                # [B,1]
        return x.squeeze(1)
