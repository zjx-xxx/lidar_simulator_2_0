# model_reg.py
import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    """
    1D-CNN 回归：
    - 输入: [B, 362]  (会在 forward 里自动变成 [B,1,362])
    - 采用自适应平均池化，避免大扁平层，参数更少，更抗过拟合
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(2),                  # L/2

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),                  # L/4

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),          # -> [B,64,1]
            nn.Flatten()                       # -> [B,64]
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, 362] 或 [B, 1, 362]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)                  # [B, 64]
        x = self.regressor(x)                # [B, 1]
        return x.squeeze(1)                  # [B]
