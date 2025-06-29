import torch
import torch.nn as nn

class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),  # 输入1通道，输出16通道，保持长度不变（362->362-1=361 近似）
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), # 长度仍为361
            nn.ReLU(),
            nn.MaxPool1d(2)  # 长度减半: 361//2=180
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 181, 64),  # 改成5792
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 把输入扩展成 [B, 1, 362]
        x = self.conv(x)    # 输出 [B, 32, 181]
        x = self.fc(x)      # 输出 [B, 1]
        return x.squeeze(1) # 返回 [B]
