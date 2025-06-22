import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)  # [B, 32, 180]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 180, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 360]
        x = self.conv(x)
        x = self.fc(x)
        return x
