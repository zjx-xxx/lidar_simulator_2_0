import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 加深网络层次，增强表达能力
        self.fc1 = nn.Linear(360, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.3)  # 防止过拟合

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(32, 4)  # 输出为4类 logits（不用Softmax）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x
