# model.py

import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 第一层：Affine层 + ReLU激活函数
        self.fc1 = nn.Linear(360, 64)  # 输入360维，输出128维
        self.relu1 = nn.ReLU()

        # 输出层：Affine层 + Softmax激活函数
        self.fc4 = nn.Linear(64, 4)   # 从32维到输出3个类别
        # self.softmax = nn.Softmax(dim=1)  # Softmax激活，用于多分类问题

    def forward(self, x):
        x = self.fc1(x)  # 第一层Affine变换
        x = self.relu1(x)  # 第一层ReLU激活
        x = self.fc4(x)  # 输出层Affine变换
        return x
