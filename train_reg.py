# train_reg.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from model_reg import RegressionNetwork
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# CUDA 检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weighted_mse_loss(pred, target, base_weight=1.0, angle_weight=10.0):
    """
    给转角不为0的数据一个更高的权重，用于处理数据不平衡。
    - base_weight: 所有样本的基础权重
    - angle_weight: 与角度大小相关的额外权重
    """
    # 权重 = 1 + angle_weight * abs(真实角度)
    weights = base_weight + angle_weight * torch.abs(target)
    loss = weights * (pred - target) ** 2
    return loss.mean()

# 回归数据集（无分类 augment）
class LidarRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

# 回归训练函数
def train(model, X_train, y_train, num_epochs=500, batch_size=64, learning_rate=0.001):
    print(f'Training on {device}')

    train_dataset = LidarRegressionDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.MSELoss()使用加权时无需criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)  # 输出 shape: [B]
            # loss = criterion(outputs, target)
            loss = weighted_mse_loss(outputs, target)#加权损失函数
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

if __name__ == '__main__':
    # 读取雷达数据（360维）
    X_main = pd.read_csv('./mydata/X_train.csv', header=None).values  # shape: [N, 360]

    # 读取路径类型（1维）和转角方向（1维）
    path_type = pd.read_csv('./mydata/type/Y_train.csv', header=None).values  # shape: [N, 1]
    turn_direction = pd.read_csv('./mydata/towards/Y_train.csv', header=None).values  # shape: [N, 1]

    # 拼接为新的输入 [N, 362]
    X_train = np.hstack([X_main, path_type, turn_direction])

    # 加载输出标签
    y_train = pd.read_csv('./mydata/direction/Y_train.csv', header=None).values  # shape: [N, 1]

    # 初始化模型
    model = RegressionNetwork().to(device)

    # 训练模型
    train(model, X_train, y_train)

    # 保存模型
    torch.save(model.state_dict(), './model/model_regression.pth')
