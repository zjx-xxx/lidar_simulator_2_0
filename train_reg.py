import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from model_reg import RegressionNetwork  # 你的新模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_mse_loss(pred, target, base_weight=1.0, angle_weight=10):
    weights = base_weight + angle_weight * torch.abs(target)
    loss = weights * (pred - target) ** 2
    return loss.mean()


class LidarRegressionDataset(Dataset):
    def __init__(self, X_lidar, road_types, turn_directions, y):
        self.X_lidar = torch.tensor(X_lidar, dtype=torch.float32)
        self.road_types = torch.tensor(road_types, dtype=torch.long).squeeze()
        self.turn_directions = torch.tensor(turn_directions, dtype=torch.long).squeeze()
        self.y = torch.tensor(y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X_lidar)

    def __getitem__(self, idx):
        return (self.X_lidar[idx],
                self.road_types[idx],
                self.turn_directions[idx],
                self.y[idx])


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            x_lidar, road_type, turn_direction, target = [b.to(device) for b in batch]
            outputs = model(x_lidar, road_type, turn_direction)
            batch_size = target.size(0)
            loss = weighted_mse_loss(outputs, target) * batch_size  # 总损失（加权）乘以样本数
            total_loss += loss.item()
            total_samples += batch_size
    return total_loss / total_samples


def train(model, train_data, val_data,
          num_epochs=3000, batch_size=64, learning_rate=0.001,
          early_stop_patience=10, min_loss_threshold=1.0):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            x_lidar, road_type, turn_direction, target = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(x_lidar, road_type, turn_direction)
            loss = weighted_mse_loss(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_lidar.size(0)

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader)

        print(f"Epoch {epoch:>4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ✅ 是否进入 early stop 区间
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, './model/model_regression_best.pth')
            print(f"✅ Best model updated and saved at epoch {epoch}")
        elif val_loss <= min_loss_threshold:
            # ✅ 只有 loss 已经足够低，才考虑 patience 停止
            patience_counter += 1
            print(f"🔁 No improvement, patience {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(
                    f"⏹ Early stopping: val_loss={val_loss:.4f} below threshold {min_loss_threshold}, but not improving")
                break
        else:
            # ❌ Loss 太大了，不允许停
            print(f"🔄 Val loss {val_loss:.4f} > min threshold {min_loss_threshold:.4f} — skipping early stop")
            patience_counter = 0

    # 保存最终模型（可选）
    torch.save(model.state_dict(), './model/model_regression_last.pth')
    print("🎯 Final model saved.")


if __name__ == '__main__':
    def load_data(prefix):
        X = pd.read_csv(f'./mydata/X_{prefix}.csv', header=None).values
        y = pd.read_csv(f'./mydata/direction/Y_{prefix}.csv', header=None).values
        road = pd.read_csv(f'./mydata/type/Y_{prefix}.csv', header=None).values
        turn = pd.read_csv(f'./mydata/towards/Y_{prefix}.csv', header=None).values
        return X, road, turn, y

    # 读取训练和验证集
    X_train, road_train, turn_train, y_train = load_data("train")
    X_val, road_val, turn_val, y_val = load_data("test")

    # 构建数据集
    train_dataset = LidarRegressionDataset(X_train, road_train, turn_train, y_train)
    val_dataset = LidarRegressionDataset(X_val, road_val, turn_val, y_val)

    # 初始化模型
    model = RegressionNetwork().to(device)

    # 启动训练
    train(model, train_dataset, val_dataset,
          num_epochs=3000, batch_size=64, learning_rate=0.001,
          early_stop_patience=15, min_loss_threshold=50000.0)
