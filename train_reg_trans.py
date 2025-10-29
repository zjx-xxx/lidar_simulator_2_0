# train_reg.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model_reg import RegressionNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 数据集
# =========================
class LidarRegressionDataset(Dataset):
    def __init__(self, X_lidar, road_types, turn_directions, y_deg, max_clip=None):
        """
        X_lidar: [N, 360] 原始距离（建议传入前先做裁剪/归一化）
        road_types: [N] 0..3
        turn_directions: [N] 0..2
        y_deg: [N] 目标角度(度)，建议在制作数据时已裁剪到 [-30,30]
        max_clip: 若提供，例如 6.0 (m)，则会将 X>6 裁剪到 6
        """
        X = np.asarray(X_lidar, dtype=np.float32)
        if max_clip is not None:
            X = np.minimum(X, max_clip)

        # 距离归一化：除以上限使落在[0,1]（如用 max_clip，否则用自身max避免全0）
        if max_clip is not None and max_clip > 0:
            X = X / max_clip
        else:
            # 兜底：按样本内最大值归一化，防止全0
            max_per_row = np.maximum(X.max(axis=1, keepdims=True), 1e-6)
            X = X / max_per_row

        self.X_lidar = torch.tensor(X, dtype=torch.float32)
        self.road_types = torch.tensor(road_types, dtype=torch.long).squeeze()
        self.turn_directions = torch.tensor(turn_directions, dtype=torch.long).squeeze()
        self.y_deg = torch.tensor(y_deg, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X_lidar)

    def __getitem__(self, idx):
        return (self.X_lidar[idx],
                self.road_types[idx],
                self.turn_directions[idx],
                self.y_deg[idx])


# =========================
# 损失函数：Huber（度数域）+ 温和权重
# =========================
def weighted_huber_loss_deg(pred_deg, target_deg,
                            base_weight: float = 1.0,
                            angle_weight: float = 1.0,
                            delta: float = 1.0):
    """
    pred_deg/target_deg: 度数（-30, 30）
    delta: Huber 转折点（度）。1~2度常用；数据噪声大可以取大一些。
    权重：w = base_weight + angle_weight * (|y|/30)
    """
    diff = pred_deg - target_deg
    abs_diff = torch.abs(diff)
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    )
    w = base_weight + angle_weight * (torch.abs(target_deg) / 30.0)
    return (w * huber).mean()


# =========================
# 评估
# =========================
def evaluate(model, dataloader,
             base_weight=1.0, angle_weight=1.0, delta=1.0,
             hit_threshold_deg=3.0):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_hit = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            x_lidar, road_type, turn_direction, target_deg = [b.to(device) for b in batch]
            outputs_deg = model(x_lidar, road_type, turn_direction)

            batch_size = target_deg.size(0)
            loss = weighted_huber_loss_deg(outputs_deg, target_deg,
                                           base_weight=base_weight,
                                           angle_weight=angle_weight,
                                           delta=delta)

            mae = torch.sum(torch.abs(outputs_deg - target_deg)).item()
            hit = torch.sum((torch.abs(outputs_deg - target_deg) < hit_threshold_deg).float()).item()

            total_loss += loss.item() * batch_size
            total_mae += mae
            total_hit += hit
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    hit_rate = total_hit / total_samples
    return avg_loss, avg_mae, hit_rate


# =========================
# 训练主循环
# =========================
def train(model, train_data, val_data,
          num_epochs=1000, batch_size=64, learning_rate=1e-3,
          early_stop_patience=50,
          base_weight=1.0, angle_weight=1.0, delta=1.0):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        for batch in train_loader:
            x_lidar, road_type, turn_direction, target_deg = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs_deg = model(x_lidar, road_type, turn_direction)

            loss = weighted_huber_loss_deg(outputs_deg, target_deg,
                                           base_weight=base_weight,
                                           angle_weight=angle_weight,
                                           delta=delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_lidar.size(0)
            running_loss += loss.item() * bs
            seen_samples += bs

        train_loss = running_loss / max(seen_samples, 1)

        # 验证
        val_loss, val_mae, val_hit3 = evaluate(model, val_loader,
                                               base_weight=base_weight,
                                               angle_weight=angle_weight,
                                               delta=delta,
                                               hit_threshold_deg=3.0)
        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE(deg): {val_mae:.3f} | "
              f"Hit@3°: {val_hit3*100:.1f}%")

        # 早停（无阈值，只看提升）
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            os.makedirs('./model', exist_ok=True)
            torch.save(best_model_state, './model/model_regression_best.pth')
            print(f"✅ Best model updated and saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # 保存最终模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model_regression_last.pth')
    print("🎯 Final model saved.")


# =========================
# 入口
# =========================
def load_data(prefix):
    # 你的目录结构约定：
    # ./mydata/X_{prefix}.csv               -> LiDAR 360 距离
    # ./mydata/direction/Y_{prefix}.csv    -> 角度标签（度）[-30,30]
    # ./mydata/type/Y_{prefix}.csv         -> 道路类型 0..3
    # ./mydata/towards/Y_{prefix}.csv      -> 转向方向 0..2
    X = pd.read_csv(f'./mydata/X_{prefix}.csv', header=None).values
    y = pd.read_csv(f'./mydata/direction/Y_{prefix}.csv', header=None).values
    road = pd.read_csv(f'./mydata/type/Y_{prefix}.csv', header=None).values
    turn = pd.read_csv(f'./mydata/towards/Y_{prefix}.csv', header=None).values
    return X, road, turn, y


if __name__ == '__main__':
    # 读取训练/验证
    X_train, road_train, turn_train, y_train = load_data("train")
    X_val, road_val, turn_val, y_val = load_data("test")

    print("Train y(min/max/mean/std):",
          y_train.min(), y_train.max(), y_train.mean(), y_train.std())
    print("Test  y(min/max/mean/std):",
          y_val.min(), y_val.max(), y_val.mean(), y_val.std())

    # 构建数据集
    # 如果你知道 LiDAR 的“有效上限距离”（如 6 米），建议设置 max_clip=6.0
    train_dataset = LidarRegressionDataset(X_train, road_train, turn_train, y_train, max_clip=6.0)
    val_dataset   = LidarRegressionDataset(X_val,   road_val,   turn_val,   y_val,   max_clip=6.0)

    # 初始化模型
    model = RegressionNetwork(
        d_model=128,
        nhead=8,
        ff=256,
        num_layers=4,
        max_len=360,
        dropout=0.1,
        max_angle=30.0,   # 输出范围 [-30,30] 度
    ).to(device)

    # 训练
    train(model, train_dataset, val_dataset,
          num_epochs=3000,
          batch_size=64,
          learning_rate=1e-3,
          early_stop_patience=100,
          base_weight=1.0,
          angle_weight=1.0,
          delta=1.0)
