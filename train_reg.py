# =========================================================
# train_reg.py
# 三输入回归训练：LiDAR + road_type + turn_direction
# =========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_reg import RegressionNetwork  # 三输入版本

# =========================
# 设备与随机种子
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

# =========================
# 加权 MSE
# =========================
def weighted_mse_loss(pred, target, base_weight=1.0, angle_weight=10.0):
    weights = base_weight + angle_weight * torch.abs(target) / 30.0
    loss = weights * (pred - target) ** 2
    return loss.mean()

# =========================
# Dataset
# =========================
class LidarRegressionDataset(Dataset):
    def __init__(self, X_main, road_type, turn_direction, y):
        self.X_main = torch.tensor(X_main, dtype=torch.float32)
        self.road_type = torch.tensor(road_type, dtype=torch.float32).squeeze()
        self.turn_direction = torch.tensor(turn_direction, dtype=torch.float32).squeeze()
        self.y = torch.tensor(y, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.X_main)

    def __getitem__(self, idx):
        return (
            self.X_main[idx],
            self.road_type[idx],
            self.turn_direction[idx],
            self.y[idx],
        )

# =========================
# 评估函数
# =========================
@torch.no_grad()
def evaluate(model, dataloader,
             base_weight=1.0, angle_weight=10.0,
             hit_threshold_deg=3.0):
    model.eval()
    total_loss, total_mae, total_hit, total_samples = 0.0, 0.0, 0.0, 0

    for x_lidar, road_type, turn_direction, target in dataloader:
        x_lidar = x_lidar.to(device)
        road_type = road_type.to(device)
        turn_direction = turn_direction.to(device)
        target = target.to(device)

        outputs = model(x_lidar, road_type, turn_direction)

        loss = weighted_mse_loss(outputs, target,
                                 base_weight=base_weight,
                                 angle_weight=angle_weight)
        mae = torch.sum(torch.abs(outputs - target)).item()
        hit = torch.sum((torch.abs(outputs - target) < hit_threshold_deg).float()).item()

        bs = target.size(0)
        total_loss += loss.item() * bs
        total_mae  += mae
        total_hit  += hit
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_mae  = total_mae / total_samples
    hit_rate = total_hit / total_samples
    return avg_loss, avg_mae, hit_rate

# =========================
# 训练主循环
# =========================
def train(model, train_data, val_data,
          num_epochs=1000, batch_size=64, learning_rate=1e-3,
          early_stop_patience=50,
          base_weight=1.0, angle_weight=10.0):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_hit3": [], "lr": []}

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        running_loss, seen = 0.0, 0

        for x_lidar, road_type, turn_direction, target in train_loader:
            x_lidar = x_lidar.to(device)
            road_type = road_type.to(device)
            turn_direction = turn_direction.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(x_lidar, road_type, turn_direction)
            loss = weighted_mse_loss(outputs, target,
                                     base_weight=base_weight,
                                     angle_weight=angle_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x_lidar.size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)

        # 验证
        val_loss, val_mae, val_hit3 = evaluate(
            model, val_loader,
            base_weight=base_weight, angle_weight=angle_weight, hit_threshold_deg=3.0
        )

        scheduler.step(val_loss)

        # 记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_hit3"].append(val_hit3)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE(deg): {val_mae:.3f} | "
              f"Hit@3°: {val_hit3*100:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early Stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience = 0
            os.makedirs('./model', exist_ok=True)
            torch.save(best_state, './model/model_regression_best.pth')
            print(f"✅ Best model updated and saved at epoch {epoch}")
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {early_stop_patience} epochs)")
                break

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model_regression_last.pth')
    print("🎯 Final model saved.")
    return history

# =========================
# 画训练曲线
# =========================
def plot_history(history, out_path='./model/training_curves.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss vs. Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_loss.png')); plt.close()

    # MAE
    plt.figure()
    plt.plot(epochs, history["val_mae"], label='Val MAE (deg)')
    plt.xlabel('Epoch'); plt.ylabel('MAE (deg)'); plt.title('Validation MAE vs. Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_mae.png')); plt.close()

    # Hit@3°
    plt.figure()
    plt.plot(epochs, history["val_hit3"], label='Hit@3°')
    plt.xlabel('Epoch'); plt.ylabel('Hit Rate'); plt.title('Hit@3° vs. Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_hit3.png')); plt.close()

    # Learning Rate
    plt.figure()
    plt.plot(epochs, history["lr"], label='Learning Rate')
    plt.xlabel('Epoch'); plt.ylabel('LR'); plt.title('Learning Rate vs. Epoch')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_lr.png')); plt.close()

# =========================
# 数据读取函数
# =========================
def load_split(prefix):
    """
    返回：
      X_main: [N,360]
      path_type: [N,1]
      turn_direction: [N,1]
      y: [N,1]
    """
    X_main = pd.read_csv(f'./mydata/X_{prefix}.csv', header=None).values.astype(np.float32)
    path_type = pd.read_csv(f'./mydata/type/Y_{prefix}.csv', header=None).values.astype(np.float32)
    turn_direction = pd.read_csv(f'./mydata/towards/Y_{prefix}.csv', header=None).values.astype(np.float32)
    y = pd.read_csv(f'./mydata/direction/Y_{prefix}.csv', header=None).values.astype(np.float32)

    if path_type.ndim == 1: path_type = path_type.reshape(-1, 1)
    if turn_direction.ndim == 1: turn_direction = turn_direction.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    return X_main, path_type, turn_direction, y

# =========================
# 主入口
# =========================
if __name__ == '__main__':
    X_train, road_train, turn_train, y_train = load_split('train')
    X_val,   road_val,   turn_val,   y_val   = load_split('test')

    print('Train shapes:', X_train.shape, road_train.shape, turn_train.shape, y_train.shape)
    print('Val   shapes:', X_val.shape,   road_val.shape,   turn_val.shape,   y_val.shape)

    train_dataset = LidarRegressionDataset(X_train, road_train, turn_train, y_train)
    val_dataset   = LidarRegressionDataset(X_val,   road_val,   turn_val,   y_val)

    model = RegressionNetwork(use_embedding=True).to(device)

    history = train(model, train_dataset, val_dataset,
                    num_epochs=1000,
                    batch_size=64,
                    learning_rate=1e-3,
                    early_stop_patience=100,
                    base_weight=1.0,
                    angle_weight=10.0)

    plot_history(history, out_path='./model/training_curves.png')
    print('📈 Curves saved to ./model/training_curves_*')
