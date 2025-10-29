
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from model_reg import RegressionNetwork
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =========================
# 设备
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 加权 MSE（与原版一致）
# =========================
def weighted_mse_loss(pred, target, base_weight=1.0, angle_weight=10.0):
    """
    给转角不为0的数据一个更高的权重，用于处理数据不平衡。
    权重 = base_weight + angle_weight * abs(真实角度/30)   # 归一化到[-1,1]范围的权重因子
    """
    weights = base_weight + angle_weight * torch.abs(target) / 30.0
    loss = weights * (pred - target) ** 2
    return loss.mean()


# =========================
# 数据集（与原版一致）
# =========================
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


# =========================
# 评估：Loss / MAE / Hit@3°
# =========================
@torch.no_grad()
def evaluate(model, dataloader,
             base_weight=1.0, angle_weight=10.0,
             hit_threshold_deg=3.0):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_hit = 0.0
    total_samples = 0

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data).squeeze()

        loss = weighted_mse_loss(outputs, target,
                                 base_weight=base_weight,
                                 angle_weight=angle_weight)

        mae = torch.sum(torch.abs(outputs - target)).item()
        hit = torch.sum((torch.abs(outputs - target) < hit_threshold_deg).float()).item()

        bs = target.size(0)
        total_loss += loss.item() * bs
        total_mae += mae
        total_hit += hit
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_mae = total_mae / max(total_samples, 1)
    hit_rate = total_hit / max(total_samples, 1)
    return avg_loss, avg_mae, hit_rate


# =========================
# 训练主循环（加入验证/早停/调度/记录历史）
# =========================
def train(model, train_data, val_data,
          num_epochs=1000, batch_size=64, learning_rate=1e-3,
          early_stop_patience=50,
          base_weight=1.0, angle_weight=10.0):

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_hit3": [],
        "lr": []
    }

    best_val_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        seen = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = weighted_mse_loss(outputs, target,
                                     base_weight=base_weight,
                                     angle_weight=angle_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = data.size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)

        # 验证
        val_loss, val_mae, val_hit3 = evaluate(model, val_loader,
                                               base_weight=base_weight,
                                               angle_weight=angle_weight,
                                               hit_threshold_deg=3.0)

        # 学习率调度看 val_loss
        scheduler.step(val_loss)

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_hit3"].append(val_hit3)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val MAE(deg): {val_mae:.3f} | "
              f"Hit@3°: {val_hit3*100:.1f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停
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
                print(f"⏹ Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # 恢复最佳并保存最终
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs('./model', exist_ok=True)
    torch.save(model.state_dict(), './model/model_regression_last.pth')
    print("🎯 Final model saved.")
    return history


# =========================
# 画图
# =========================
def plot_history(history, out_path='./model/training_curves.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # 曲线1：Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_loss.png'))
    plt.close()

    # 曲线2：Val MAE
    plt.figure()
    plt.plot(epochs, history["val_mae"], label='Val MAE (deg)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (deg)')
    plt.title('Validation MAE vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_mae.png'))
    plt.close()

    # 曲线3：Hit@3°
    plt.figure()
    plt.plot(epochs, history["val_hit3"], label='Hit@3°')
    plt.xlabel('Epoch')
    plt.ylabel('Hit Rate')
    plt.title('Hit@3° vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_hit3.png'))
    plt.close()

    # 曲线4：LR
    plt.figure()
    plt.plot(epochs, history["lr"], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.replace('.png', '_lr.png'))
    plt.close()


# =========================
# 数据载入
# =========================
def load_split(prefix):
    """
    约定：
      ./mydata/X_{prefix}.csv               -> LiDAR 360
      ./mydata/direction/Y_{prefix}.csv    -> 角度（度）
      ./mydata/type/Y_{prefix}.csv         -> 道路类型（可选，未使用）
      ./mydata/towards/Y_{prefix}.csv      -> 转向方向（可选，未使用）
    对齐旧版：X = [X_main, path_type, turn_direction] (共 362维)
    """
    X_main = pd.read_csv(f'./mydata/X_{prefix}.csv', header=None).values
    path_type = pd.read_csv(f'./mydata/type/Y_{prefix}.csv', header=None).values
    turn_direction = pd.read_csv(f'./mydata/towards/Y_{prefix}.csv', header=None).values
    X = np.hstack([X_main, path_type, turn_direction])
    y = pd.read_csv(f'./mydata/direction/Y_{prefix}.csv', header=None).values
    return X, y


# =========================
# 入口
# =========================
if __name__ == '__main__':
    # 读取训练/验证集（train/test 命名与另一个脚本保持一致）
    X_train, y_train = load_split('train')
    X_val,   y_val   = load_split('test')

    # 构建数据集/加载器
    train_dataset = LidarRegressionDataset(X_train, y_train)
    val_dataset   = LidarRegressionDataset(X_val,   y_val)

    # 初始化模型（保持你原本的 RegressionNetwork 定义）
    model = RegressionNetwork().to(device)

    # 训练
    history = train(model, train_dataset, val_dataset,
                    num_epochs=3000,
                    batch_size=64,
                    learning_rate=1e-3,
                    early_stop_patience=100,
                    base_weight=1.0,
                    angle_weight=10.0)

    # 画曲线
    plot_history(history, out_path='./model/training_curves.png')
    print('📈 Curves saved to ./model/training_curves_*')
