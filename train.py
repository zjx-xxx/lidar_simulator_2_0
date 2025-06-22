import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from evaluate import *
from model import NeuralNetwork
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, X_train, y_train, num_epochs=10, batch_size=32, learning_rate=0.001):
    print(f'Training on {device}')
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_train_tensor = y_train_tensor.squeeze()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()

            # 计算 accuracy
            _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别
            correct += (predicted == target).sum().item()
            total += target.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        # print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')


if __name__ == '__main__':
    # 假设你有训练数据 X_train 和 y_train
    X_train = pd.read_csv('./mydata/X_train.csv', header=None)  # 加载特征数据
    # X_train.drop(X_train.columns[169:189], axis=1, inplace=True)#去除了车屁股部分的数据
    X_train = X_train.values
    y_train = pd.read_csv('./mydata/type/Y_train.csv', header=None).values  # 加载标签数据

    # 初始化模型并将其转移到GPU
    model = NeuralNetwork().to(device)

    num_epochs = 150
    batch_size = 64
    # 训练模型
    train(model, X_train, y_train, num_epochs, batch_size)

    # 读取X_test 和 Y_test
    X_test = pd.read_csv('./mydata/X_test.csv', header=None)  # 从CSV文件读取X_test
    # X_test.drop(X_test.columns[169:189], axis=1, inplace=True)#去除了车屁股部分的数据
    X_test = X_test.values
    y_test = pd.read_csv('./mydata/type/Y_test.csv', header=None).values.flatten()  # 从CSV文件读取Y_test

    # 评估模型
    acc = evaluate(model, X_test, y_test, device)

    while acc < 0.75:
        model = NeuralNetwork().to(device)  # 重新初始化模型并转移到GPU
        train(model, X_train, y_train, num_epochs, batch_size)
        acc = evaluate(model, X_test, y_test, device)

    # 保存训练好的模型
    torch.save(model.state_dict(), f'./model/model_{acc * 100:.2f}acc')
    torch.save(model.state_dict(), './model/model')
