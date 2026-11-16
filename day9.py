import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

print("=== 第6天：批量大小（Batch Size）实验 ===\n")

# 1. 生成数据集（跟 Day 4/5 完全一样）
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=80, random_state=42   # 这次用 80 个训练，避免过拟合太严重
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print(f"训练集大小: {len(X_train_t)} 个样本")
print(f"测试集大小: {len(X_test_t)} 个样本\n")

# 2. 定义一个函数，用于训练模型（参数：batch_size）
def train_with_batch_size(batch_size, lr=0.1, epochs=500):
    # 模型结构跟 Day 4 一样，但只保留 2 层（避免过拟合干扰实验）
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    # 使用 DataLoader 来按批量喂数据
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(X_train_t)  # 平均每个样本的损失

        # 计算测试 Loss
        with torch.no_grad():
            logits_test = model(X_test_t)
            loss_test = loss_fn(logits_test, y_test_t).item()

        train_losses.append(epoch_loss)
        test_losses.append(loss_test)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {epoch_loss:.4f} | Test Loss: {loss_test:.4f}")

    return train_losses, test_losses

# 3. 实验 1：Batch Size = 1（纯随机，每次只算 1 个样本）
print("--- 实验 1：Batch Size = 1（过山车模式） ---")
train_losses_1, test_losses_1 = train_with_batch_size(batch_size=1, epochs=500)

# 4. 实验 2：Batch Size = 全部数据（全批量，每次算全部 80 个样本）
print("\n--- 实验 2：Batch Size = 全部数据（蜗牛模式） ---")
train_losses_all, test_losses_all = train_with_batch_size(batch_size=len(X_train_t), epochs=500)

# 5. 画图对比
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses_1, label='Batch Size=1 (训练)', color='blue', alpha=0.7)
plt.plot(test_losses_1, label='Batch Size=1 (测试)', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Batch Size=1：噪声大，震荡剧烈')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses_all, label='Batch Size=全部 (训练)', color='blue', alpha=0.7)
plt.plot(test_losses_all, label='Batch Size=全部 (测试)', color='red', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Batch Size=全部：平滑但收敛慢')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n【核心结论】")
print("1. Batch Size = 1：Loss 曲线像过山车（噪声大），但收敛速度快（因为它频繁更新）。")
print("2. Batch Size = 全部数据：Loss 曲线平滑，但每轮更新一次，收敛极慢。")
print("3. 大模型（如 ChatGPT）通常取中间值（如 32、64、128），在噪声和速度之间取平衡。") 