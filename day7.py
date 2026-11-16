import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

print("=== 第4天：亲手制造过拟合（5层网络 vs 20个样本） ===\n")

# 1. 生成数据（造200个点，只用其中20个训练）
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# 强行拆分成：20个训练，180个测试（你绝对没看错，只用20个来训练！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=20, random_state=42
)

# 转为张量
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

print(f"训练集大小: {len(X_train_t)} 个样本")
print(f"测试集大小: {len(X_test_t)} 个样本")

# 2. 定义“超级过拟合模型”：5层网络，每层100个神经元（这就是大炮！）
class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)     # 输出层
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = OverfitModel()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. 疯狂训练 2000 轮
epochs = 2000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # --- 训练 ---
    optimizer.zero_grad()
    logits_train = model(X_train_t)
    loss_train = loss_fn(logits_train, y_train_t)
    loss_train.backward()
    optimizer.step()
    
    # --- 测试 ---
    with torch.no_grad():
        logits_test = model(X_test_t)
        loss_test = loss_fn(logits_test, y_test_t)
    
    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())
    
    # 打印结果
    if epoch % 400 == 0:
        # 计算训练准确率（看训练集是否完美）
        probs_train = torch.sigmoid(logits_train)
        preds_train = (probs_train > 0.5).float()
        acc_train = (preds_train == y_train_t).float().mean().item() * 100
        
        # 计算测试准确率（看泛化能力）
        probs_test = torch.sigmoid(logits_test)
        preds_test = (probs_test > 0.5).float()
        acc_test = (preds_test == y_test_t).float().mean().item() * 100
        
        print(f"Epoch {epoch:4d} | 训练 Loss: {loss_train.item():.4f} (Acc: {acc_train:.1f}%) | 测试 Loss: {loss_test.item():.4f} (Acc: {acc_test:.1f}%)")

print("\n训练完成！观察下面的现象：")

# 4. 画损失曲线（这是看“过拟合”的核心证据！）
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练 Loss', color='blue')
plt.plot(test_losses, label='测试 Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')  # 对数坐标，看得更清楚
plt.title('训练 Loss 下降，测试 Loss 上升（过拟合！）')
plt.legend()
plt.grid(True)

# 5. 画决策边界（看模型如何在 20 个点上“扭曲”）
plt.subplot(1, 2, 2)
# 画所有原始点（训练集 + 测试集，用颜色区分）
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdBu', edgecolors='k', s=100, label='训练集')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolors='k', s=20, alpha=0.5, label='测试集')

# 画决策边界（背景色）
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    logits_grid = model(grid)
    probs_grid = torch.sigmoid(logits_grid).numpy().reshape(xx.shape)

plt.contourf(xx, yy, probs_grid, levels=50, cmap='RdBu', alpha=0.3)
plt.title('5层网络强行扭曲边界去适应20个点')
plt.legend()
plt.tight_layout()
plt.show()

print("\n【核心结论】")
print("1. 训练 Loss 下降到接近 0（20 个点全部记对了）。")
print("2. 测试 Loss 反而上升（模型对没见过的 180 个点预测极差）。")
print("3. 决策边界极度扭曲——它为了‘死记’那 20 个点，把整片区域撕碎了。")