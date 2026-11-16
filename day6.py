import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

print("=== 第3天：非线性二分类（make_moons） ===\n")

# 1. 生成数据集：两个交错的新月（各50个点，共100个点）
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
# X 形状 (100, 2)  → 每个样本有2个特征（x坐标, y坐标）
# y 形状 (100,)   → 每个样本的标签（0 或 1）

# 转为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # 形状变为 (100, 1)

# 2. 定义模型（输入2个特征，隐藏层64个神经元，输出1个logit）
model = nn.Sequential(
    nn.Linear(2, 64),   # 输入层：2个坐标
    nn.ReLU(),          # 非线性灵魂
    nn.Linear(64, 1)    # 输出层：1个logit（未经过Sigmoid）
)

# 3. 定义损失函数和优化器
# BCEWithLogitsLoss = 内置Sigmoid + 二分类交叉熵（数值稳定）
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练循环
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X_tensor)          # 输出原始logit（未压缩到0-1）
    loss = loss_fn(logits, y_tensor)  # 自动计算Sigmoid+交叉熵
    loss.backward()
    optimizer.step()
    
    if epoch % 400 == 0:
        # 计算当前准确率
        probs = torch.sigmoid(logits)  # 转为概率 (0~1)
        preds = (probs > 0.5).float()
        acc = (preds == y_tensor).float().mean().item() * 100
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Acc: {acc:.1f}%")

print("\n训练完成！绘制决策边界...")

# 5. 可视化：绘制决策边界（核心灵魂！）
def plot_decision_boundary(model, X, y):
    # 生成网格点
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 将网格点转为张量，输入模型预测
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        logits = model(grid)
        probs = torch.sigmoid(logits).numpy().reshape(xx.shape)
    
    # 画彩色背景（决策区域）
    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
    # 画原始数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
    plt.title('两层网络 (Linear+ReLU+Linear) 分类结果')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.show()

plot_decision_boundary(model, X, y)
print("蓝色和红色区域就是模型学到的分类边界。两个交错的月亮被分开了！")