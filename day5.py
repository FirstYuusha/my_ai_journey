import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

print("=== 第5天（新Day2）：引入非线性（ReLU） ===\n")

# 1. 造一个“弯弯曲曲”的数据（非线性！）
# 真实规律：y = x^2 + 0.5*x + 0.1，这绝对不是直线
x = torch.linspace(-1, 1, 100).reshape(-1, 1)  # 100个点，形状(100,1)，从-1开始到1 end 随机生成100 个数字，-1 自动匹配行数，就固定1列的列数
y_true = x**2 + 0.5*x + 0.1                     # 真实的弯曲曲线

# 2. 定义两个模型：
#   - 模型A（没有ReLU）：两层Linear叠在一起，等价于一层，只能画直线
#   - 模型B（有ReLU）：真正的2层网络，能画曲线

print("--- 模型A：两层Linear堆叠（无激活函数） ---")
model_A = nn.Sequential(
    nn.Linear(1, 10),#列，把每一行的10个数变成10个数（w和b个数），10个函数
    nn.Linear(10, 1)   # 注意：中间没有任何激活函数！
)

print("--- 模型B：两层Linear + ReLU（有激活函数） ---")
model_B = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),          # 关键灵魂！就是它让模型能画曲线
    nn.Linear(10, 1)
)

# 3. 训练函数（复用昨天的自动挡流程）
def train_model(model, x, y_true, lr=0.01, epochs=9000):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    return model

# 4. 分别训练两个模型
print("\n训练模型A（无ReLU）...")
model_A = train_model(model_A, x, y_true)

print("\n训练模型B（有ReLU）...")
model_B = train_model(model_B, x, y_true)

# 5. 画图对比（可视化！）
with torch.no_grad():
    y_pred_A = model_A(x)
    y_pred_B = model_B(x)

# 用matplotlib画图（你之前说不碰Pandas，但matplotlib是画图必需品，这次破例用一次）
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(x.numpy(), y_true.numpy(), label='真实曲线', s=5, color='blue')
plt.plot(x.numpy(), y_pred_A.numpy(), label='模型A（无ReLU）', color='red', linewidth=2)
plt.title('模型A：两层Linear堆叠 = 直线')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y_true.numpy(), label='真实曲线', s=5, color='blue')
plt.plot(x.numpy(), y_pred_B.numpy(), label='模型B（有ReLU）', color='green', linewidth=2)
plt.title('模型B：两层Linear + ReLU = 曲线')
plt.legend()

plt.tight_layout()
plt.show()

print("\n结论：")
print("- 模型A（无ReLU）只能画直线，因为它等价于一个单层Linear(1,1)")
print("- 模型B（有ReLU）能画出弯曲的曲线，因为ReLU引入了非线性")
print("- 这就是大模型里FFN（前馈网络）的核心结构：Linear -> ReLU -> Linear")