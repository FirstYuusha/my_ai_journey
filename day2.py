import torch

print("=== 第2天：验证梯度方向 ===")

# 1. 造数据：真实规律是 y = 2 * x + 0（即真实 w=2, b=0）
x = torch.tensor([1.0])          # 输入
y_true = torch.tensor([2.0])     # 真实输出

# 2. 初始化参数：故意把 w 设成 3.0（比真实值 2.0 大）
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# 3. 前向传播：算预测值
y_pred = w * x + b

# 4. 算损失（误差的平方）
loss = (y_pred - y_true) ** 2

# 5. 反向传播！让电脑自动算梯度
loss.backward()

# 6. 打印梯度（这是今天的核心输出）,输出斜率，告诉哪个方向是shangpo
print(f"当前 w 的梯度: {w.grad.item():.4f}")
print(f"当前 b 的梯度: {b.grad.item():.4f}")