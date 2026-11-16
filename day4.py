import torch
import torch.nn as nn
import torch.optim as optim

print("=== 第4天（新Day1）：告别手动，拥抱 nn.Linear + optim ===\n")

# 1. 造数据（给两个样本，强行锁定真实规律 w=2, b=0）
x = torch.tensor([[1.0], [2.0]])          # 形状：(2, 1)
y_true = torch.tensor([[2.0], [4.0]])     # 形状：(2, 1)

# 2. 定义模型（这1行替换了你上周的 w 和 b 两个张量）
model = nn.Linear(1, 1)  # 输入1维，输出1维

# 3. 定义优化器（这1行替换了你上周手写的 w = w - lr * grad）
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 定义损失函数（这1行替换了你上周的 loss = (y_pred - y_true) ** 2）
loss_fn = nn.MSELoss()

# 5. 训练循环（结构跟你上周完全一致，只是把内部手动操作换成了API）
for step in range(500):  # 多跑几步，确保收敛
    # 清零梯度（对应你上周手写的 w.grad.zero_()）
    optimizer.zero_grad()
    
    # 前向传播（算预测值）
    y_pred = model(x)
    
    # 算损失
    loss = loss_fn(y_pred, y_true)
    
    # 反向传播（算梯度，对应你上周的 loss.backward()）
    loss.backward()
    
    # 参数更新（对应你上周的 w -= lr * w.grad，这就是传说中的“自动挡”）
    optimizer.step()
    
    # 每50步打印一次
    if step % 50 == 0:
        # 看看模型学到的 w 和 b 是多少
        w = model.weight.item()
        b = model.bias.item()
        print(f"Step {step:3d} | Loss: {loss.item():.6f} | w: {w:.4f} | b: {b:.4f}")

print("\n训练完成！")
print(f"最终参数：w = {model.weight.item():.4f}，b = {model.bias.item():.4f}")
print("真实值：w = 2.0，b = 0.0")