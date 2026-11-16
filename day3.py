import torch

print("=== 第3天：手写下山循环（绝对不用 optimizer.step()） ===\n")

# 真实规律：y = 2 * x + 0
x = torch.tensor([[10.0], [20.0], [30.0]])
y_true = torch.tensor([[20.0], [40.0], [60.0]])

# 初始参数（故意设偏，w=3, b=0）
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# 学习率（先设为 0.1，待会改成 0.00001 和 0.5 做灵魂实验）
lr = 0.001

# 开始下山！循环 50 步
for step in range(500):
    # 1. 前向传播：算预测值
    y_pred = w * x + b
    
    # 2. 算损失
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 3. 反向传播：算梯度（电脑告诉你哪边是上坡）
    loss.backward()
    
    # 4. 【铁血禁令】必须手写更新，绝不用 optimizer.step()
    #    朝梯度的反方向走（减号就是你的意志！）
    with torch.no_grad():  # 切断计算图，纯粹做算术
        w -= lr * w.grad
        b -= lr * b.grad
        
        # 5. 重要！手动清零梯度，否则下次会累加（这是手写时的唯一坑）
        w.grad.zero_()
        b.grad.zero_()
    
    # 每 10 步打印一次 Loss
    if step % 10 == 0:
        print(f"Step {step:2d} | Loss: {loss.item():.6f} | w: {w.item():.4f} | b: {b.item():.4f}")

print(f"\n最终结果：w = {w.item():.4f}，b = {b.item():.4f}")
print("真实值：w = 2.0，b = 0.0")