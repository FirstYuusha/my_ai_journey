import torch
x=torch.tensor([1.0])
y_teue=torch.tensor([2.0])

w=torch.tensor([3.0],requires_grad=True)
b=torch.tensor([0.0],requires_grad=True)

i=0.01
for step in range(50):
    y_pred=wx+b
    loss=(y_pred-y_teue)**2
    loss.backward()
    with torch.no_grad():
        w=w-l*w.grad
        b=b-l*b,grad
        w.grad.zero_()
        b.grad.zero_()
    if step % 10 == 0:
            print(f"Step {step:2d} | Loss: {loss.item():.6f} | w: {w.item():.4f} | b: {b.item():.4f}")
    
print(f"\n最终结果：w = {w.item():.4f}，b = {b.item():.4f}")
print("真实值：w = 2.0，b = 0.0")