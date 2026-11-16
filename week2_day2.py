import torch
import torch.nn as nn
x=torch.randn(3,8)
print("输入的形状：",x.shape)

w_q=nn.Linear(8,8)
w_k=nn.Linear(8,8)
w_v=nn.Linear(8,8)

Q=w_q(x)
K=w_k(x)
V=w_v(x)

print("Q的形状",Q.shape)
print("K的形状",K.shape)
print("V的形状",V.shape)

