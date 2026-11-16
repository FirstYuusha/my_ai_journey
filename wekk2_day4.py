import torch
import torch.nn as nn
import math

x=torch.randn(3,8)

w_q=nn.Linear(8,8)
w_k=nn.Linear(8,8)
w_v=nn.Linear(8,8)

Q=w_q(x)
K=w_k(x)
V=w_v(x)
print("Q/K/V形状",Q.shape)

score=Q@K.transpose(-2,-1)#-1是列维度，-2是行维度
print("\nscore的形状：",score.shape)
d_k=8#8是K的维度
scaled_score=score/math.sqrt(d_k)

attention_weight=torch.softmax(scaled_score,dim=-1)#-1对每一行做softmax，沿着右方向遍历
print("注意力权重的形状：",attention_weight.shape)
print("权重矩阵第一行的和：",attention_weight[0].sum().item())#item把张量转换成数字

output=attention_weight@V#最后出来的soft和v做乘法
print("最终形状：",output.shape)
print("\n最终输出:\n",output)