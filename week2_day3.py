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
print("缩放后的score形状：",scaled_score.shape)
print("缩放后的分数矩阵形状：",scaled_score.shape)
