import torch
import torch.nn as nn
import math

#数据准备
seq_len=3#3个词
embed_dim=8#总维度8
num_heads=2#总共2个头
head_dim=embed_dim//num_heads#每个头4维
x=torch.randn(seq_len,embed_dim)
print("初始形状：",x.shape)

#定义权重矩阵
w_q=nn.Linear(embed_dim,embed_dim)
w_k=nn.Linear(embed_dim,embed_dim)
w_v=nn.Linear(embed_dim,embed_dim)
w_o=nn.Linear(embed_dim,embed_dim)

Q=w_q(x)
K=w_k(x)
V=w_v(x)

#多头劈叉
Q_split=Q.reshape(seq_len,num_heads,head_dim)
K_split=K.reshape(seq_len,num_heads,head_dim)
V_split=V.reshape(seq_len,num_heads,head_dim)

#劈叉后调整位置
Q_heads=Q_split.transpose(0,1)
K_heads=K_split.transpose(0,1)
V_heads=V_split.transpose(0,1)

print("拆分后Q的形状：",Q_heads.shape)