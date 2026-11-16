import torch
import torch.nn as nn
import math

x=torch.randn(3,8)
w_q=nn.Linear(8,8)#输入维度8，输出维度8
w_k=nn.Linear(8,8)
w_v=nn.Linear(8,8)
w_o=nn.Linear(8,8)

Q=w_q(x)
K=w_k(x)
V=w_v(x)

seq_len=3
embed_dim=8
num_heads=2
head_dim=embed_dim//num_heads#每个头的维度是4
print("Q的初始形状：",Q.shape)

Q_split=Q.reshape(seq_len,num_heads,head_dim)#把QKV形状从（3，8）变成（3，2，4），三个词两个头每个头四维
K_split=K.reshape(seq_len,num_heads,head_dim)
V_split=V.reshape(seq_len,num_heads,head_dim)

Q_head=Q_split.transpose(0,1)#为了方便计算，词头维改成头词维
K_head=K_split.transpose(0,1)
V_head=V_split.transpose(0,1)
print("多头拆解Q的形状：",Q_head.shape)

score=Q_head@K_head.transpose(-2,-1)#遇到高于两维度的（行，列），照着最后两个维度做矩阵乘法，前面当作小batch
scaled_score=score/math.sqrt(head_dim)
attention_weight=torch.softmax(scaled_score,dim=-1)

output_head=attention_weight@V_head
print("多头输出形状：",output_head.shape)

output_transposed=output_head.transpose(0,1)
output_concat=output_transposed.reshape(3,8)#这个不是真实的，因为最后b11,b12,b13要乘一个w，最后变成一个，这个只是单纯的把头1和头2结果拼接到一起
print("拼接输出的形状：",output_concat.shape)
output_fianl=w_o(output_concat)#最后这个参数被定义成完整的网络模块了
print("最终输出的形状：",output_fianl.shape)