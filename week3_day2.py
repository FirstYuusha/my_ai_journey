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

#计算分数
score=Q_heads@K_heads.transpose(-2,-1)
#缩放加soft
scaled_score=score/math.sqrt(head_dim)
attention_weight=torch.softmax(scaled_score)
#加权求和
output_heads=attention_weight@V_heads
print("多头计算各自的形状：",output_heads.shape)

#终极合并大法
#转回初始的样子
output_transpose=output_heads.transpose(0,1)
#直接拍扁不容合,刚才用了transpose数据在内存不连续，必须加contiguous，transpose只改变读取视角
output_concat=output_transpose.contiguous().view(seq_len,embed_dim)

#transpose小讲堂
#假设矩阵A=[[a,b],[c,d],[e,f]]形状是(3,2)
#他的物理内存是[a,b,c,d,e,f]不会变
#想要读取第二行第一列也就是c，他知道第一行有两个数所以直接读取第三个（索引是2），这个规则叫做步长
#现在对他做transpose（0，1），期望变成[[a,c,e][b,d,f]],形状（2，3）
#这里只是修改了步长,第一行步长是1，第二行步长是2，逻辑相连物理不相连
#view小讲堂
#只改变数据读取规则，而resahpe是先contiguous在进行切分
#contiguous小讲堂
#重新开辟一块空间，按照逻辑位置重新排列