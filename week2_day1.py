import torch
import torch.nn as nn
import math
 

vocal_size=10#词表有10个词
embed_dim=8#每个词维度8

#词嵌入层
embed=nn.Embedding(vocal_size,embed_dim)
#模拟一句话，索引123
input_ids=torch.tensor([1,2,3])
#索引变向量
x=embed(input_ids)
print("输入序列形状：",x.shape)


#初始化参数 
W_q=nn.Linear(embed_dim,embed_dim)
W_k=nn.Linear(embed_dim,embed_dim)
W_v=nn.Linear(embed_dim,embed_dim)

Q=W_q(x)
K=W_k(x)
V=W_v(x)
print("Q地形状：",Q.shape)

score=Q@K.transpose(-2,-1)
print("Score的形状:",score.shape)

d_k=embed_dim
scaled_score=score/math.sqrt(d_k)

attention_weight=torch.softmax(scaled_score,dim=-1)
print("注意力权重形状：",attention_weight.shape)
print("权重矩阵第一行：\n",attention_weight[0])

output=attention_weight@V
print("最终形状：",output.shape)
