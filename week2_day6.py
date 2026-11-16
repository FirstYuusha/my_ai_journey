import torch
import torch.nn as nn
import math

class MyAttention(nn.Module):
    def_init_(self,embed_dim,num_heads):
    super()._init_()#必须调用父类初始化
    self.embed_dim=embed_dim
    self.num_heads=num_heads
    self.head_dim=embed_dim//num_heads


    self.w_q=torch.Linear(embed_dim,embed_dim)
    self.w_k=torch.Linear(embed_dim,embed_dim)
    self.w_v=torch.Linear(embed_dim,embed_dim)
    self.w_o=torch.Linear(embed_dim,embed_dim)


    def forward(self,x):
        seq_len-x.shape[0]#x的形状，一次输入几个词，比如x为[3,8],[seq_len,embed_dim]

        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        Q_heads=Q.reshape(seq_len,num_heads,head_dim)
        K_heads=
        V_heads=