import torch
import torch.nn as nn
import math

class MyMUltiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads

        self.w_q=nn.Linear(embed_dim,embed_dim)
        self.w_k=nn.Linear(embed_dim,embed_dim)
        self.w_v=nn.Linear(embed_dim,embed_dim)
        self.w_o=nn.Linear(embed_dim,embed_dim)

    def forward(self,x):
        #x的形状seq，embed_dimQ
        batch_size=x.shape[0]#这里的batch-size是序列长度（词数）
        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        #多头劈叉
        Q_split=Q.reshape(batch_size,self.num_heads,self.head_dim)
        K_split=K.reshape(batch_size,self.num_heads,self.head_dim)
        V_split=V.reshape(batch_size,self.num_heads,self.head_dim)

        #转换
        Q_heads=Q_split.transpose(0,1)
        K_heads=K_split.transpose(0,1)
        V_heads=V_split.transpose(0,1)

        #计算
        score=Q_heads@K_heads.transpose(-2,-1)
        scaled_score=score/math.sqrt(self.head_dim)
        attention_weight=torch.softmax(scaled_score,dim=-1)

        #加权求和
        output_heads=attention_weight@V_heads

        #拼接与融合
        output_transpose=output_heads.transpose(0,1)
        output_concat=output_transpose.contiguous().view(batch_size,self.embed_dim)

        output_final=self.w_o(output_concat)
        return output_final

#-------------------测试---------------
if __name__ =="__main__":
    x=torch.randn(3,8)
    attn=MyMUltiHeadAttention(embed_dim=8,num_heads=2)
    out=attn(x)
    print("输入形状",x.shape)
    print("输出形状",out.shape)
