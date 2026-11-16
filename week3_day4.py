import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #引入多头注意力，昨天封装的，现在yonglinear模拟下
        self.attn=nn.Linear(embed_dim,embed_dim)

        #层归一化Layernorm
        #传入embed_dim(8),意思是对每个词的8个特征进行归一化
        self.norm1=nn.LayerNorm(embed_dim)

    def forward(self,x):
        #计算attention
        attn_output=self.attn(x)
        #残差链接，把输入x和attention直接相加
        x=x+attn_output
        #经过layernorm归一化
        x=self.norm1(x)
        return x

if __name__=="__main__":
    x=torch.randn(3,8)
    block=TransformerBlock(embed_dim=8,num_heads=2)
    out=block(x)
    print("输入形状：",x.shape)
    print("输出形状：",out.shape)