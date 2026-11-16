import torch
import torch.nn as nn
import math

class TRansformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,ff_dim):
        super().__init__()
        #多头注意力层，这里用Linear代指
        self.attn=nn.Linear(embed_dim,embed_dim)
        self.norm1=nn.LayerNorm(embed_dim)

        self.ffn=nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
        )
        self.norm2=nn.LayerNorm(embed_dim)

    def forward(self,x):
        #attention+残差+Norm
        attn_output=self.attn(x)
        x=x+attn_output
        x=self.norm1(x)
        #FFN+残差+norm,attention只是线性容器只负责融合没有记忆，FFn担任执行任务查字典和输出具体知识
        ffn_output=self.ffn(x)
        x=x+ffn_output
        x=self.norm2(x)

        return x

if __name__=="__main__":
    x=torch.randn(3,8)
    block=TRansformerBlock(embed_dim=8,num_heads=2,ff_dim=32)
    out=block(x)
    print("输入形状：",x.shape)
    print("输出形状：",out.shape)

           