import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,ff_dim):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.norm1=nn.LayerNorm(embed_dim)

        self.ffn=nn.Sequential(
            nn.Linear(embed_dim,ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_dim)
        )
        self.norm2=nn.LayerNorm(embed_dim)

    def forward(self,x):
        #x形状 batch_size，seq_len,embed_dim
        #官方返回值是attn_output，attn_weight
        attn_output,_=self.attn(x,x,x)
        x=x+attn_output
        x=self.norm1(x)

        ffn_output=self.ffn(x)
        x=x+ffn_output
        x=self.norm2(x)
        return x

if __name__=="__main__":
    x=torch.randn(2,3,8)
    block=TransformerBlock(embed_dim=8,num_heads=2,ff_dim=32)
    out=block(x)
    print("输入形状：",x.shape)
    print("输出形状：",out.shape)