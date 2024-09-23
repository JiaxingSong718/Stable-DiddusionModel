import torch
from torch import nn
from config import *
import math

class CrossAttention(nn.Module):
    def __init__(self,channel,qsize,vsize,fsize,class_embedding_size) -> None:
        super().__init__()
        self.w_q = nn.Linear(channel,qsize)
        self.w_k = nn.Linear(class_embedding_size,qsize)
        self.w_v = nn.Linear(class_embedding_size,vsize)
        self.softmax = nn.Softmax(dim=-1)
        self.z_linear = nn.Linear(vsize,channel)
        self.norm1 = nn.LayerNorm(channel)

        # feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(channel,fsize),
            nn.ReLU(),
            nn.Linear(fsize,channel)
        )
        self.norm2 = nn.LayerNorm(channel)

    def forward(self,x,class_embedding): # x:(batch_size,channel,width,height), class_embedding:(batch_size,class_embedding)
        x = x.permute(0,2,3,1) # x:(batch_size,width,height,channel)
        # X = x.view(x.size(0),x.size(2)*x.size(3),x.size(1))

        #像素是query
        Q = self.w_q(x) # x:(batch_size,width,height,qsize)
        Q = Q.view(Q.size(0),Q.size(1)*Q.size(2),Q.size(3)) # x:(batch_size,width*height,qsize)

        # 引导分类是key和value
        K = self.w_k(class_embedding) # K:(batch_size,qsize)
        K = K.view(K.size(0),K.size(1),1) # K:(batch_size,qsize,1)
        V = self.w_v(class_embedding) # K:(batch_size,vsize)
        V = V.view(V.size(0),1,V.size(1)) # K:(batch_size,1,vsize)

        #注意力打分矩阵
        attention = torch.matmul(Q,K)/math.sqrt(Q.size(2)) # x:(batch_size,width*height,1)
        attention = self.softmax(attention)

        #注意力层输出
        Z = torch.matmul(attention,V) # Z:(batch_size,width*height,vsize)
        Z = self.z_linear(Z) # Z:(batch_size,width*height,channel)
        Z = Z.view(x.size(0),x.size(1),x.size(2),x.size(3)) # Z:(batch_size,width,height,channel)

        # 残差&layernorm
        Z = self.norm1(Z+x) # Z:(batch_size,width,height,channel)

        # FeedForward
        output = self.feed_forward(Z) # Z:(batch_size,width,height,channel) -> (batch_size,width,height,fsize) ->(batch_size,width,height,channel)
        # 残差&layernorm
        output = self.norm2(output+Z) # Z:(batch_size,width,height,channel)
        return output.permute(0,3,1,2)
        # return output.view(x.size(0),x.size(1),x.size(2),x.size(3))

    
if __name__ =='__main__':
    batch_size = 2
    channel = 1
    qsize = 256
    class_embedding_size = 32

    cross_attention = CrossAttention(channel=1,qsize=qsize,vsize=128,fsize=512,class_embedding_size=32)

    x = torch.randn((batch_size,channel,IMG_SIZE,IMG_SIZE))
    class_embedding = torch.randn((batch_size,class_embedding_size))

    Z = cross_attention(x,class_embedding)
    print(Z.size())
