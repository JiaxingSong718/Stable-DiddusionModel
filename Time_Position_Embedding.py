import torch
from torch import nn
import math
from config import *

class TimePositionEmbedding(nn.Module):
    def __init__(self,embedding_size) -> None: #举例：向量宽度为8
        super().__init__()
        self.half_embedding_size = embedding_size // 2
        half_embedding = torch.exp(torch.arange(self.half_embedding_size)*(-1*math.log(10000)/(self.half_embedding_size-1)))
        #[exp(0*(-1)*log(10000)/3),exp(1*(-1)*log(10000)/3),exp(2*(-1)*log(10000)/3),exp(3*(-1)*log(10000)/3)]
        self.register_buffer('half_embedding', half_embedding)

    def forward(self,t): #[631,65]
        t = t.view(t.size(0),1) #[[631],[65]]
        half_embedding = self.half_embedding.unsqueeze(0).expand(t.size(0),self.half_embedding_size)
        #[[exp(0*(-1)*log(10000)/3),exp(1*(-1)*log(10000)/3),exp(2*(-1)*log(10000)/3),exp(3*(-1)*log(10000)/3)]]
        half_embedding_t = half_embedding * t
        #[[631,631,631,631],[65,65,65,65]]*[[exp(0*(-1)*log(10000)/3),exp(1*(-1)*log(10000)/3),exp(2*(-1)*log(10000)/3),exp(3*(-1)*log(10000)/3)]]
        embedding_t = torch.cat((half_embedding_t.sin(), half_embedding_t.cos()),dim=-1) #(2,8)
        return embedding_t
    
if __name__ == '__main__':
    time_pos_embedding = TimePositionEmbedding(8).to(DEVICE)
    # t = torch.randint(0,T,(2,)).to(DEVICE)
    t = torch.tensor([482, 249]).to(DEVICE)
    print(t)
    embedding_t = time_pos_embedding(t)
    print(embedding_t)