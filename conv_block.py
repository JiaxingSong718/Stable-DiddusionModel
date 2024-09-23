from torch import nn
from Cross_Attention import CrossAttention

class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,time_embedding_size,qsize,vsize,fsize,class_embedding_size) -> None:
        super().__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1), # 更改通道数，不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.time_embedding_linear = nn.Linear(time_embedding_size,out_channel) # Time时刻embedding转成channel宽，加到每个像素点上
        self.relu = nn.ReLU()

        self.seq2 = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1), # 不改通道数，不改大小
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # 像素做Query, 计算对分类ID的注意力，实现分类信息融入图像，不改变图像形状和通道数
        self.cross_attention = CrossAttention(channel = out_channel,qsize=qsize,vsize=vsize,fsize=fsize,class_embedding_size=class_embedding_size)

    def forward(self,x,time_embedding,class_embedding): #time_embedding:(batch_size,time_embedding_size)
        x = self.seq1(x)
        time_embedding = self.relu(self.time_embedding_linear(time_embedding)).view(x.size(0),x.size(1),1,1)
        output = self.seq2(x+time_embedding) 
        return self.cross_attention(output,class_embedding)