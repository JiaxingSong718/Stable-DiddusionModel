import torch
from torch import nn
from dataset.dataset import train_dataset
from conv_block import ConvBlock
from config import *
from difussion import forward_difussion
from Time_Position_Embedding import TimePositionEmbedding

class UNet(nn.Module):
    def __init__(self,img_channel,channels=[64,128,256,512,1024],time_embedding_size=256,qsize=16,vsize=16,fsize=32,class_embedding_size=32,num_class=10) -> None:
        super().__init__()

        channels = [img_channel] + channels

        # time转embedding
        self.time_embedding = nn.Sequential(
            TimePositionEmbedding(time_embedding_size),
            nn.Linear(time_embedding_size,time_embedding_size),
            nn.ReLU()
        )

        # 引导词class转embedding
        self.class_embedding = nn.Embedding(num_class,class_embedding_size)

        # 每个encoder conv_block增加一倍通道数
        self.encoder_convs = nn.ModuleList()
        for i in range(len(channels)-1):
            self.encoder_convs.append(ConvBlock(channels[i],channels[i+1],time_embedding_size,qsize,vsize,fsize,class_embedding_size))

        # 每个encoder conv后马上缩小一倍图像储存，最后一个不缩小
        self.maxpools = nn.ModuleList()
        for i in range(len(channels)-2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        
        # 每个decoder前减少一倍通道数，放大一倍图像尺寸，最后一次不放大
        self.deconvs = nn.ModuleList()
        for i in range(len(channels)-2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2],kernel_size=2,stride=2,padding=0))
        
        # 每个decoder conv block
        self.decoder_convs = nn.ModuleList()
        for i in range(len(channels)-2):
            self.decoder_convs.append(ConvBlock(channels[-i-1],channels[-i-2],time_embedding_size,qsize,vsize,fsize,class_embedding_size))

        # 还原通道数，尺寸不变
        self.output = nn.Conv2d(channels[1],img_channel,kernel_size=1,stride=1,padding=0)

    def forward(self,x,t,cls):
        # time -> embedding
        t_embedding = self.time_embedding(t)

        # cls -> embedding
        class_embedding = self.class_embedding(cls)

        # encoder阶段
        residual = []
        for i,conv in enumerate(self.encoder_convs):
            x = conv(x,t_embedding,class_embedding)
            if i != len(self.encoder_convs)-1:
                residual.append(x)
                x = self.maxpools[i](x)

        # decoder阶段
        for i,conv in enumerate(self.deconvs):
            x = conv(x)
            residual_x = residual.pop(-1)
            x = torch.cat((residual_x,x),dim=1)
            x = self.decoder_convs[i](x,t_embedding,class_embedding)

        return self.output(x) #还原通道数
    
if __name__ == '__main__':
    batch_x = torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) #两个图片按照维度0堆叠拼成batch，(2,1,48,48)
    batch_x = batch_x * 2 - 1 #像素值调整到[-1,1]之间，以便与高斯噪声值匹配
    batch_cls=torch.tensor([train_dataset[0][1],train_dataset[1][1]],dtype=torch.long).to(DEVICE)  # 引导ID
    batch_t = torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE) # 生成随机步数

    batch_x_t, batch_noise_t = forward_difussion(batch_x,batch_t)
    print(batch_t)
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())

    unet = UNet(batch_x.size(1)).to(DEVICE)
    batch_pred_noise_t = unet(batch_x_t, batch_t,batch_cls)
    print('batch_pred_noise_t:',batch_pred_noise_t.size())