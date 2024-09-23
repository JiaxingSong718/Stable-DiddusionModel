import torch
from config import *
from dataset.dataset import train_dataset, tensor_to_pil
import matplotlib.pyplot as plt

#前向difussion计算参数
betas = torch.linspace(0.0001,0.02,T)
alphas = 1 - betas

alphas_cumprod = torch.cumprod(alphas,dim=-1) #alphas_t累乘(T,)， [a1,a2,a3,...,aN] -> [a1,a1*a2,a1*a2*a3,...,a1*a2*a3*...*aN]
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) #alphas_t-1累乘 [a1,a1*a2,a1*a2*a3,...,a1*a2*a3*...*aN] -> [1,a1,a1*a2,a1*a2*a3,...,a1*a2*a3*...*aN-1]
# print(alphas_cumprod_prev)
variance = (1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod) #denose用的方差(T,)

def forward_difussion(batch_x,batch_t): #batch_x:(batch_size,channel,width,height),batch_t:(batch_size,)
    batch_noise_t = torch.randn_like(batch_x) #每张图片生成第t步的高斯噪声 (batch_size, channel, width, height)
    batch_alphas_cumprod = alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0),1,1,1) #后面三个维度的1会广播到(batch_size, channel, width, height)
    batch_x_t = torch.sqrt(batch_alphas_cumprod) * batch_x + torch.sqrt(1-batch_alphas_cumprod) * batch_noise_t #基于公式直接生成第t步加噪后的图片
    return batch_x_t, batch_noise_t

if __name__ == '__main__':
    batch_x = torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) #两个图片按照维度0堆叠拼成batch，(2,1,48,48)
    batch_x = batch_x * 2 - 1 #像素值调整到[-1,1]之间，以便与高斯噪声值匹配
    batch_t = torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)
    batch_x_t, batch_noise_t = forward_difussion(batch_x,batch_t)
    print(batch_t)
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())
    plt.figure(figsize=(5,10))
    plt.subplot(3,2,1)
    plt.imshow(tensor_to_pil((batch_x[0]+1)/2))
    plt.subplot(3,2,2)
    plt.imshow(tensor_to_pil((batch_x[1]+1)/2))
    plt.subplot(3,2,3)
    plt.imshow(tensor_to_pil(batch_noise_t[0]))
    plt.subplot(3,2,4)
    plt.imshow(tensor_to_pil(batch_noise_t[1]))
    plt.subplot(3,2,5)
    plt.imshow(tensor_to_pil(batch_x_t[0]))
    plt.subplot(3,2,6)
    plt.imshow(tensor_to_pil(batch_x_t[1]))
    plt.show()
    
