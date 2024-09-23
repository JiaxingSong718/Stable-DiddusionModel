from config import *
from Unet import UNet
import torch
from torch import nn
from difussion import *
import matplotlib.pyplot as plt
from dataset.dataset import tensor_to_pil
from lora import inject_lora, LoraLayer

def backward_senoise(model, batch_x_t, batch_class):
    step = [batch_x_t,]

    global alphas,alphas_cumprod,variance

    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    batch_class = batch_class.to(DEVICE)

    with torch.no_grad():
        for t in range(T-1,-1,-1):
            batch_t = torch.full((batch_x_t.size(0),),t) #[999,999,999,999,999,999,999,999,999,999]
            batch_t = batch_t.to(DEVICE)
            # 预测x_t时刻的噪音
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_class)
            # 生成t-1时刻的图像
            shape = (batch_x_t.size(0),batch_x_t.size(1),1,1)
            batch_mean_t = 1 / torch.sqrt(alphas[batch_t].view(*shape)) * \
                (
                    batch_x_t -
                    ((1-alphas[batch_t].view(*shape)) / torch.sqrt(1-alphas_cumprod[batch_t].view(*shape))) * batch_predict_noise_t
                )
            if t != 0:
                batch_x_t = batch_mean_t + torch.randn_like(batch_x_t) * torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t = batch_mean_t 
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach() #将像素点调整到-1，1之间, detach()将数据从GPU拉到CPU
            step.append(batch_x_t)

    return step

if __name__ == '__main__': 
    #加载模型
    model = torch.load('./checkpoints/model5.pt')

    # LORA
    USE_LORA = False
    if USE_LORA == True:
        # 向nn.Linear层注入Lora
        for name,layer in model.named_modules():
            name_cols = name.split('.')
            # 过滤出cross attention使用的linear权重
            filter_names = ['w_q','w_k','w_v']
            if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
                inject_lora(model,name,layer)

        # lora权重加载
        try:
            restore_lora_state = torch.load('./checkpoints/lora5.pt')
            model.load_state_dict(restore_lora_state,strict=False)
        except:
            pass

        model = model.to(DEVICE)

        # lora权重合并到主模型
        for name,layer in model.named_modules():
            name_cols = name.split('.')

            if isinstance(layer,LoraLayer):
                children = name_cols[:-1]  # [encoder_convs, 0, cross_attention]
                current_layer = model
                for child in children:
                    current_layer = getattr(current_layer, child)
                lora_weight = (layer.lora_a @ layer.lora_b)*layer.alpha/layer.r
                before_weight = layer.raw_linear.weight.clone()
                layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(lora_weight.T)).to(DEVICE)
                setattr(current_layer,name_cols[-1],layer.raw_linear)

    # 打印模型结构
    # print(model)
    # 生成噪音图
    batch_size = 10
    img_channel = 1
    batch_x_t = torch.randn(size=(batch_size,img_channel,IMG_SIZE,IMG_SIZE))
    batch_class = torch.arange(start=0, end=10, dtype=torch.long)
    # 逐步得到去噪原图
    steps = backward_senoise(model, batch_x_t, batch_class)
    # 绘制数量
    num_imgs = 10
    print(len(steps[-1]))

    #绘制还原过程
    # plt.figure(figsize=(20,20))
    # for i in range(batch_size):
    #     for j in range(0,num_imgs):
    #         idx = int(T/num_imgs)*(j+1)
    #         #像素值还原到0，1
    #         final_img = (steps[idx][i].to('cpu')+1)/2
    #         final_img = tensor_to_pil(final_img)
    #         plt.subplot(batch_size,num_imgs,i*num_imgs+j+1)
    #         plt.imshow(final_img)
    # plt.show()

    plt.figure(figsize=(10,10))
    for i in range(10):
        #像素值还原到0，1
        final_img = (steps[-1][i].to('cpu')+1)/2
        final_img = tensor_to_pil(final_img)
        plt.subplot(int(batch_size/5),int(batch_size/2),i+1)
        plt.title(i)
        plt.axis('off')
        plt.imshow(final_img)
    plt.show()
