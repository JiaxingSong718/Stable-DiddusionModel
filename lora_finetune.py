import torch
from torch import nn
from config import *
from torch.utils.data import DataLoader
from dataset.dataset import train_dataset
from lora import inject_lora
from tqdm import tqdm
from difussion import forward_difussion
import os
# 11.4G
EPOCH = 200
BATCH_SIZE = 400

if __name__ == '__main__':
    # 预训练模型
    model = torch.load('./checkpoints/model5.pt')

    # 向nn.Linear层注入Lora
    for name,layer in model.named_modules():
        # print(name) # encoder_convs.0.cross_attention.w_q
        # print('=======================================================')
        # print(layer) # Linear(in_features=64, out_features=16, bias=True)
        # print('=======================================================')
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

    # 冻结非lora参数
    for name,param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a','lora_b']:
            param.requires_grad = False
        else:
            param.requires_grad = True

    dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=4,persistent_workers=True,shuffle=True)

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad == True,model.parameters()),lr=0.001) #优化器只更新lora参数
    loss_fn = nn.L1Loss()

    print(model)

    model.train()
    n_iter = 0
    for epoch in range(EPOCH):
        last_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch",ncols=200)  # Create a tqdm progress bar
        for batch_x, batch_cls in progress_bar:
            batch_cls = batch_cls.to(DEVICE)
            # 像素值调整到[-1,1]之间，以便与高斯噪声值匹配
            batch_x = batch_x.to(DEVICE) * 2 - 1
            batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)  # 每张图片生成随机步数
            # 生成t时刻的加噪图片和对应噪声
            batch_x_t, batch_noise_t = forward_difussion(batch_x, batch_t)
            # 模型预测t时刻噪声
            batch_predict_t = model(batch_x_t, batch_t, batch_cls)
            # 求损失
            loss = loss_fn(batch_predict_t, batch_noise_t)
            # 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            n_iter += 1

            # Get the current GPU memory usage
            gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)  # Convert to MB

            # Update tqdm progress bar description with current loss and GPU memory usage
            progress_bar.set_postfix(loss=last_loss, gpu_mem=f'{gpu_mem:.2f}MB')

        print('epoch:{} loss={}'.format(epoch, last_loss))

        # 保存训练好的Lora权重
        lora_state = {}
        for name, param in model.named_parameters():
            name_cols = name.split('.')
            filter_names = ['lora_a','lora_b']
            if any(n==name_cols[-1] for n in filter_names):
                lora_state[name] = param
        torch.save(lora_state, './checkpoints/lora5.pt.tmp')
        os.replace('./checkpoints/lora5.pt.tmp', './checkpoints/lora5.pt')