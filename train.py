from config import *
from torch.utils.data import DataLoader
from dataset.dataset import train_dataset
from Unet import UNet
from difussion import forward_difussion
import torch
from torch import nn
import os
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

epochs = 400
batch_size = 400
img_channel = 1

dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=True)

try:
    model = torch.load('./checkpoints/model7.pt')
except:
    model = UNet(img_channel=img_channel).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.L1Loss()  # 绝对值误差均值
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=5,gamma=0.1)

# writer = SummaryWriter()

if __name__ == '__main__':
    model.train()
    n_iter = 0
    for epoch in range(epochs):
        last_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch",ncols=200)  # Create a tqdm progress bar
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
            # writer.add_scalar('Loss/train', last_loss, n_iter)
            n_iter += 1

            # Get the current GPU memory usage
            gpu_mem = torch.cuda.memory_allocated(DEVICE) / (1024 * 1024)  # Convert to MB

            # Update tqdm progress bar description with current loss and GPU memory usage
            progress_bar.set_postfix(loss=last_loss, gpu_mem=f'{gpu_mem:.2f}MB')
        
        # scheduler.step()

        print('epoch:{} loss={}'.format(epoch, last_loss))
        torch.save(model, './checkpoints/model7.pt.tmp')
        os.replace('./checkpoints/model7.pt.tmp', './checkpoints/model7.pt')
