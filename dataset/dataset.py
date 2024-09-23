import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from config import *

# PIL图像转Tensor
pil_to_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # PIL图像统一尺寸
    # transforms.RandomHorizontalFlip(), # PIL图像左右随即反转
    transforms.ToTensor() #PIL图像转为Tensor, (H,W,C) -> (C,H,W),像素值[0,1]
])

#Tensor转PIL图像
tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda t: t*255),  #像素值还原
    transforms.Lambda(lambda t: t.type(torch.uint8)),  #像素值取整
    transforms.ToPILImage() #Tensor图像转为PIL, (C,H,W) -> (H,W,C)

])

#数据集
train_dataset = torchvision.datasets.MNIST(root="./dataset/data/",train=True,download=False,transform=pil_to_tensor)

if __name__ =='__main__':
    img_tensor, label = train_dataset[0]
    print(label)

    #转回PIL图像绘制
    plt.figure(figsize=(5,5))
    pil_img = tensor_to_pil(img_tensor)
    plt.imshow(pil_img)
    plt.show()