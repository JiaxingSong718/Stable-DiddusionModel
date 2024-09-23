import torch

IMG_SIZE = 48  #图像尺寸
T = 1500
LORA_R = 8
LORA_ALPHA = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"