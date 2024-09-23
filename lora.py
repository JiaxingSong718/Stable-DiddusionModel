import torch
import math
from torch import nn
from config import *

class LoraLayer(nn.Module):
    def __init__(self,raw_linear,in_features,out_features,r,alpha) -> None:
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.lora_a = nn.Parameter(torch.empty((in_features,r)))
        self.lora_b = nn.Parameter(torch.zeros((r, out_features)))

        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

        self.raw_linear = raw_linear

    def forward(self,x):
        raw_output = self.raw_linear(x)
        lora_output = x @ ((self.lora_a @ self.lora_b) * self.alpha / self.r)
        return raw_output + lora_output
    
def inject_lora(model,name,layer):
    name_cols = name.split('.') # [encoder_convs, 0, cross_attention, w_q]
    # print('===============================================================================')
    # print(name_cols)
    children = name_cols[:-1]  # [encoder_convs, 0, cross_attention]
    # print(children)
    current_layer = model
    for child in children:
        current_layer = getattr(current_layer, child)
    # print(current_layer)

    # print(layer==getattr(urrent_layer,name_cols[-1]))
    lora_layer = LoraLayer(layer,layer.in_features,layer.out_features,LORA_R,LORA_ALPHA)
    setattr(current_layer,name_cols[-1],lora_layer)
    # print('===============================================================================')
    # print(current_layer)

    """
    ===============================================================================
    ['encoder_convs', '0', 'cross_attention', 'w_q']
    ['encoder_convs', '0', 'cross_attention']
    CrossAttention(
    (w_q): Linear(in_features=64, out_features=16, bias=True)
    (w_k): Linear(in_features=32, out_features=16, bias=True)
    (w_v): Linear(in_features=32, out_features=16, bias=True)
    (softmax): Softmax(dim=-1)
    (z_linear): Linear(in_features=16, out_features=64, bias=True)
    (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (feed_forward): Sequential(
        (0): Linear(in_features=64, out_features=32, bias=True)
        (1): ReLU()
        (2): Linear(in_features=32, out_features=64, bias=True)
    )
    (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
    ===============================================================================
    CrossAttention(
    (w_q): LoraLayer(
        (raw_linear): Linear(in_features=64, out_features=16, bias=True)
    )
    (w_k): Linear(in_features=32, out_features=16, bias=True)
    (w_v): Linear(in_features=32, out_features=16, bias=True)
    (softmax): Softmax(dim=-1)
    (z_linear): Linear(in_features=16, out_features=64, bias=True)
    (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (feed_forward): Sequential(
        (0): Linear(in_features=64, out_features=32, bias=True)
        (1): ReLU()
        (2): Linear(in_features=32, out_features=64, bias=True)
    )
    (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
    """