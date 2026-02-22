import torch
import torch.nn as nn
from model.prefix_model import apply_prefix

class LoRA(nn.Module):
    def __init__(self, linear, rank=8, scale=0.02):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(linear.in_features, rank) * scale)
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))
        for param in linear.parameters():
            param.requires_grad = False
        self.original_forward = linear.forward

    def forward(self, x):
        return self.original_forward(x) + (x @ self.lora_A @ self.lora_B)

def apply_lora(model,rank=8,scale=0.02,prefix=True,target_id1=9,target_id2=10,target_id3=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(key in name for key in ['w_dq', 'w_dkv', 'w_o']):
            lora = LoRA(module, rank=rank, scale=scale).to(module.weight.device)

            setattr(module, "lora", lora)
            module.forward = lora.forward
    if prefix==True:
        apply_prefix(model=model,scale=scale,target_id1=target_id1,target_id2=target_id2,target_id3=target_id3)