import torch
import torch.nn as nn
from modules import TokenEmbedding

class Prefix_TokenEmbedding(nn.Module):
    def __init__(self, tokenEmbedding, scale=0.02, target_id1=9, target_id2=10, target_id3=4):
        super().__init__()
        self.scale = scale
        self.target_id1 = target_id1
        self.target_id2 = target_id2
        self.target_id3 = target_id3
        self.prefix_1 = nn.Parameter(torch.randn(1,tokenEmbedding.embedding_dim) * scale)
        self.prefix_2 = nn.Parameter(torch.randn(1,tokenEmbedding.embedding_dim) * scale)
        self.prefix_3 = nn.Parameter(torch.randn(1,tokenEmbedding.embedding_dim) * scale)

        self.original_forward = tokenEmbedding.forward
        for param in tokenEmbedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        mask1 = (x == self.target_id1)
        mask2 = (x == self.target_id2)
        mask3 = (x == self.target_id3) #4
        mask4 = (x == self.target_id3+1) #5
        mask5 = (x == self.target_id3+2) #6
        mask6 = (x == self.target_id3+3) #7
        mask7 = (x == self.target_id3+4) #8

        x = self.original_forward(x)
        
        x[mask1] = self.prefix_1
        x[mask2] = self.prefix_2

        x[mask3] = self.prefix_3
        x[mask4] = self.prefix_3
        x[mask5] = self.prefix_3
        x[mask6] = self.prefix_3
        x[mask7] = self.prefix_3
        return x
    
def apply_prefix(model,scale=0.02,target_id1=9, target_id2=10, target_id3=4):
    for name, module in model.named_modules():
        if isinstance(module,TokenEmbedding) and any(key in name for key in ['tokenEmbedding']):
            prefix = Prefix_TokenEmbedding(module,scale=scale,target_id1=target_id1,target_id2=target_id2,target_id3=target_id3).to(model.device)

            setattr(module, "prefix", prefix)
            module.forward = prefix.forward