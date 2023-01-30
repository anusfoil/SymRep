import torch.nn as nn
from einops.layers.torch import Rearrange

def get_convblock(in_channel, out_channel, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(kernel)
    )

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [get_convblock(i, o, k) for i, o, k in [
            [2, 16, 3],
            [16, 32, 3],
            [32, 32, (3, 5)]
        ]]
        self.cnn_modules = nn.Sequential(
            *self.blocks,
            nn.AvgPool2d(1, 17),
            Rearrange('b c h w -> b (c h w)')
            )
        
    def forward(self, x):
        # x: (b s) c h w
        x = self.cnn_modules(x)
        # x: b (c h w) -- b 32
        return x
    
