import torch.nn as nn
import torchvision.models as torchmodels
from einops.layers.torch import Rearrange

def get_convblock(in_channel, out_channel, kernel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(kernel)
    )

class Resnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torchmodels.resnet18(
            layers=cfg.matrix.res_layers,
            num_classes=cfg.experiment.emb_dim) # the resnet blocks are hard-modified with input channel and layers. (Since I can't just pass in this)
        self.blocks = nn.Sequential(
            self.model,
            # Rearrange('b c h w -> b (c h w)') 
            )
        
    def forward(self, x):
        # x: (b s) c h w
        x = self.blocks(x) 
        # x: b (c h w) -- b 32
        assert(x.shape[-1] == self.cfg.experiment.emb_dim)
        return x


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = [get_convblock(i, o, k) for i, o, k in [
            [2, 16, 3],
            [16, 32, 3],
            [32, cfg.experiment.emb_dim, (5, 5)]
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
        assert(x.shape[-1] == self.cfg.experiment.emb_dim)
        return x
    
