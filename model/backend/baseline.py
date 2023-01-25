import torch.nn as nn
from einops.layers.torch import Rearrange


class Aggregator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.aggregator_modules = nn.Sequential(
            Rearrange('b s v -> b (s v)'),
            nn.Linear(4*32, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, n_classes)
            )
        
    def forward(self, x):
        # x: (b s v)
        x = self.aggregator_modules(x)
        return x
    
