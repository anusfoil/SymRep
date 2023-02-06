import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class RNN(nn.Module):
    def __init__(self, cfg):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(1600, 256)
        self.rnn = nn.RNN(256, 256, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True)
        self.blocks = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, int(cfg.experiment.emb_dim / 4)),
            Rearrange('(dl) b hout-> b (dl hout)')
        )
        

    def forward(self, x):

        x = self.emb(x.long())
        _, hidden = self.rnn(x)
        x = self.blocks(hidden)
        
        return x
        
