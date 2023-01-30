import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class RNN(nn.Module):
    def __init__(self, *args):
        super(RNN, self).__init__(*args)
        self.emb = nn.Embedding(1600, 256)
        self.rnn = nn.RNN(256, 256, 
            num_layers=3, 
            batch_first=True,
            bidirectional=True)
        self.blocks = nn.Sequential(
            nn.Linear(256, 32),
            Rearrange('(d l) b hout-> b (d l hout)')
        )
        

    def forward(self, x):
        b, l = x.shape

        x = self.emb(x)
        _, hidden = self.rnn(x)
        x = self.blocks(hidden)
        
        return x
        
