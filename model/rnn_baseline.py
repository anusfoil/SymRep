import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
import model_utils as utils


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
        

class AttentionEncoder(nn.Module):
    def __init__(self, cfg, n_layers=1):
        super().__init__()

        self.hid_dim = 64
        self.emb = nn.Embedding(500, self.hid_dim, padding_idx=0) #vocabulary shouldn't be larger than 500
        self.pe = PositionalEncoding(self.hid_dim)
        self.attn_layers = nn.Sequential(
            *[utils.AttentionEncodingBlock(self.hid_dim) for _ in range(n_layers)]
            )

        self.blocks = nn.Sequential(
            nn.Linear(self.hid_dim, cfg.experiment.emb_dim),
            Reduce("b l e -> b e", "mean")
        )
        
    def forward(self, x):
        """x: (b, l)"""
        x = self.emb(x.long())
        x = self.pe(x)
        x = self.attn_layers(x)
        x = self.blocks(x)
        
        return x
        


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=10000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x