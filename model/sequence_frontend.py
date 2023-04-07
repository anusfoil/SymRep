import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import reduce
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
    def __init__(self, cfg, n_layers=5):
        super().__init__()

        self.hid_dim = cfg.sequence.hid_dim
        self.num_heads = cfg.sequence.n_heads
        self.cfg = cfg
        if cfg.sequence.mid_encoding == "CPWord":
            self.emb = nn.ModuleList([
                nn.Embedding(n_tokens, self.hid_dim, padding_idx=0) for n_tokens in cfg.sequence.vocab_size
            ])
        else:
            n_tokens = (sum(cfg.sequence.vocab_size) * cfg.sequence.BPE) if cfg.sequence.BPE else 400
            self.emb = nn.Embedding(n_tokens, self.hid_dim, padding_idx=0) 
        self.pe = PositionalEncoding(self.hid_dim, max_len=cfg.sequence.max_seq_len)
        self.attn_layers = nn.ModuleList(
            [utils.AttentionEncodingBlock(self.hid_dim, num_heads=self.num_heads, output_weights=cfg.sequence.output_weights) for _ in range(n_layers)]
            )

        self.blocks = nn.Sequential(
            nn.Linear(self.hid_dim, cfg.experiment.emb_dim),
            Reduce("b l e -> b e", "mean") # play with this!
        )
        
    def forward(self, x):
        """x: (b, l) or (b l 6)"""
        if self.cfg.sequence.mid_encoding == "CPWord":
            x = [emb(x[:, :, i].long()) for i, emb in enumerate(self.emb)]
            x = reduce(torch.stack(x), "6 b l e -> b l e", "mean")
        else:
            x = self.emb(x.long())
        x = self.pe(x)
        attn_weights = []
        for layer in self.attn_layers:
            if not self.cfg.sequence.output_weights:
                x = layer(x)
                continue
            x, weights = layer(x)
            attn_weights.append(weights)
        
        x = self.blocks(x)
        
        if self.cfg.sequence.output_weights:
            attn_weights = torch.stack(attn_weights)
            return x, attn_weights

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len) / d_model))
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