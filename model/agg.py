import torch.nn as nn
from torch.nn import MultiheadAttention
from einops.layers.torch import Rearrange, Reduce
from einops import repeat
import model_utils as model_utils


class BaselineAggregator(nn.Module):
    """aggregating the segment-level embedding for final prediction"""
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
    

class AttentionAggregator(nn.Module):
    def __init__(self, cfg, n_classes, ff_dim=512):
        super().__init__()

        self.n_classes = n_classes
        self.emb_dim =  cfg.experiment.emb_dim
        self.attn_block = model_utils.AttentionEncodingBlock(cfg.experiment.emb_dim)
        self.pred_proj = nn.Sequential(
            Reduce('b s v -> b v', "mean"),
            nn.Linear(self.emb_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, n_classes)
            )

    def forward(self, x):
        """x: (b, n_segs, emb_dim)"""
        x = self.attn_block(x)
        x = self.pred_proj(x)
        return x
        
