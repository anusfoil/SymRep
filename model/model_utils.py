import torch.nn as nn
from torch.nn import MultiheadAttention
from einops.layers.torch import Rearrange



class AttentionEncodingBlock(nn.Module):
    """The self-attention + linear encoding block in the transformer encoder
    
    All hidden dimensions are in embedding_dim
    """
    def __init__(self, emb_dim, dropout=0.2, num_heads=8, ff_dim=1024, output_weights=False):
        super().__init__()
        self.output_weights = output_weights
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.ModuleList([nn.Linear(self.emb_dim, self.emb_dim) for _ in range(3)])

        self.self_attn = MultiheadAttention(self.emb_dim, num_heads, batch_first=True)
        self.linear_blocks = nn.Sequential(
            nn.Linear(self.emb_dim, ff_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ff_dim, self.emb_dim)
        )
        self.lnorm1 = nn.LayerNorm(self.emb_dim)
        self.lnorm2 = nn.LayerNorm(self.emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (b, n_segs, emb_dim)"""
        # project q k v and expand the heads 
        q, k, v = (proj(x) for proj in self.qkv_proj)

        # attention + add&norm
        x, weights = self.self_attn(q, k ,v, need_weights=True)
        x = self.lnorm1(x + self.dropout(x))

        # Linear + add&norm
        x = self.linear_blocks(x)
        x = self.lnorm2(x + self.dropout(x))

        if self.output_weights:
            return x, weights

        return x
        
