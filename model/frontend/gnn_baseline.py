import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.data
from dgl.nn import GraphConv, SAGEConv
from einops.layers.torch import Rearrange


class GNN(nn.Module):
    def __init__(self, cfg):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(6, 32, aggregator_type='gcn')
        self.conv2 = SAGEConv(32, cfg.experiment.emb_dim, aggregator_type='gcn')
        

    def forward(self, g):
        h = self.conv1(g, g.ndata["general_note_feats"].float())
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, "h")
        
