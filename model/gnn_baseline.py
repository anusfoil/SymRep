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
        self.conv1 = SAGEConv(6, 16, aggregator_type='gcn')
        self.conv2 = SAGEConv(16, 32, aggregator_type='gcn')
        self.conv3 = SAGEConv(32, 64, aggregator_type='gcn')
        self.conv4 = SAGEConv(64, cfg.experiment.emb_dim, aggregator_type='gcn')
        

    def forward(self, g):
        h = F.relu(self.conv1(g, g.ndata["feat"].float()))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        g.ndata['h'] = h
        return dgl.mean_nodes(g, "h")
        
