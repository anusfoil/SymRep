import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.data
from dgl.nn import GraphConv, SAGEConv
import dgl.nn.pytorch as dglnn
from einops.layers.torch import Rearrange

#NOTE: out_dim was cfg.experiment.emb_dim
class GNN(nn.Module):
    def __init__(self, in_dim=23, hid_dim=128, out_dim=64, n_layers=1, etypes=["osnet", "consecutive", "sustain", "silence", "voice"], activation=F.relu, dropout=0.2):
        super(GNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: SAGEConv(self.in_dim, self.hid_dim, aggregator_type='gcn')
            for edge_type in etypes}, aggregate='sum'))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                edge_type: SAGEConv(self.hid_dim, self.hid_dim, aggregator_type='gcn')
            for edge_type in etypes}, aggregate='sum'))
        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: SAGEConv(self.hid_dim, self.out_dim, aggregator_type='gcn')
            for edge_type in etypes}, aggregate='sum'))

    def forward(self, g, x):
        h = x
        for conv in self.layers[:-1]:
            h = conv(g, h)
            h = {k: self.activation(v) for k, v in h.items()}
            h = {k: self.dropout(v) for k, v in h.items()}
        h = self.layers[-1](g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            return dgl.mean_nodes(g, 'h')
        
