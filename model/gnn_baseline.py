import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.data
from dgl.nn import GraphConv, SAGEConv
import dgl.nn.pytorch as dglnn
from einops.layers.torch import Rearrange


class GNN(nn.Module):
    def __init__(self, cfg):
        super(GNN, self).__init__()
        self.in_feat_dim = 23
        self.hid_dim = 32
        self.conv1 = dglnn.HeteroGraphConv({
            "onset": SAGEConv(self.in_feat_dim, self.hid_dim, aggregator_type='gcn'),
            "consecutive": SAGEConv(self.in_feat_dim, self.hid_dim, aggregator_type='gcn'),
            "sustain": SAGEConv(self.in_feat_dim, self.hid_dim, aggregator_type='gcn'),
            "silence": SAGEConv(self.in_feat_dim, self.hid_dim, aggregator_type='gcn'),
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            "onset": SAGEConv(self.hid_dim, cfg.experiment.emb_dim, aggregator_type='gcn'),
            "consecutive": SAGEConv(self.hid_dim, cfg.experiment.emb_dim, aggregator_type='gcn'),
            "sustain": SAGEConv(self.hid_dim, cfg.experiment.emb_dim, aggregator_type='gcn'),
            "silence": SAGEConv(self.hid_dim, cfg.experiment.emb_dim, aggregator_type='gcn'),
        }, aggregate='sum')
        

    def forward(self, g):
        h = self.conv1(g, {"note": g.ndata["feat"].float()})
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        g.ndata['h'] = h['note']
        return dgl.mean_nodes(g, "h")
        
