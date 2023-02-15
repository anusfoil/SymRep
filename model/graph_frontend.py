import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.data
from dgl.nn import GraphConv, SAGEConv, GATConv
import dgl.nn.pytorch as dglnn
from einops.layers.torch import Rearrange, Reduce

#NOTE: out_dim was cfg.experiment.emb_dim
class GNN(nn.Module):
    def __init__(self, cfg, in_dim=23, hid_dim=128, n_layers=1,
                 activation=F.relu, dropout=0.2):
        super(GNN, self).__init__()

        self.cfg = cfg
        self.hid_dim = hid_dim
        self.out_dim = cfg.experiment.emb_dim
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.in_dim = in_dim
        self.etypes = cfg.graph.basic_edges
        if cfg.experiment.feat_level == 1:
            self.etypes.append('voice')

        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: GATConv(self.in_dim, self.hid_dim, num_heads=2)
            for edge_type in self.etypes}, aggregate='sum'))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                edge_type: GATConv(self.hid_dim, self.hid_dim, num_heads=2)
            for edge_type in self.etypes}, aggregate='sum'))
        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: GATConv(self.hid_dim, self.out_dim, num_heads=2)
            for edge_type in self.etypes}, aggregate='sum'))
        self.attn_agg = Reduce("b 2 2 d -> b d", reduction='mean')

    def forward(self, g):

        h = self.feature_by_level(g)

        for conv in self.layers[:-1]:
            h = conv(g, h)
            h = {k: self.activation(v) for k, v in h.items()}
            h = {k: F.normalize(v) for k, v in h.items()}
            h = {k: self.dropout(v) for k, v in h.items()}

        h = self.layers[-1](g, h)

        g.ndata['h'] = h['note']
        # Calculate graph representation by average readout.
        out = dgl.mean_nodes(g, 'h')

        return self.attn_agg(out)
    

    def feature_by_level(self, g):
        """select the subset of features to use based on the level control"""
    
        if self.cfg.experiment.feat_level == 1:
            node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_1']], dim=1)
        else:
            # node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_-1'].unsqueeze(1)], dim=1)
            node_features = g.ndata['feat_0']

        return {'note': node_features}
        

