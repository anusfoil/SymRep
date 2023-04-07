import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.data
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv
import dgl.nn.pytorch as dglnn
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from einops import reduce

class GNN_GAT(nn.Module):
    def __init__(self, cfg, in_dim=23,
                 activation=F.relu, dropout=0.2):
        super().__init__()

        self.cfg = cfg
        self.hid_dim = cfg.graph.hid_dim
        self.out_dim = cfg.experiment.emb_dim
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.num_head = 2
        self.in_dim = in_dim
        self.etypes = cfg.graph.basic_edges
        if (not cfg.graph.bi_dir):
            self.etypes = [et for et in cfg.graph.basic_edges if ("rev" not in et)]
        if cfg.experiment.feat_level == 1:
            self.etypes.append('voice')
            if cfg.graph.bi_dir:
                self.etypes.append('voice_rev')

        n_layers = cfg.graph.n_layers

        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: GATConv(self.in_dim, self.hid_dim, num_heads=self.num_head)
            for edge_type in self.etypes}, aggregate='mean'))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                edge_type: GATConv(self.hid_dim, self.hid_dim, num_heads=self.num_head)
            for edge_type in self.etypes}, aggregate='mean'))
        self.layers.append(dglnn.HeteroGraphConv({
            edge_type: GATConv(self.hid_dim, self.out_dim, num_heads=self.num_head)
            for edge_type in self.etypes}, aggregate='mean'))
        # self.attn_agg = Reduce("b 2 2 d -> b d", reduction='mean')

    def forward(self, g):

        h = self.feature_by_level(g)

        for conv in self.layers[:-1]:
            h = conv(g, h)
            h = {k: self.activation(v) for k, v in h.items()}
            h = {k: F.normalize(v) for k, v in h.items()}
            h = {k: self.dropout(v) for k, v in h.items()}
            h = {k: reduce(v, f"b {self.num_head} d -> b d", 'mean') for k, v in h.items()}

        h = self.layers[-1](g, h)
        h = {k: reduce(v, f"b {self.num_head} d -> b d", 'mean') for k, v in h.items()}

        g.ndata['h'] = h['note']
        # Calculate graph representation by average readout. TODO: play with this
        out = dgl.mean_nodes(g, 'h') 

        return out
    

    def feature_by_level(self, g):
        """select the subset of features to use based on the level control"""
    
        if self.cfg.experiment.feat_level == 1:
            node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_1']], dim=1)
        else:
            # node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_-1'].unsqueeze(1)], dim=1)
            node_features = g.ndata['feat_0']

        return {'note': node_features}
        
def get_base_conv(conv_type, in_dim, out_dim, cfg, num_head=8):
    """get the base convolution layer depending on the given conv_type """
    
    if conv_type == "SAGEConv":
        return SAGEConv(in_dim, out_dim, aggregator_type=cfg.graph.sage_agg)
    if conv_type == "GATConv":
        return GATConv(in_dim, out_dim, num_heads=num_head)
    if conv_type == "GINConv":
        lin = nn.Linear(in_dim, out_dim)
        return GINConv(lin, "max")


def edge_agg(edge_agg_layer, stacked_output):
    outputs = []
    for i in range(stacked_output.shape[1]):
        outputs.append(edge_agg_layer[i](stacked_output[:, i, :]))
    return reduce(torch.stack(outputs), "e n d -> n d", "mean")


class GNN(nn.Module):
    def __init__(self, cfg, in_dim=23, activation=F.relu):
        super().__init__()

        self.cfg = cfg
        self.hid_dim = cfg.graph.hid_dim
        self.out_dim = cfg.experiment.emb_dim
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(cfg.graph.dropout)

        self.in_dim = in_dim
        self.etypes=cfg.graph.basic_edges
        if (not cfg.graph.bi_dir):
            self.etypes = [et for et in cfg.graph.basic_edges if ("rev" not in et)]
        if cfg.experiment.feat_level == 1:
            self.etypes.append('voice')
            if cfg.graph.bi_dir:
                self.etypes.append('voice_rev')

        n_layers = cfg.graph.n_layers

        # an linear layer that aggregates the edge's outputs
        # self.edg_agg_layer = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.hid_dim, self.hid_dim),
        #         nn.BatchNorm1d(self.hid_dim),
        #         nn.Dropout(dropout)
        #     )
        #     for _ in range(len(self.etypes))
        # ])
        # self.edg_agg_layer_end = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.out_dim, self.out_dim),
        #         nn.BatchNorm1d(self.out_dim),
        #         nn.Dropout(dropout)
        #     )
        #     for _ in range(len(self.etypes))
        # ])
        # self.edg_agg_end = nn.Sequential(
        #     Rearrange("n e d -> n (e d)"),
        #     nn.Linear(len(self.etypes) * self.out_dim, self.out_dim) 
        # )
        if cfg.graph.homo:
            self.layers.append(get_base_conv(cfg.graph.conv_type, self.in_dim, self.hid_dim, cfg))
            for _ in range(n_layers - 1):
                self.layers.append(get_base_conv(cfg.graph.conv_type, self.hid_dim, self.hid_dim, cfg))
            self.layers.append(get_base_conv(cfg.graph.conv_type, self.hid_dim, self.out_dim, cfg)) 
        else:
            self.layers.append(dglnn.HeteroGraphConv({
                edge_type: get_base_conv(cfg.graph.conv_type, self.in_dim, self.hid_dim, cfg)
                for edge_type in self.etypes}, aggregate=cfg.graph.edge_agg)) # do operation with aggregation 
            for _ in range(n_layers - 1):
                self.layers.append(dglnn.HeteroGraphConv({
                    edge_type: get_base_conv(cfg.graph.conv_type, self.hid_dim, self.hid_dim, cfg)
                for edge_type in self.etypes}, aggregate=cfg.graph.edge_agg))
            self.layers.append(dglnn.HeteroGraphConv({
                edge_type: get_base_conv(cfg.graph.conv_type, self.hid_dim, self.out_dim, cfg)
                for edge_type in self.etypes}, aggregate=cfg.graph.edge_agg))


    def forward(self, g):

        h = self.feature_by_level(g)

        for idx, conv in enumerate(self.layers[:-1]):
            # h_ = h
            h = conv(g, h)
            if self.cfg.graph.homo:
                h = self.activation(h)
                h = F.normalize(h) 
                h = self.dropout(h)
            else:
                # h = {k: edge_agg(self.edg_agg_layer, v) for k, v in h.items()}
                # if idx != 0:
                #     h = {k: v + h_[k] for k, v in h.items()} # add residual connection
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: F.normalize(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
                
                if self.cfg.graph.conv_type == "GATConv": 
                    h = {k: reduce(v, f"b {self.num_head} d -> b d", 'mean') for k, v in h.items()}

        h = self.layers[-1](g, h)
        # h = {k: edge_agg(self.edg_agg_layer_end, v) for k, v in h.items()}

        g.ndata['h'] = (h if self.cfg.graph.homo else h['note'])
        # Calculate graph representation by average readout.
        return dgl.mean_nodes(g, 'h')
    

    def feature_by_level(self, g):
        """select the subset of features to use based on the level control"""
    
        if self.cfg.experiment.feat_level == 1:
            node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_1']], dim=1)
        else:
            # node_features = torch.cat([g.ndata['feat_0'], g.ndata['feat_-1'].unsqueeze(1)], dim=1)
            node_features = g.ndata['feat_0']

        return node_features if self.cfg.graph.homo else {'note': node_features}