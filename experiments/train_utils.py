import os, sys, json
sys.path.extend(["../converters"])
import json
import torch
from converters import utils

def get_colorgroup_by_name(token_name):
    if 'Note' in token_name:
        return 1
    if 'Velocity' in token_name:
        return 2
    if 'Time' in token_name:
        return 3  
    return 4


def attn_visualization(_input, weights, tokenizer, seg_len, k, piece_idx=0, attn_layer=0, subseq_len=50, token_type=None, return_attn=False):
    """record the attention of a specified subsequence and send to vega for plotting arc diagram.
    _input: (batch * segments, seq_len)
     
    args:
     attn_layer: the 
     subseq_len: the total length of sub token sequence  
     token_type: specify a kind of token type to display their attention """
    seg_idx = seg_len * piece_idx
    example = _input[seg_idx, :subseq_len].int().tolist()
    tokens = tokenizer.tokens_to_events(example)
    tokens_names = [f"{t.type}_{t.value}" for t in tokens]
    if token_type:
        tokens_indices = [i for i, t in enumerate(tokens) if (token_type in t.type)]
        tokens_names = [tokens_names[i] for i in tokens_indices]
    
    cweights = weights.detach().clone()
    weights_example = cweights[attn_layer, seg_idx, :subseq_len, :subseq_len].cpu()
    weights_example = weights_example[tokens_indices, :][:, tokens_indices]

    # data processing: 
    w_top = torch.quantile(weights_example, 0.8) # get the top 10 percent of data. 
    # w_mean = weights_example.mean()
    weights_example[weights_example < w_top] = w_top
    weights_example -= weights_example.min()
    weights_example /= weights_example.max()
    
    N = len(tokens_names)
    if return_attn:
        return N, weights_example

    nodes = [{'name': tokens_names[i], 'group': get_colorgroup_by_name(tokens_names[i]), 'index': i} for i in range(N)]
    links = [{'source': i, "target": j, "value": (weights_example[i, j]).item()} for i in range(N) for j in range(N) if weights_example[i, j].item() > 0]

    with open(f"attention_{k}.json", "w") as outfile:
        json.dump({"nodes": nodes, "links": links}, outfile)

    return N


def graph_visualization(cfg, file, N, i, return_adj=False):
    cfg.experiment.symrep = 'graph'
    cfg.experiment.input_format = 'musicxml'
    file = "/homes/hz009/Research/Datasets/asap-dataset/Mozart/Piano_Sonatas/12-1/xml_score.musicxml"
    graph = utils.load_data(file, cfg)[0]
    adj = (graph.adj(etype="consecutive").to_dense()[:N, :N] 
           + graph.adj(etype="onset").to_dense()[:N, :N] 
           + graph.adj(etype="sustain").to_dense()[:N, :N]
           + graph.adj(etype="silence").to_dense()[:N, :N])
    if return_adj:
        return adj
    with open(f"attention_{i}.json", "r") as file:
    # with open("attention_mozart.json", "r") as file:
        data_json = json.load(file)
    links = [{'source': i, "target": j, "value": (adj[i, j]).item()} for i in range(N) for j in range(N) if adj[i, j].item() > 0]
    data_json['links'] = links
    with open(f"graph_connection_{i}.json", "w") as outfile:
        json.dump(data_json, outfile)
    hook()