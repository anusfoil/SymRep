import os, sys
import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat

import dgl
os.environ['DGLBACKEND'] = 'pytorch'
import dgl.function as fn
import dgl.data
from dgl.dataloading import GraphDataLoader
import pandas as pd

import utils as utils


def get_pc_one_hot(note_array):
    """Get one-hot encoding of pitch class."""
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)),np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_octave_one_hot(note_array):
    """Get one-hot encoding of octave."""
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def feature_extraction_score(note_array):
    '''Extract features from note_array.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    Returns
    -------
    features : np.array
        The features in the shape of (num_nodes, 3). Every node has 3 features, i.e. duration (1), pitch class one hot (12), octave one hot (10).
    '''
    pc_oh = get_pc_one_hot(note_array)
    octave_oh = get_octave_one_hot(note_array)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"] / note_array["ts_beats"]), 1)
    out = np.hstack((duration_feature, pc_oh, octave_oh))
    return out


def edges_from_note_array(note_array):
    '''Turn note_array to list of edges.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    Returns
    -------
    edg_src : np.array
        The edges in the shape of (3, num_edges). every edge is of the form (u, v, t) where u is the source node, v is the destination node and t is the edge type.
    edge_types: dict
        A dictionary with keys 0, 1, 2, 3 and values "onset", "consecutive", "sustain", "silence".
        TODO: do we consider other score information, e.g. voice which is natural for representing score structure? 
    '''

    edge_dict = {0: "onset", 1: "consecutive", 2: "sustain", 3: "silence"}
    edg_src = list()
    edg_dst = list()
    edg_type = list()
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]) & (note_array["pitch"] != x["pitch"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(0)

        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(1)

        for j in np.where((x["onset_div"] < note_array["onset_div"]) & (
                x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edg_type.append(2)

    """connect any note without consecutive edges with their nearest follower"""
    end_times = note_array["onset_div"] + note_array["duration_div"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_div"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_div"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)
                    edg_type.append(3)

    edges = np.array([edg_src, edg_dst, edg_type])
    return edges, edge_dict


def load_graph(path, cfg):

    metadata = pd.read_csv(f"{cfg.graph.save_dir}/metadata.csv")
    res = metadata[metadata['path'] == path]
    if len(res):
        return np.array(dgl.load_graphs(f"{cfg.graph.save_dir}/{res['save_dir'].item()}")[0])

    return None


def save_graph(path, computed_graphs, cfg):

    metadata = pd.read_csv(f"{cfg.graph.save_dir}/metadata.csv")
    N = len(metadata) 
    metadata = metadata.append({"path": path, "save_dir": f"{N}.dgl"}, ignore_index=True)
    metadata.to_csv(f"{cfg.graph.save_dir}/metadata.csv", index=False)

    dgl.save_graphs(f"{cfg.graph.save_dir}/{N}.dgl", computed_graphs)
    return 


def perfmidi_to_graph(path, cfg):
    """Process MIDI events to graphs for training
    - Each note as one node, with features
    - three type of edges: E_onset, E_consec, E_sustain

    In the first run, save all graphs into assets/ and load in future runs.
    Returns:
        graphs: list of dgl.DGLGraph
    """

    res = load_graph(path, cfg)
    if type(res) == np.ndarray: 
        return res[0]

    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    note_events = pd.DataFrame([(n.start, n.end, n.duration, n.pitch, n.velocity) for n in note_events], 
                columns=['start', 'end', 'duration', 'pitch', 'velocity'])
    
    """add sustain pedal value as a feature at the event"""
    pedal_events = perf_data.instruments[0].control_changes 
    pedal_events = pd.concat([
        pd.DataFrame({'time': [0], "value": [0]}),
        pd.DataFrame([(p.time, p.value) for p in pedal_events if p.number == 67], columns=['time', 'value'])])
    note_events["sustain_value"] = note_events.apply(
        lambda row: pedal_events[pedal_events['time'] <= row['start']].iloc[0]['value'], axis=1)

    perfmidi_graphs = []
    seg_length = int(len(note_events) / cfg.experiment.n_segs)
    for i in range(cfg.experiment.n_segs):
        """process on each segment of note events"""
        seg_note_events = note_events.iloc[i*seg_length : i*seg_length+seg_length]
        seg_note_events.index = range(seg_length)
        perfmidi_g = dgl.graph(([], []), num_nodes=seg_length)

        """add node(note) features"""
        perfmidi_g.ndata['feat'] = torch.tensor(np.array(seg_note_events)) # general features are just the 5 terms in note events, plus pedal value at the time

        for j, note in seg_note_events.iterrows():
            """note_event: e.g., Note(start=1.009115, end=1.066406, duration=0.057291, pitch=40, velocity=93)"""
            
            """onset, consecutive, and sustain neighbors, and add edges"""
            onset_nbs = seg_note_events[np.absolute(seg_note_events['start'] - note['start']) < cfg.graph.mid_window]
            consec_nbs = seg_note_events[np.absolute(seg_note_events['start'] - note['end']) < cfg.graph.mid_window]
            sustain_nbs = seg_note_events[(seg_note_events['start'] <= note['start']) & (seg_note_events['end'] > note['end'])]

            perfmidi_g.add_edges(np.array([j] * len(onset_nbs)), np.array(onset_nbs.index))
            perfmidi_g.add_edges(np.array([j] * len(consec_nbs)), np.array(consec_nbs.index))
            perfmidi_g.add_edges(np.array([j] * len(sustain_nbs)), np.array(sustain_nbs.index))

        perfmidi_graphs.append(perfmidi_g)

    save_graph(path, perfmidi_graphs, cfg)
    return perfmidi_graphs # (s, )


def musicxml_to_graph(path, cfg):

    res = load_graph(path, cfg)
    if type(res) == np.ndarray: 
        return res[0]

    import warnings
    warnings.filterwarnings("ignore")  # mute partitura warnings

    try:  # some parsing error....
        score_data = pt.load_musicxml(path)
        note_array = pt.utils.music.ensure_notearray(
            score_data,
            include_pitch_spelling=True, # adds 3 fields: step, alter, octave
            include_key_signature=True, # adds 2 fields: ks_fifths, ks_mode
            include_time_signature=True, # adds 2 fields: ts_beats, ts_beat_type
            # include_metrical_position=True, # adds 3 fields: is_downbeat, rel_onset_div, tot_measure_div
            include_grace_notes=True # adds 2 fields: is_grace, grace_type
        )
    except Exception as e:
        print("Failed on score {} with exception {}".format(os.path.splitext(os.path.basename(path))[0], e))
        return None
    
    # Get edges from note array
    edges, edge_types = edges_from_note_array(note_array)
    # Build graph dict for dgl
    graph_dict = {}
    for type_num, type_name in edge_types.items():
        e = edges[:, edges[2, :] == type_num]
        graph_dict[('note', type_name, 'note')] = torch.tensor(e[0]), torch.tensor(e[1])
    # Built HeteroGraph
    hg = dgl.heterograph(graph_dict)
    # Add node features
    hg.ndata['feat'] = torch.tensor(feature_extraction_score(note_array)).float()
    # Save graph
    save_graph(path, hg, cfg)
    return hg


def get_subgraphs(graph, cfg):
    """ split the graph into segment subgraphs with fixed or flexible number of n_segs
    Return:
        seg_subgraphs (list of dgl.DGLHeteroGraph)
    """

    """choose the number of segments to clip"""
    if cfg.experiment.n_segs: # fixed
        n_segs = cfg.experiment.n_segs
        l = int(graph.num_nodes() / cfg.experiment.n_segs)
    else: # flexible
        n_segs = int(graph.num_nodes() / cfg.graph.subgraph_size) + 1
        l = cfg.graph.subgraph_size

    seg_subgraphs = [dgl.node_subgraph(graph, list(range(i*l, i*l+l))) for i in range(n_segs-1)]

    return seg_subgraphs


def batch_to_graph(batch, cfg, device):
    """Map the batch to input graphs

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        batch_graphs: 
        batch_labels: (b, )
    """
    files, labels = batch
    b = len(batch[0])
    batch_graphs, batch_labels = [], []

    if not os.path.exists(cfg.graph.save_dir): # make saving dir if not exist
        os.makedirs(cfg.graph.save_dir)
        with open(f"{cfg.graph.save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

    for idx, (path, l) in enumerate(zip(files, labels)):
        if cfg.experiment.input_format == "perfmidi":
            graph = perfmidi_to_graph(path, cfg)
        elif cfg.experiment.input_format == "musicxml":
            res = musicxml_to_graph(path, cfg)
            if type(res) == dgl.DGLHeteroGraph:
                graph = res
            else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                continue
        elif cfg.experiment.input_format == "kern":
            graph = kern_to_graph(path, cfg)
        
        batch_graphs.append(get_subgraphs(graph, cfg))
        batch_labels.append(l)
    
    batch_graphs, batch_labels = utils.pad_batch(b, cfg, device, batch_graphs, batch_labels)

    return np.array(batch_graphs), batch_labels
