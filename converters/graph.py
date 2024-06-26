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
import wandb

import utils as utils


def feature_extraction_score(note_array, score=None, include_meta=False):
    '''Extract features from note_array.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    score : partitura score object (optional)
        The partitura score object. If provided, the meta features can be extracted.
    include_meta : bool
        Whether to include meta features. ()

    Returns
    -------
    features : np.array
        level 0 features: duration (1), pitch class one hot (12), octave one hot (10).
        level 1 features: 61 dim
    '''
    # Solution for the problem of note tied in make_note_features() but it takes longer to parse each score.
    # if include_meta and isinstance(score, pt.score.Score):
    #     score = pt.score.merge_parts(score.parts)
    #     note_array = score.note_array()
    pc_oh = utils.get_pc_one_hot(note_array)
    octave_oh = utils.get_octave_one_hot(note_array)
    duration_feature = np.expand_dims(1 - np.tanh(note_array["duration_beat"] / note_array["ts_beats"]), 1)

    feat_0, feat_1 = np.hstack((duration_feature, pc_oh, octave_oh)), None
    if include_meta and score is not None:
        # All feature functions in partitura
        # ['articulation_direction_feature', 'articulation_feature', 'duration_feature', 'fermata_feature',
        # 'grace_feature', 'loudness_direction_feature', 'metrical_feature', 'metrical_strength_feature',
        # 'onset_feature', 'ornament_feature', 'polynomial_pitch_feature', 'relative_score_position_feature',
        # 'slur_feature', 'staff_feature', 'tempo_direction_feature', 'time_signature_feature',
        # 'vertical_neighbor_feature']
        # meta_features, _ = pt.musicanalysis.make_note_features(
        #     score,
        #     ["articulation_direction_feature", "articulation_feature", "fermata_feature", 'loudness_direction_feature',
        #      'metrical_feature', 'metrical_strength_feature', 'ornament_feature', 'slur_feature', 'staff_feature',
        #      'tempo_direction_feature', 'time_signature_feature'])
        # NOTE: If that create different features length for each score then use this:
        articulation, _ = pt.musicanalysis.note_features.articulation_feature(note_array, score, include_empty_features=True)
        art_direction, _ = pt.musicanalysis.note_features.articulation_direction_feature(note_array, score, include_empty_features=True)
        loudness, _ = pt.musicanalysis.note_features.loudness_direction_feature(note_array, score, include_empty_features=True)
        direction, _ = pt.musicanalysis.note_features.tempo_direction_feature(note_array, score, include_empty_features=True)
        staff_feature, _ = pt.musicanalysis.note_features.staff_feature(note_array, score, include_empty_features=True)
        meta_features = np.hstack((articulation, art_direction, loudness, direction, staff_feature))

        feat_1 = meta_features
    return feat_0, feat_1


def edges_from_note_array(note_array, measures=None):
    '''Turn note_array to list of edges.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    measures : numpy array (optional)
        The measures array. If provided, it will create voice edges, for consecutive notes in the same voices that belong in the same measure.

    Returns
    -------
    edg_src : np.array
        The edges in the shape of (3, num_edges). every edge is of the form (u, v, t) where u is the source node, v is the destination node and t is the edge type.
    edge_types: dict
        A dictionary with keys 0, 1, 2, 3 and values "onset", "consecutive", "sustain", "silence", "voice.
    '''

    edge_dict = {0: "onset", 1: "consecutive", 2: "sustain", 3: "silence", 4: "voice"}
    edg_src = list()
    edg_dst = list()
    edg_type = list()
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]) & (note_array["id"] != x["id"]))[0]:
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
        
        # adding self-loop
        edg_src.extend([i] * 3)
        edg_dst.extend([i] * 3)
        edg_type.extend([0, 1, 2])

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

    if measures is not None:
        for m_num in range(len(measures)):
            start = measures[m_num, 0]
            end = measures[m_num, 1]
            note_array_seg = np.where((note_array["onset_div"] >= start) & (note_array["onset_div"] < end))[0]
            for idx, i in enumerate(note_array_seg):
                for j in note_array_seg[idx:]:
                    if note_array[i]["voice"] == note_array[j]["voice"] and i != j:
                        edg_src.append(i)
                        edg_dst.append(j)
                        edg_type.append(4)

    edges = np.array([edg_src, edg_dst, edg_type])
    return edges, edge_dict


def load_graph(path, cfg):

    metadata = pd.read_csv(f"{cfg.graph.save_dir}/metadata.csv")
    res = metadata[metadata['path'] == path]
    if len(res):
        return np.array(dgl.load_graphs(f"{cfg.graph.save_dir}/{res['save_dir'].iloc[0]}")[0])

    return None


def save_graph(path, computed_graphs, cfg):

    if not os.path.exists(cfg.graph.save_dir): # make saving dir if not exist
        os.makedirs(cfg.graph.save_dir)
        with open(f"{cfg.graph.save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

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
        Graph: dgl.DGLGraph
    """

    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    note_events = pd.DataFrame([(n.start, n.end, n.duration, n.pitch, n.velocity) for n in note_events], 
                columns=['start', 'end', 'duration', 'pitch', 'velocity'])
    
    """add sustain pedal value as a feature at the event"""
    pedal_events = perf_data.instruments[0].control_changes 
    pedal_events = pd.concat([
        pd.DataFrame({'time': [0], "value": [0]}),
        pd.DataFrame([(p.time, p.value) for p in pedal_events if p.number == 64], columns=['time', 'value'])])
    note_events["sustain_value"] = note_events.apply(
        lambda row: pedal_events[pedal_events['time'] <= row['start']].iloc[0]['value'], axis=1)

    edg_src, edg_dst, edg_type = [], [], []
    for j, note in note_events.iterrows():
        """note_event: e.g., Note(start=1.009115, end=1.066406, duration=0.057291, pitch=40, velocity=93)"""
        
        """onset, consecutive, and sustain neighbors, and add edges"""
        onset_nbs = note_events[np.absolute(note_events['start'] - note['start']) < cfg.graph.mid_window]
        consec_nbs = note_events[np.absolute(note_events['start'] - note['end']) < cfg.graph.mid_window]
        sustain_nbs = note_events[(note_events['start'] >= note['start']) & (note_events['end'] < note['end'])]
        if not len(consec_nbs):
            """silence edge: """
            silence_gap = pd.Series(note_events['start'] - note['end'])
            silence_gap_min = silence_gap.loc[lambda x: x > 0].min()
            silence_nbs = note_events[((silence_gap - silence_gap_min) < cfg.graph.mid_window) & (silence_gap > 0)]
            edg_src.extend([j] * len(silence_nbs))
            edg_dst.extend(silence_nbs.index)
            edg_type.extend(['silence'] * len(silence_nbs))

        edg_src.extend([j] * len(onset_nbs) 
                                    + [j] * len(consec_nbs)
                                    + [j] * len(sustain_nbs))
        edg_dst.extend(list(onset_nbs.index) 
                                    + list(consec_nbs.index)
                                    + list(sustain_nbs.index))
        edg_type.extend(['onset'] * len(onset_nbs) 
                                    + ['consecutive'] * len(consec_nbs)
                                    + ['sustain'] * len(sustain_nbs))

    edges = np.array([edg_src, edg_dst, edg_type])
    graph_dict = {}
    for type_name in ['onset', 'consecutive', 'sustain', 'silence']:
        e = edges[:, edges[2, :] == type_name]
        graph_dict[('note', type_name, 'note')] = torch.tensor(e[0].astype(int)), torch.tensor(e[1].astype(int))
        if type_name != 'onset':
            graph_dict[('note', type_name+'_rev', 'note')] = torch.tensor(e[1].astype(int)), torch.tensor(e[0].astype(int))

    graph_dict = {k: (torch.tensor(s), torch.tensor(d)) for k, (s, d) in graph_dict.items()}
    perfmidi_hg = dgl.heterograph(graph_dict)

    """add node(note) features"""
    perfmidi_hg.ndata['feat_0'] = torch.tensor(np.hstack(
        [
        # utils.get_onset_one_hot(note_events),
        np.expand_dims(np.array(note_events["start"]) % cfg.segmentation.seg_time, 1),  # the relative onset time
        # utils.get_duration_one_hot(note_events),
        np.expand_dims(np.array(note_events["duration"]), 1), 
        utils.get_pc_one_hot(note_events), utils.get_octave_one_hot(note_events)]
        ) ).float()
    
    # add level 1 features. pedal and velocity in one-hot format
    perfmidi_hg.ndata['feat_1'] = torch.tensor(np.hstack((
        utils.get_pedal_one_hot(note_events), utils.get_velocity_one_hot(note_events)
    ))).float() # 16 dim

    """add timepoint features (level -1)"""
    perfmidi_hg.ndata['feat_-1'] = torch.tensor(note_events['start']).float()

    return perfmidi_hg # (s, )


def load_musicxml(path):
    import warnings
    warnings.filterwarnings("ignore")  # mute partitura warnings

    score_data = pt.load_musicxml(path, force_note_ids=True)
    score_data = pt.score.merge_parts(score_data.parts)
    note_array = pt.utils.music.ensure_notearray(
        score_data,
        include_pitch_spelling=True, # adds 3 fields: step, alter, octave
        include_key_signature=True, # adds 2 fields: ks_fifths, ks_mode
        include_time_signature=True, # adds 2 fields: ts_beats, ts_beat_type
        # include_metrical_position=True, # adds 3 fields: is_downbeat, rel_onset_div, tot_measure_div
        include_grace_notes=True # adds 2 fields: is_grace, grace_type
    )
    return score_data, note_array


def musicxml_to_graph(path, cfg):
    
    try:  # some parsing error....
        score_data, note_array = load_musicxml(path)
        if len(note_array) == 0:
            return None
    except Exception as e:
        print("Failed on score {} with exception {}".format(os.path.splitext(os.path.basename(path))[0], e))
        return None

    # Get edges from note array
    measures = np.array([[m.start.t, m.end.t] for m in score_data.measures])
    edges, edge_types = edges_from_note_array(note_array, measures)

    # Build graph dict for dgl
    graph_dict = {}
    for type_num, type_name in edge_types.items():

        e = edges[:, edges[2, :] == type_num]
        graph_dict[('note', type_name, 'note')] = torch.tensor(e[0]), torch.tensor(e[1])
        # Pipeline for adding reverse edges on consecutive, sustain, rest, and voice edges.
        if type_name == "onset":
            continue
        graph_dict[('note', type_name+"_rev", 'note')] = torch.tensor(e[1]), torch.tensor(e[0])
    hg = dgl.heterograph(graph_dict)

    # Add node features for segmentation purpose
    hg.ndata['feat_-1'] = torch.tensor(note_array['onset_beat'].copy())

    feat_0, feat_1 = feature_extraction_score(note_array, score=score_data, include_meta=cfg.experiment.feat_level)
    # print(f'score {path} has feature dimension: {feat_1.shape}')
    hg.ndata['feat_0'] = torch.tensor(feat_0).float()
    if cfg.experiment.feat_level:
        hg.ndata['feat_1'] = torch.tensor(feat_1).float()

    return hg


def get_subgraphs(graph, cfg):
    """ 
    - split the graph into segment subgraphs 
    - remove higher level edges if feat_level is 0 (as we can't do it after the graphs are batched). Remove reverse edges if bi_dir is false
    - Also convert to homogeneous graphs when it needs to.

    Return:
        seg_subgraphs (list of dgl.DGLHeteroGraph)
    """
    if cfg.graph.homo:
        # disregard the edge types and add all of them into a new graph
        all_src, all_dst = torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)
        for etype in graph.etypes:
            src, dst = graph.edges(etype=etype)
            all_src = torch.cat((all_src, src))
            all_dst = torch.cat((all_dst, dst))
        h_graph = dgl.heterograph({("note", "dummy", "note"): (all_src, all_dst)})
        # copy over the original node features
        for k, v in graph.ndata.items():
            h_graph.ndata[k] = v
        graph = h_graph

    # only get the basic edge types if the features level is 0.
    if (not cfg.graph.homo) and cfg.experiment.feat_level == 0:
        graph = dgl.edge_type_subgraph(graph, [('note', et, 'note') for et in cfg.graph.basic_edges])
    
    # only get the non-reverse edge types if bi_dir is false.
    if (not cfg.graph.homo) and (not cfg.graph.bi_dir):
        graph = dgl.edge_type_subgraph(graph, [('note', et, 'note') for et in cfg.graph.basic_edges if ("rev" not in et)])

    # segmentation into subgraphs of music segs
    if cfg.segmentation.seg_type == "fix_time":
        window = cfg.segmentation.seg_time if cfg.experiment.input_format == "perfmidi" else cfg.segmentation.seg_beat
        nodes_onset_times = graph.ndata['feat_-1']  / window
        bins = np.arange(np.ceil(nodes_onset_times.max()), dtype=int)
        inds = np.digitize(nodes_onset_times, bins)
        seg_subgraphs = [dgl.node_subgraph(graph, np.where(inds==(i+1))[0]) for i in bins]
    else:
        if cfg.segmentation.seg_type == "fix_num": 
            n_segs = cfg.segmentation.seg_num
            l = int(graph.num_nodes() / n_segs)
        elif cfg.segmentation.seg_type == "fix_size":
            n_segs = int(graph.num_nodes() / cfg.segmentation.seg_size) + 1
            l = cfg.segmentation.seg_size        

        seg_subgraphs = [dgl.node_subgraph(graph, list(range(i*l, i*l+l))) for i in range(n_segs-1)]
        seg_subgraphs.append(dgl.node_subgraph(graph, list(range((n_segs-1)*l, graph.number_of_nodes()))))

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

    for idx, (path, l) in enumerate(zip(files, labels)):

        recompute = True
        if cfg.experiment.load_data: # load existing data
            res = utils.load_data(path, cfg)
            if type(res) == np.ndarray: # keep computing if not exist
                graph = res[0]
                recompute = False

        if recompute:
            if cfg.experiment.input_format == "perfmidi":
                graph = perfmidi_to_graph(path, cfg)
            elif cfg.experiment.input_format == "musicxml":
                res = musicxml_to_graph(path, cfg)
                if type(res) == dgl.DGLHeteroGraph:
                    graph = res
                else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                    continue
            
            utils.save_data(path, graph, cfg)
        
        batch_graphs.append(get_subgraphs(graph, cfg))
        batch_labels.append(l)
    
    if cfg.experiment.tmp:
        example = batch_graphs[13][0]
    # if cfg.experiment.tmp:
    #     byte_counts = []
    #     for piece_graphs in batch_graphs:
    #         total_bytes = 0
    #         for sg in piece_graphs:
    #             total_bytes += np.array(sg.ndata['feat_0']).nbytes
    #             for etype in ['onset', 'consecutive', 'sustain', 'silence']:
    #                 total_bytes += np.array(sg.edges(etype=etype)[0]).nbytes * 2
    #         byte_counts.append(total_bytes)
    #     byte_counts = np.array(byte_counts)
    #     hook()
    batch_graphs, batch_labels = utils.pad_batch(b, cfg, device, batch_graphs, batch_labels)

    return np.array(batch_graphs, dtype='object'), batch_labels
