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
        return res

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
        perfmidi_g.ndata['general_note_feats'] = torch.tensor(np.array(seg_note_events)) # general features are just the 5 terms in note events, plus pedal value at the time

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


def musicxml_to_matrix(path, cfg):
    """Process musicXML to roll matrices for training"""

    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings

    try: # some parsing error....
        score_data = pt.load_musicxml(path)
        note_events = pd.DataFrame(score_data.note_array(), columns=score_data.note_array().dtype.names)
    except:
        return None

    end_time_divs = note_events['onset_div'].max() + note_events['duration_div'].max()
    frames_num = cfg.matrix.resolution
    frames_per_second = (frames_num / end_time_divs)
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    voice_roll = np.zeros((frames_num, cfg.matrix.bins))

    for _, note_event in note_events.iterrows():
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""

        bgn_frame = min(int(round((note_event['onset_div']) * frames_per_second)), frames_num-1)
        fin_frame = min(bgn_frame + int(round((note_event['duration_div']) * frames_per_second)), frames_num-1)
        voice_roll[bgn_frame : fin_frame + 1, note_event['pitch']] = note_event['voice']
        onset_roll[bgn_frame, note_event.pitch] = 1


    """Clip rolls into segments and concatenate"""
    onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    voice_roll = rearrange(voice_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    matrices = rearrange([onset_roll, voice_roll], "c s f n -> s c f n")

    return matrices # (s 2 h w)




def batch_to_graph(batch, cfg, device):
    """Map the batch to input graphs

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        graph: 
        label: (b, )
    """
    files, labels = batch
    b = len(batch[0])
    batch_graphs = []

    if not os.path.exists(cfg.graph.save_dir):
        os.makedirs(cfg.graph.save_dir)
        with open(f"{cfg.graph.save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

    for idx, (path, _) in enumerate(zip(files, labels)):
        if cfg.experiment.input_format == "perfmidi":
            seg_graphs = perfmidi_to_graph(path, cfg)
        elif cfg.experiment.input_format == "musicxml":
            res = musicxml_to_graph(path, cfg)
            if type(res) == np.ndarray:
                seg_graphs = res
            else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                continue
        elif cfg.experiment.input_format == "kern":
            seg_graphs = kern_to_graph(path, cfg)
        batch_graphs.append(seg_graphs)
    
    batch_graphs, batch_labels = utils.pad_batch(b, device, batch_graphs, batch_labels)

    return batch_graphs, labels