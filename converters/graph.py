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


def perfmidi_to_graph(path, cfg):
    """Process MIDI events to graphs for training
    
    Returns:
        graphs: list of dgl.DGLGraph
    """
    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    note_events = pd.DataFrame([(n.start, n.end, n.duration, n.pitch, n.velocity) for n in note_events], 
                columns=['start', 'end', 'duration', 'pitch', 'velocity'])
    pedal_events = perf_data.instruments[0].control_changes # TODO: deal with pedal events!!!
    pedal_events = pd.concat([
        pd.DataFrame({'time': [0], "value": [0]}),
        pd.DataFrame([(p.time, p.value) for p in pedal_events if p.number == 67], columns=['time', 'value'])])

    note_events["sustain_value"] = note_events.apply(
        lambda row: pedal_events[pedal_events['time'] < row['start']].iloc[0]['value'], axis=1)

    perfmidi_graphs = []
    for i in range(cfg.experiment.n_segs):
        seg_note_events = note_events.iloc[i:]
        perfmidi_g = dgl.graph(([], []), num_nodes=len(seg_note_events))

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

    print('here')
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

    for idx, (path, _) in enumerate(zip(files, labels)):
        if cfg.experiment.input_format == "perfmidi":
            seg_matrices = perfmidi_to_graph(path, cfg)
        elif cfg.experiment.input_format == "musicxml":
            res = musicxml_to_graph(path, cfg)
            if type(res) == np.ndarray:
                seg_matrices = res
            else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                labels = torch.cat((labels[0:idx], labels[idx+1:]))
                continue
        elif cfg.experiment.input_format == "kern":
            seg_matrices = kern_to_graph(path, cfg)
        batch_graphs.append(seg_matrices)
    
    n_skipped = b - len(batch_graphs)
    batch_graphs += [batch_graphs[-1]] * n_skipped
    labels = torch.cat((labels, repeat(labels[-1:], "n -> (n b)", b=n_skipped)))

    batch_graphs = np.array(batch_graphs)
    return batch_graphs, labels
