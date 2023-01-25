import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import repeat

def perfmidi_to_matrix(path, cfg):
    
    perf_data = pretty_midi.PrettyMIDI(path)
    # for a 160 second piece, roughtly have a resolution of 0.05s
    seg_dur = perf_data.get_end_time() / cfg.experiment.n_segs
    fs = cfg.matrix.resolution / seg_dur # sampling frequency per second
    seg_matrices = np.nan_to_num(np.array([
        perf_data.get_piano_roll(fs=fs, times=np.arange(i*seg_dur, (i+1)*seg_dur, 1/fs))[:, :cfg.matrix.resolution] 
        for i in range(4)
    ]))
    assert np.array_equal(seg_matrices, seg_matrices)
    
    return seg_matrices

def musicxml_to_matrix(path, cfg):
    
    score_data = pt.load_musicxml(path)
    # for a 160 second piece, roughtly have a resolution of 0.05s
    seg_dur = perf_data.get_end_time() / cfg.experiment.n_segs
    fs = cfg.matrix.resolution / seg_dur # sampling frequency per second
    seg_matrices = np.nan_to_num(np.array([
        perf_data.get_piano_roll(fs=fs, times=np.arange(i*seg_dur, (i+1)*seg_dur, 1/fs))[:, :cfg.matrix.resolution] 
        for i in range(4)
    ]))
    assert np.array_equal(seg_matrices, seg_matrices)
    
    return seg_matrices

def batch_to_matrix(batch, cfg):
    """Map the batch to input piano roll matrices, and label into index.
    TODO: try onsets+frames 3d matrix, and try the pedal matrix as well

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        matrix: (b, n_segs, pitch_bins, resolution)
        label: (b, )
    """
    files, labels = batch
    batch_matrix = []
    for path, _ in zip(files, labels):
        if cfg.experiment.input_format == "perfmidi":
            seg_matrices = perfmidi_to_matrix(path, cfg)
        elif cfg.experiment.input_format == "musicxml":
            seg_matrices = musicxml_to_matrix(path, cfg)
        batch_matrix.append(seg_matrices)

    batch_matrix = torch.tensor(repeat(np.array(batch_matrix), "b s h w -> b s c h w", c=1), device=cfg.experiment.device, dtype=torch.float32) # add channel dimension

    return batch_matrix, labels
