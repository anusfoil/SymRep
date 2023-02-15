import numpy as np
import torch
import dgl

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


def get_pedal_one_hot(note_events):
    """Get one-hot encoding of sustain pedal values."""
    one_hot = np.zeros((len(note_events), 8))
    idx = (np.arange(len(note_events)), np.floor_divide(note_events["sustain_value"], 16).astype(int))
    one_hot[idx] = 1
    return one_hot

def get_velocity_one_hot(note_events):
    """Get one-hot encoding of velocity values."""
    one_hot = np.zeros((len(note_events), 8))
    idx = (np.arange(len(note_events)), np.floor_divide(note_events["velocity"], 16).astype(int))
    one_hot[idx] = 1
    return one_hot


def pad_batch(b, cfg, device, batch_data, batch_labels):
    """padding batch: 
    1. refill value to batch size: when the processed batch lost data because of parsing error, 
        refill the batch with the last one in the batch
    2. for batch with variable segments length, pad the shorter data util they have the same
        number of segments.
        - For matrix: also pad with all-zero matrices
        - For sequence: pad the remaining segments with 0 (a non-vocab value)
    """

    # refill
    n_skipped = b - len(batch_data)
    batch_data += [batch_data[-1]] * n_skipped
    batch_labels = torch.tensor(batch_labels + [batch_labels[-1]] * n_skipped, device=device)

    # pad seg
    max_n_segs = max([len(data) for data in batch_data])
    if cfg.experiment.symrep != "graph":
        batch_data = [*map(lambda data: np.concatenate((data, np.zeros((max_n_segs - len(data), *data.shape[1:])) )),
                          batch_data)]
        # batch_data = [
        #     np.concatenate((data, np.zeros((max_n_segs - len(data), *data.shape[1:])) ))
        #     for data in batch_data
        # ]
    
    return batch_data, batch_labels
