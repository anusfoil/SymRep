import os
import numpy as np
import pandas as pd
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


def load_data(path, cfg):
    """generic load data function for any type of representation """
    
    save_dir = cfg.experiment.data_save_dir
    if cfg.experiment.symrep in ["matrix", "sequence"]: # add further parameterized dirs for matrix and sequence
        save_dir = f"{save_dir}/{cfg[cfg.experiment.symrep].save_dir}"

    if not os.path.exists(save_dir):
        return None

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    res = metadata[metadata['path'] == path]
    if len(res):
        if cfg.experiment.symrep == "graph":
            return np.array(dgl.load_graphs(f"{save_dir}/{res['save_dir'].iloc[0]}")[0])
        else:
            return np.load(f"{save_dir}/{res['save_dir'].iloc[0]}")


def save_data(path, computed_data, cfg):
    """generic save_data function for any type of representation
    - write the corresponding path with the saved index in metadata.csv
    
    graphs: dgl 
    matrix and sequence: numpy npy
    """

    save_dir = cfg.experiment.data_save_dir
    if cfg.experiment.symrep in ["matrix", "sequence"]: # add further parameterized dirs for matrix and sequence
        save_dir = f"{save_dir}/{cfg[cfg.experiment.symrep].save_dir}"

    if not os.path.exists(save_dir): # make saving dir if not exist
        os.makedirs(save_dir)
        with open(f"{save_dir}/metadata.csv", "w") as f:
            f.write("path,save_dir\n")

    metadata = pd.read_csv(f"{save_dir}/metadata.csv")
    if path in metadata['path']: # don't write and save if it existed
        return

    N = len(metadata) 
    if cfg.experiment.symrep == 'graph':
        save_path = f"{N}.dgl"
        dgl.save_graphs(f"{save_dir}/{save_path}", computed_data)
    else:
        save_path = f"{N}.npy"
        np.save(f"{save_dir}/{save_path}", computed_data)
    
    metadata = metadata.append({"path": path, "save_dir": save_path}, ignore_index=True)
    metadata.to_csv(f"{save_dir}/metadata.csv", index=False)


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
