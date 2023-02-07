import numpy as np
import torch
import dgl

def pad_batch(b, cfg, device, batch_data, batch_labels):
    """padding batch: 
    1. refill value to batch size: when the processed batch lost data because of parsing error, 
        refill the batch with the last one in the batch
    2. for batch with variable segments length, pad the shorter data util they have the same
        number of segments.
        - For matrix: 
        - For sequence: pad the remaining segments with 0 (a non-vocab value)
    """

    # refill
    n_skipped = b - len(batch_data)
    batch_data += [batch_data[-1]] * n_skipped
    batch_labels = torch.tensor(batch_labels + [batch_labels[-1]] * n_skipped, device=device)

    # pad seg
    max_n_segs = max([len(data) for data in batch_data])
    if cfg.experiment.symrep != "graph":
        batch_data = [
            np.concatenate((data, np.zeros((max_n_segs - len(data), *data.shape[1:])) ))
            for data in batch_data
        ]

    return batch_data, batch_labels
