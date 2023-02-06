import numpy as np
import torch

def pad_batch(b, device, batch_data, batch_labels):
    """padding batch: when the processed batch lost data because of parsing error, refill the batch with the last one in the batch"""

    n_skipped = b - len(batch_data)
    batch_data += [batch_data[-1]] * n_skipped
    batch_labels = torch.tensor(batch_labels + [batch_labels[-1]] * n_skipped, device=device)

    return batch_data, batch_labels
