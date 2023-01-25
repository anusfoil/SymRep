import sys
sys.path.extend(["../../symrep", "../../"])
import torch
from torch.utils.data import Dataset, DataLoader
import torchnlp 
from torchnlp.encoders import LabelEncoder
import pandas as pd

import hook


class ASAP(Dataset):
    """Returns the (symbolic data, label) datapair from the ASAP dataset.
    The task-corresponding label is encoded with the label_encoder
    """
    def __init__(self, cfg):
        self.input_format = cfg.experiment.input_format
        self.task = cfg.experiment.task
        self.metadata = pd.read_csv(cfg.dataset.ASAP.metadata_file)
        self.dataset_dir = cfg.dataset.ASAP.dataset_dir

        if self.task == "composer_id":
            self.label_column = self.metadata['composer']
        elif self.task == "performer_id":
            self.label_column = self.metadata['performer']

        self.label_encoder = LabelEncoder(self.label_column.unique(), 
                reserved_labels=['unknown'], unknown_index=0)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column[idx]
        label = self.label_encoder.encode(label)

        if self.input_format == "perf_midi":
            return (self.dataset_dir + self.metadata['midi_performance'][idx], label)
        elif self.input_format == "musicxml":
            return (self.metadata['xml_score'][idx], label)
        else:
            raise RuntimeError("Wrong type of input format!")



