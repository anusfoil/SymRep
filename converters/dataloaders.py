import os, sys
sys.path.extend(["../../symrep", "../../"])
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torchnlp.encoders import LabelEncoder
import pandas as pd

def drop_uncommon_classes(metadata, label_col, threshold=0.03):
    """sort the classes by distribution and drop the data that consists of the last threshold\% of the set"""

    count = metadata[label_col].value_counts().to_frame("count")
    count['agg_percent'] = count.loc[::-1, 'count'].cumsum()[::-1] / count.sum().values
    uncommon_label = count[count['agg_percent'] < threshold].index
    return metadata[~metadata[label_col].isin(uncommon_label)]


class LengthSampler(BatchSampler):
    """Bucket the data that's of similar length into one batch, to avoid padding redundancy
    
    How it works:
        using the duration from metadata, order the pieces from short to long and then sample
    """
    
    def __init__(self, dataset):
        
        self.dataset = dataset
        self.metadata = dataset.dataset.metadata.iloc[dataset.indices]
        self.num_samples = len(dataset)
        self.ordered_index = self.order_by_length()
        
    def __iter__(self):
        for i in self.ordered_index:
            yield self.dataset.indices.index(i)
    
    def __len__(self):
        return self.num_samples
    
    def order_by_length(self):
        return self.metadata.sort_values(by=['track_duration'], ascending=False).index


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
            self.metadata = drop_uncommon_classes(self.metadata, 'composer')
            self.label_column = self.metadata['composer']
        elif self.task == "performer_id":
            self.metadata = drop_uncommon_classes(self.metadata, 'perfomer')
            self.label_column = self.metadata['performer']
        elif self.task == "difficulty_id":
            self.metadata = drop_uncommon_classes(self.metadata, 'difficulty_label')
            self.metadata = self.metadata[~self.metadata['difficulty_label'].isna()]
            self.label_column = self.metadata['difficulty_label']

        self.label_encoder = LabelEncoder(self.label_column.unique(), 
                reserved_labels=['unknown'], unknown_index=0)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column.iloc[idx]
        label = self.label_encoder.encode(label)

        if self.input_format == "perfmidi":
            return (self.dataset_dir + self.metadata['midi_performance'].iloc[idx], label)
        elif self.input_format == "musicxml":
            return (self.dataset_dir + self.metadata['xml_score'].iloc[idx], label)
        else:
            raise RuntimeError("Wrong type of input format!")


class ATEPP(Dataset):
    """Returns the (symbolic data, label) datapair from the ATEPP dataset.
    The task-corresponding label is encoded with the label_encoder
    """
    def __init__(self, cfg):
        self.input_format = cfg.experiment.input_format
        self.task = cfg.experiment.task
        self.dataset_dir = cfg.dataset.ATEPP.dataset_dir
        self.metadata = pd.read_csv(cfg.dataset.ATEPP.metadata_file, encoding="utf-8-sig")

        if self.input_format == "musicxml":
            # filter out the ones without score
            self.metadata = self.metadata[~self.metadata['score_path'].isna()]

        if self.task == "composer_id":
            self.metadata = drop_uncommon_classes(self.metadata, 'composer')
            self.label_column = self.metadata['composer']
        elif self.task == "performer_id":
            self.metadata = drop_uncommon_classes(self.metadata, 'artist', threshold=0.1)
            self.label_column = self.metadata['artist']

        self.label_encoder = LabelEncoder(self.label_column.unique(), 
                reserved_labels=['unknown'], unknown_index=0)
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.label_column.iloc[idx]
        label = self.label_encoder.encode(label)

        if self.input_format == "perfmidi":
            return (os.path.join(self.dataset_dir, self.metadata['midi_path'].iloc[idx]), label)
        elif self.input_format == "musicxml":
            return (self.dataset_dir + self.metadata['score_path'].iloc[idx], label)


