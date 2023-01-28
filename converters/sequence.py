import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat
import pandas as pd
from miditok import MIDILike
from miditoolkit import MidiFile


def construct_tokenizer(cfg):

    tokenizer = MIDILike(
        range(cfg.sequence.pr_start, cfg.sequence.pr_end), 
        {(0, 12): cfg.sequence.beat_res}, # given the bpm 120, this can only represent time gaps less than 6s
        cfg.sequence.nb_velocities, 
        additional_tokens = {'Chord': False, 'Rest': False, 'Program': False,
                    'Tempo': True, 
                    'nb_tempos': 32,  # nb of tempo bins
                    'tempo_range': (40, 250)},  # (min, max)
        mask=True)
    
    return tokenizer

def perfmidi_to_sequence(path, tokenizer, cfg):
    """Process MIDI events to sequences using miditok"""

    midi = MidiFile(path)
    tokens = tokenizer(midi)[0] # (l, )

    """Clip rolls into segments and add padding"""
    l = int(len(tokens) / cfg.experiment.n_segs)
    seg_tokens = []
    for i in range(cfg.experiment.n_segs):
        seg_tokens.append(np.pad(tokens[ i*l: i*l+l ], 
                            (0, cfg.sequence.seq_len - len(tokens[ i*l: i*l+l ])), 
                            mode="constant",
                            constant_values=1)) 
    seg_tokens = np.array(seg_tokens)
    assert(seg_tokens.shape == (cfg.experiment.n_segs, cfg.sequence.seq_len))
    return seg_tokens # (s l)


def batch_to_sequence(batch, cfg, device):
    """Map the batch to input token sequences 

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        matrix: (b, n_segs, pitch_bins, resolution)
        label: (b, )
    """
    files, labels = batch
    batch_sequence = []

    tokenizer = construct_tokenizer(cfg)
    for path, _ in zip(files, labels):
        if cfg.experiment.input_format == "perfmidi":
            seg_sequences = perfmidi_to_sequence(path, tokenizer, cfg)
        elif cfg.experiment.input_format == "musicxml":
            seg_sequences = musicxml_to_sequence(path, cfg)
        elif cfg.experiment.input_format == "kern":
            seg_sequences = kern_to_sequence(path, cfg)
        batch_sequence.append(seg_sequences)

    batch_sequence = torch.tensor(np.array(batch_sequence), device=device, dtype=torch.float32) 
    return batch_sequence, labels
