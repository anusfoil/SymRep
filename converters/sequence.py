import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat
import pandas as pd
from miditok import MIDILike, REMI, MusicXML
from miditoolkit import MidiFile
import utils as utils


def construct_tokenizer(cfg):

    if cfg.experiment.input_format == "perfmidi":
        tokenizer = eval(cfg.sequence.mid_encoding)( # MidiLike or REMI
            range(cfg.sequence.pr_start, cfg.sequence.pr_end), 
            {(0, 12): cfg.sequence.beat_res}, # given the bpm 120, this can only represent time gaps less than 6s
            cfg.sequence.nb_velocities, 
            additional_tokens = {'Chord': False, 'Rest': False, 'Program': False,
                        'Tempo': True, 
                        'nb_tempos': 32,  # nb of tempo bins
                        'tempo_range': (40, 250)},  # (min, max)
            mask=False)
    elif cfg.experiment.input_format == "musicxml":
        tokenizer = MusicXML(
            range(cfg.sequence.pr_start, cfg.sequence.pr_end), 
            {(0, 12): cfg.sequence.beat_res}, # given the bpm 120, this can only represent time gaps less than 6s
            cfg.sequence.nb_velocities, 
            additional_tokens = {'Chord': False, 'Rest': False, 'Program': False,
                        'Tempo': True, 
                        'nb_tempos': 32,  # nb of tempo bins
                        'tempo_range': (40, 250)},  # (min, max)
            mask=False)
        
    assert(len(tokenizer.vocab.event_to_token.keys()) < 500) # embeding project at most 500 value
    return tokenizer


def clip_and_pad(tokens, cfg):
    """
    1. clip the token sequence into fixed or flexible length
    2. pad each segment 
    
    Return:
        seg_tokens: np.array: (n_segs, seg_length)
    """

    """choose the number of segments to clip"""
    if cfg.experiment.n_segs:
        n_segs = cfg.experiment.n_segs
        l = int(len(tokens) / cfg.experiment.n_segs)
    else:
        n_segs = int(len(tokens) / cfg.sequence.seq_len) + 1
        l = cfg.sequence.seq_len

    """Clip rolls into segments and add padding"""
    seg_tokens = []
    for i in range(n_segs): 
        seg_token = tokens[ i*l: i*l+l ][:cfg.sequence.seq_len] # clip the segments with maximum seq len - some parts are lost
        seg_tokens.append(np.pad(seg_token, 
                            (0, cfg.sequence.seq_len - len(seg_token)), 
                            mode="constant",
                            constant_values=1)) 
    seg_tokens = np.array(seg_tokens)
    return seg_tokens


def perfmidi_to_sequence(path, tokenizer, cfg):
    """Process MIDI events to sequences using miditok
    tokenization scheme: MidiLike, REMI
    
    Returns:
        seg_tokens: (n_segs, max_seq_len)
    """

    midi = MidiFile(path)
    tokens = tokenizer(midi)[0] # (l, )

    seg_tokens = clip_and_pad(tokens, cfg)

    assert(seg_tokens.shape[1] == cfg.sequence.seq_len)
    return seg_tokens # (s l)


def musicxml_to_sequence(path, tokenizer, cfg):
    """Process musicxml to sequences using miditok"""
    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings

    # midi = MidiFile(path)
    try:
        score = pt.load_musicxml(path)
    except:
        return None
    tokens = tokenizer.track_to_tokens(score) # (l, )

    seg_tokens = clip_and_pad(tokens, cfg)

    assert(seg_tokens.shape[1] == cfg.sequence.seq_len)
    return seg_tokens # (s l)


def batch_to_sequence(batch, cfg, device):
    """Map the batch to input token sequences 

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        batch_sequence: (b, )
        batch_label: (b, )
    """
    files, labels = batch
    b = len(batch[0])
    batch_sequence, batch_labels = [], []

    tokenizer = construct_tokenizer(cfg)
    for idx, (path, l) in enumerate(zip(files, labels)):
        if cfg.experiment.input_format == "perfmidi":
            seg_sequences = perfmidi_to_sequence(path, tokenizer, cfg)
        elif cfg.experiment.input_format == "musicxml":
            res = musicxml_to_sequence(path, tokenizer, cfg)
            if type(res) == np.ndarray:
                seg_sequences = res
            else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                continue
        elif cfg.experiment.input_format == "kern":
            seg_sequences = kern_to_sequence(path, tokenizer, cfg)
        batch_sequence.append(seg_sequences)
        batch_labels.append(l)

    batch_sequence, batch_labels = utils.pad_batch(b, cfg, device, batch_sequence, batch_labels)
    batch_sequence = torch.tensor(np.array(batch_sequence), device=device, dtype=torch.float32) 
    return batch_sequence, batch_labels
