import os
import torch
import numpy as np
import copy
import partitura as pt
from einops import rearrange, repeat
import pandas as pd
from miditoolkit import MidiFile
from einops import repeat
import utils as utils


def clip_segs(tokens, cfg):
    """clip the token sequence according to segmentation scheme

    Return:
        seg_tokens: np.array: (n_segs, seg_length)
    """

    """choose the number of segments to clip"""
    if cfg.segmentation.seg_type == "fix_num":
        n_segs = cfg.experiment.n_segs
        l = int(len(tokens) / cfg.experiment.n_segs)
    elif cfg.segmentation.seg_type == "fix_size":
        n_segs = int(len(tokens) / cfg.sequence.max_seq_len) + 1
        l = cfg.sequence.max_seq_len


    """Clip rolls into segments and add padding"""
    seg_tokens = []
    for i in range(n_segs): 
        seg_tokens.append(tokens[ i*l: i*l+l ][:cfg.sequence.max_seq_len])
    return seg_tokens


def pad_segs(seg_tokens, cfg):
    if cfg.sequence.mid_encoding == "CPWord":
        seg_tokens = [np.concatenate([seg_token, repeat(np.array([0] * 6), 'd -> k d', k=( cfg.sequence.max_seq_len - len(seg_token)))])
                    for seg_token in seg_tokens if seg_token]        
    seg_tokens = [np.pad(seg_token, (0, cfg.sequence.max_seq_len - len(seg_token)), mode="constant", constant_values=0)
                 for seg_token in seg_tokens]
    return np.array(seg_tokens) # ()
    

def perfmidi_to_sequence(path, tokenizer, cfg):
    """Process MIDI events to sequences using miditok
    - segment the sequence in various segmentation scheme, and then pad the sequences
    
    Returns:
        seg_tokens: (n_segs, max_seq_len)
    """
    midi = MidiFile(path)
    if cfg.segmentation.seg_type == "fix_time":
        """For the fix_time segmentation, we get different segments in midi and then tokenize them"""
        seg_tokens, i = [], 0
        mapping = midi.get_tick_to_time_mapping()
        instrument_track = copy.deepcopy(midi.instruments[0])
        while True:
            # _midi = copy.deepcopy(midi)
            # instrument_track = _midi.instruments[0]
            start, end = (i)*cfg.segmentation.seg_time, (i+1)*cfg.segmentation.seg_time 
            midi.instruments[0].notes = [note for note in instrument_track.notes 
                                        if (note.start < len(mapping) and 
                                            (mapping[note.start] < end and (mapping[note.start]) > start))]
            if not midi.instruments[0].notes:
                break
            print(len(midi.instruments[0].notes))
            tokens = tokenizer(midi)[0]
            utils.try_save_BPE_tokens(tokenizer, tokens, cfg)
            if cfg.sequence.BPE:
                tokens = tokenizer.apply_bpe(tokens)
            seg_tokens.append(tokens[:cfg.sequence.max_seq_len])
            i += 1
    else:
        tokens = tokenizer(midi)[0] # (l, )
        if cfg.sequence.BPE:
            tokens = tokenizer.apply_bpe(tokens)
        seg_tokens = clip_segs(tokens, cfg)

    seg_tokens = pad_segs(seg_tokens, cfg)
    assert(seg_tokens.shape[1] == cfg.sequence.max_seq_len)
    return seg_tokens # (s l)


def musicxml_to_sequence(path, tokenizer, cfg):
    """Process musicxml to sequences using miditok"""
    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings

    try:
        score = pt.load_musicxml(path)
        if "Kreisleriana,_Op._16/VIII._Schnell_und_spielend/" in path:
            raise RuntimeError
    except Exception as e:
        print("Failed on score {} with exception {}".format(os.path.splitext(os.path.basename(path))[0], e))
        return None
    
    if cfg.segmentation.seg_type == "fix_time":
        """For the fix_time segmentation, we get different segments in score and then tokenize them"""
        seg_tokens, i = [], 0
        for i in range(int(score.note_array()['onset_beat'].max() / cfg.segmentation.seg_beat) + 1):
            tokens = tokenizer.track_to_tokens(score, start_end_beat=(i*cfg.segmentation.seg_beat, (i+1)*cfg.segmentation.seg_beat))
            utils.try_save_BPE_tokens(tokenizer, tokens, cfg)
            if cfg.sequence.BPE:
                tokens = tokenizer.apply_bpe(tokens)
            seg_tokens.append(tokens[:cfg.sequence.max_seq_len])
            print(len(tokens))
    else:
        tokens = tokenizer.track_to_tokens(score)
        if cfg.sequence.BPE:
            tokens = tokenizer.apply_bpe(tokens)
        seg_tokens = clip_segs(tokens, cfg)    

    seg_tokens = pad_segs(seg_tokens, cfg)

    assert(seg_tokens.shape[1] == cfg.sequence.max_seq_len)
    return seg_tokens # (s l)


def batch_to_sequence(batch, cfg, device, tokenizer):
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

    for idx, (path, l) in enumerate(zip(files, labels)):
        # print(path)
        recompute = True
        if cfg.experiment.load_data: # load existing data
            res = utils.load_data(path, cfg)
            if type(res) == np.ndarray: # keep computing if not exist
                seg_sequences =  res
                recompute = False

        # events = tokenizer.tokens_to_events(list(seg_sequences[0]))
        if recompute:
            if cfg.experiment.input_format == "perfmidi":
                seg_sequences = perfmidi_to_sequence(path, tokenizer, cfg)
            elif cfg.experiment.input_format == "musicxml":
                res = musicxml_to_sequence(path, tokenizer, cfg)
                if type(res) == np.ndarray:
                    seg_sequences = res
                else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                    continue

            utils.save_data(path, seg_sequences, cfg)

        batch_sequence.append(seg_sequences)
        batch_labels.append(l)
    
    if cfg.experiment.tmp:
        example = batch_sequence[10][0, :50]
        for e in tokenizer.tokens_to_events(example):
            print(e)
    #     byte_counts = []
    #     for piece_segments in batch_sequence:
    #         total_bytes = 0
    #         for ss in piece_segments:
    #             total_bytes += np.array(ss).nbytes
    #         byte_counts.append(total_bytes)
    #     byte_counts = np.array(byte_counts)
    batch_sequence, batch_labels = utils.pad_batch(b, cfg, device, batch_sequence, batch_labels)
    batch_sequence = torch.tensor(np.array(batch_sequence), device=device, dtype=torch.float32) 
    return batch_sequence, batch_labels
