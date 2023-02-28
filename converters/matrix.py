import os
import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat
import pandas as pd
import utils as utils

def midi_generate_rolls(note_events, pedal_events, cfg, duration=None):
    """Given the list of note_events, paint the rolls based on the duration of the segment and resolution
    Adapted from https://github.com/bytedance/piano_transcription/blob/master/utils/utilities.py
    """

    frames_num = cfg.matrix.resolution
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    velocity_roll = np.zeros((frames_num, cfg.matrix.bins))

    if not note_events:
        return onset_roll, velocity_roll
    
    start_delta = int(min([n.start for n in note_events]))
    if not duration:
        duration = note_events[-1].end - note_events[0].start + 1
    frames_per_second = (cfg.matrix.resolution / duration)

    for note_event in note_events:
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""

        bgn_frame = min(int(round((note_event.start - start_delta) * frames_per_second)), frames_num-1)
        fin_frame = min(int(round((note_event.end - start_delta) * frames_per_second)), frames_num-1)
        velocity_roll[bgn_frame : fin_frame + 1, note_event.pitch] = (
            note_event.velocity if cfg.experiment.feat_level else 1)
        onset_roll[bgn_frame, note_event.pitch] = 1

    if cfg.experiment.feat_level:
        for pedal_event in pedal_events:
            """pedal_event: e.g., ControlChange(number=67, value=111, time=5.492188)"""

            if pedal_event.number == 64: ped_index = 128
            elif pedal_event.number == 66: ped_index = 129
            elif pedal_event.number == 67: ped_index = 130
            else: continue

            bgn_frame = min(int(round((pedal_event.time - start_delta) * frames_per_second)), frames_num-1)
            velocity_roll[bgn_frame : , ped_index] = pedal_event.value
            onset_roll[bgn_frame, ped_index] = 1

    return onset_roll, velocity_roll


def musicxml_generate_rolls(note_events, cfg):

    if len(note_events) == 0:
        return None

    start_delta = int(min([n['onset_div'] for n in note_events]))

    end_time_divs = note_events['onset_div'].max() + note_events['duration_div'].max()
    frames_num = cfg.matrix.resolution
    frames_per_second = (frames_num / end_time_divs)
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    voice_roll = np.zeros((frames_num, cfg.matrix.bins))

    for note_event in note_events:

        bgn_frame = min(int(round((note_event['onset_div'] - start_delta) * frames_per_second)), frames_num-1)
        fin_frame = min(bgn_frame + int(round((note_event['duration_div']) * frames_per_second)), frames_num-1)
        voice_roll[bgn_frame : fin_frame + 1, note_event['pitch']] = (
            note_event['voice'] if cfg.experiment.feat_level else 1)
        onset_roll[bgn_frame, note_event['pitch']] = 1

    if cfg.experiment.feat_level:
        # add the score markings feature to matrix.
        raise NotImplementedError

    return onset_roll, voice_roll


def perfmidi_to_matrix(path, cfg):
    """Process MIDI events to roll matrices for training"""
    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    pedal_events = perf_data.instruments[0].control_changes

    if cfg.segmentation.seg_type == "fix_num":

        onset_roll, velocity_roll = midi_generate_rolls(note_events, pedal_events, cfg)
        onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
        velocity_roll = rearrange(velocity_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
    
    elif cfg.segmentation.seg_type == "fix_size":
        """in matrices, we define the <size> as amount of musical event"""
        onset_roll, velocity_roll = [], []
        __onset_append, __velocity_append = onset_roll.append, velocity_roll.append # this make things faster..
        """get segments by size and produce rolls"""
        for i in range(0, len(note_events), cfg.segmentation.seg_size):
            end = i+cfg.segmentation.seg_size
            seg_note_events = note_events[i:end]
            timings = [*map(lambda n: n.start, seg_note_events)]
            start, end = min(timings), max(timings)
            seg_pedal_events = [*filter(lambda p: (p.time > start and p.time < end)
                                , pedal_events)]
            seg_onset_roll, seg_velocity_roll = midi_generate_rolls(seg_note_events, seg_pedal_events, cfg)
            __onset_append(seg_onset_roll)
            __velocity_append(seg_velocity_roll)            
    
    elif cfg.segmentation.seg_type == "fix_time":  
        duration = cfg.segmentation.seg_time
        onset_roll, velocity_roll = [], []
        __onset_append, __velocity_append = onset_roll.append, velocity_roll.append # this make things faster..
        """get segment by time and produce rolls"""
        for i in range(0, int(perf_data.get_end_time()), cfg.segmentation.seg_time):
            start, end = i, i + cfg.segmentation.seg_time
            seg_note_events = [*filter(lambda n: (n.start > start and n.end < end) 
                                          , note_events)]  # losing the cross segment events..
            seg_pedal_events = [*filter(lambda p: p.time > start and p.time < end
                                           , pedal_events)]
            seg_onset_roll, seg_velocity_roll = midi_generate_rolls(seg_note_events, seg_pedal_events, cfg, duration=duration)
            __onset_append(seg_onset_roll)
            __velocity_append(seg_velocity_roll)

    matrices = torch.tensor(np.array([onset_roll, velocity_roll]))
    matrices = rearrange(matrices, "c s f n -> s c f n") # stack them in channel, c=2
    return matrices # (s 2 h w)


def musicxml_to_matrix(path, cfg):
    """Process musicXML to roll matrices for training"""

    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings

    try: # some parsing error....
        score_data = pt.load_musicxml(path)
        note_events = score_data.note_array()
    except Exception as e:
        print(f'failed on score {path} with exception {e}')
        return None

    if cfg.segmentation.seg_type == "fix_num":
        onset_roll, voice_roll = musicxml_generate_rolls(note_events, cfg)
        onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
        voice_roll = rearrange(voice_roll, "(s f) n -> s f n", s=cfg.segmentation.seg_num)
    
    elif cfg.segmentation.seg_type == "fix_size":
        """in matrices, we define the <size> as amount of musical event"""
        onset_roll, voice_roll = [], []
        __onset_append, __voice_append = onset_roll.append, voice_roll.append # this make things faster..
        """get segments by size and produce rolls"""
        for i in range(0, len(note_events), cfg.segmentation.seg_size):
            end = i+cfg.segmentation.seg_size
            seg_note_events = note_events[i:end]
            res = musicxml_generate_rolls(seg_note_events, cfg)
            if res:
                seg_onset_roll, seg_voice_roll = res
                __onset_append(seg_onset_roll)
                __voice_append(seg_voice_roll)        
    
    elif cfg.segmentation.seg_type == "fix_time":  
        onset_roll, voice_roll = [], []
        __onset_append, __voice_append = onset_roll.append, voice_roll.append # this make things faster..
        """get segment by time (in beats) and produce rolls"""
        for i in range(0, int(max(note_events['onset_beat'])), cfg.segmentation.seg_beat):
            start, end = i, i + cfg.segmentation.seg_beat
            seg_note_events = note_events[(note_events['onset_beat'] > start) & (note_events['onset_beat'] < end)] # losing the cross segment events..
            res = musicxml_generate_rolls(seg_note_events, cfg)
            if res:
                seg_onset_roll, seg_voice_roll = res
                __onset_append(seg_onset_roll)
                __voice_append(seg_voice_roll)

    matrices = torch.tensor(np.array([onset_roll, voice_roll]))
    matrices = rearrange(matrices, "c s f n -> s c f n") # stack them in channel, c=2
    return matrices # (s 2 h w)


def batch_to_matrix(batch, cfg, device):
    """Map the batch to input piano roll matrices

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        matrix: (b, n_segs, n_channels, resolution, pitch_bins)
        label: (b, )
    """
    files, labels = batch
    b = len(batch[0])
    batch_matrix, batch_labels = [], []

    for idx, (path, l) in enumerate(zip(files, labels)):

        recompute = True
        if cfg.experiment.load_data: # load existing data
            res = utils.load_data(path, cfg)
            if type(res) == np.ndarray: # keep computing if not exist
                seg_matrices =  res
                if cfg.matrix.n_channels == 1:
                    seg_matrices = seg_matrices[:, 1:, :, :]
                recompute = False

        if recompute:
            if cfg.experiment.input_format == "perfmidi":
                seg_matrices = perfmidi_to_matrix(path, cfg)
            elif cfg.experiment.input_format == "musicxml":
                res = musicxml_to_matrix(path, cfg)
                if type(res) == torch.Tensor:
                    seg_matrices = res
                else: # in case that the xml has parsing error, we skip and copy existing data at the end.
                    continue
            elif cfg.experiment.input_format == "kern":
                seg_matrices = kern_to_matrix(path, cfg)

            utils.save_data(path, seg_matrices, cfg)
        
        batch_matrix.append(seg_matrices)
        batch_labels.append(l)

    batch_matrix, batch_labels = utils.pad_batch(b, cfg,  device, batch_matrix, batch_labels)
    batch_matrix = torch.tensor(np.array(batch_matrix), device=device, dtype=torch.float32) 

    # assert(batch_matrix.shape == (b, cfg.experiment.n_segs, cfg.matrix.n_channels,
    #                             int(cfg.matrix.resolution / cfg.experiment.n_segs), 
    #                             cfg.matrix.bins,))
    
    return batch_matrix, batch_labels
