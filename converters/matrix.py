import torch
import numpy as np
import pretty_midi
import partitura as pt
from einops import rearrange, repeat
import pandas as pd
import muspy


def perfmidi_to_matrix(path, cfg):
    """Process MIDI events to roll matrices for training
    Adapted from https://github.com/bytedance/piano_transcription/blob/master/utils/utilities.py
    """
    perf_data = pretty_midi.PrettyMIDI(path)

    note_events = perf_data.instruments[0].notes
    pedal_events = perf_data.instruments[0].control_changes

    frames_num = cfg.matrix.resolution
    frames_per_second = (cfg.matrix.resolution / perf_data.get_end_time())
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    velocity_roll = np.zeros((frames_num, cfg.matrix.bins))

    for note_event in note_events:
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""

        bgn_frame = min(int(round((note_event.start) * frames_per_second)), frames_num-1)
        fin_frame = min(int(round((note_event.end) * frames_per_second)), frames_num-1)
        velocity_roll[bgn_frame : fin_frame + 1, note_event.pitch] = note_event.velocity
        onset_roll[bgn_frame, note_event.pitch] = 1

    for pedal_event in pedal_events:
        """pedal_event: e.g., ControlChange(number=67, value=111, time=5.492188)"""

        if pedal_event.number == 64: ped_index = 128
        elif pedal_event.number == 66: ped_index = 129
        elif pedal_event.number == 67: ped_index = 130
        else: continue

        bgn_frame = min(int(round((pedal_event.time) * frames_per_second)), frames_num-1)
        velocity_roll[bgn_frame : , ped_index] = pedal_event.value
        onset_roll[bgn_frame, ped_index] = 1

    """Clip rolls into segments and concatenate"""
    onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    velocity_roll = rearrange(velocity_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    matrices = np.stack([onset_roll, velocity_roll], axis=1)

    return matrices # (s 2 h w)


def musicxml_to_matrix(path, cfg):
    """Process musicXML to roll matrices for training"""

    import warnings
    warnings.filterwarnings("ignore") # mute partitura warnings
    score_data = pt.load_musicxml(path)

    note_events = pd.DataFrame(score_data.note_array(), columns=score_data.note_array().dtype.names)

    end_time_divs = note_events['onset_div'].max() + note_events['duration_div'].max()
    frames_num = cfg.matrix.resolution
    frames_per_second = (frames_num / end_time_divs)
    onset_roll = np.zeros((frames_num, cfg.matrix.bins))
    voice_roll = np.zeros((frames_num, cfg.matrix.bins))

    for _, note_event in note_events.iterrows():
        """note_event: e.g., Note(start=1.009115, end=1.066406, pitch=40, velocity=93)"""

        bgn_frame = min(int(round((note_event['onset_div']) * frames_per_second)), frames_num-1)
        fin_frame = min(bgn_frame + int(round((note_event['duration_div']) * frames_per_second)), frames_num-1)
        voice_roll[bgn_frame : fin_frame + 1, note_event['pitch']] = note_event['voice']
        onset_roll[bgn_frame, note_event.pitch] = 1


    """Clip rolls into segments and concatenate"""
    onset_roll = rearrange(onset_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    voice_roll = rearrange(voice_roll, "(s f) n -> s f n", s=cfg.experiment.n_segs)
    matrices = np.stack([onset_roll, voice_roll], axis=1)

    return matrices # (s 2 h w)


def batch_to_matrix(batch, cfg, device):
    """Map the batch to input piano roll matrices
    TODO: try onsets+frames 3d matrix, and try the pedal matrix as well

    Args:
        batch (2, b): ([path, path, ...], [label, label, ...])
    Returns: (matrix, label)
        matrix: (b, n_segs, pitch_bins, resolution)
        label: (b, )
    """
    files, labels = batch
    batch_matrix = []
    for path, _ in zip(files, labels):
        if cfg.experiment.input_format == "perfmidi":
            seg_matrices = perfmidi_to_matrix(path, cfg)
        elif cfg.experiment.input_format == "musicxml":
            seg_matrices = musicxml_to_matrix(path, cfg)
        elif cfg.experiment.input_format == "kern":
            seg_matrices = kern_to_matrix(path, cfg)
        batch_matrix.append(seg_matrices)

    batch_matrix = torch.tensor(np.array(batch_matrix), device=device, dtype=torch.float32) 
    return batch_matrix, labels
