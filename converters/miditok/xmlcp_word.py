""" Self-defined MusicXML tokenization class, inherited from the miditok base class.
Usage: put this file under the miditok package. Also add this class into miditok __init__.py
"""

from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict

import numpy as np
from miditoolkit import Instrument, Note, TempoChange
import partitura as pt

from .midi_tokenizer_base import MIDITokenizer, Vocabulary, Event
from .constants import *


class XMLCPWord(MIDITokenizer):
    """ MusicXML encoding - REMILike

    This strategy is similar to the REMI midi encoding, with measure boundries and relative position in measure.
    The token types are 
        Signatures: (KeySig: 14*2=28 + TimeSig: 6*5=30) 
        Bar, Repeat: 2
        Position: 33
        Note: 169 (Note: 7*8*3=168 + Rest: 1)
        Voice: 12
        Duration: 10
        Markings:  (Dynamics: 9 + Articulation: 2) 

    (feat_level is designed to control what information to put in, if we want a fair comparison with midi)
    Feat_level 0

    """

    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 sos_eos_tokens: bool = False, mask: bool = False, feat_level=0, params=None
                 ):
        
        self.pitches = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        self.alters = ["#", "b", ""]

        self.positions = np.linspace(0, 1, 33) 
        self.octaves = list(range(0, 9))
        self.full_note_names = [f"{p}{a}" for p in self.pitches for a in self.alters]
        self.full_note_names_octaves = [f"{p}{o}" for p in self.full_note_names for o in self.octaves]
        self.voices = list(range(20))
        self.symbolic_durations = ['breve', 'whole', 'half', 'quarter', '8th', '16th', '32nd', '64th', '128th', '256th', '512th', 'long']
        self.numerators = [2, 3, 4, 6, 9, 12]
        self.denominators = [2, 4, 8, 12, 16]
        self.dynamics = ['pp', 'p', 'mp', 'mf', 'f', 'ff', 'fp', "sf", 'fz'] # TODO: add time-wise dynamics like crescendo
        self.articulations = ['staccato', 'legato'] # TODO: add legato parsing

        self.feat_level = feat_level

        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, mask, params)


    def track_to_tokens(self, score: pt.score.Score, start_end_beat=None) -> List[int]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens

        :param score: partitura.score.Score
        :return: sequence of corresponding tokens
        """
        tokens = []  # list of lists of tokens

        measure_timepoints = []
        score_voices = []
        for score_part in score.parts:
            for i, mes in enumerate(score_part.measures):
                # events.append(Event(type='Bar', time=mes.start.t, value=None, desc=0))
                tokens.append(
                        self.__create_cp_token(
                            mes.start.t, bar=True, desc="Bar"
                        )
                    )
                measure_timepoints.append(mes.start.t)
            measure_timepoints.append(mes.end.t)
            
            timepoint_async_count = defaultdict(int) # a hack, to give an order to the Position - Note (Rest) - Voice - Duration group. 
            # Position - Note (Rest) - Voice - Duration group
            for _, note in enumerate(score_part.notes_tied + score_part.rests):
                timepoint_async = note.start.t + 0.01 * timepoint_async_count[note.start.t]
                timepoint_async_count[note.start.t] += 1
                # events.append(Event(type='Position', time=timepoint_async, 
                #                     value=self._find_position(note.start.t, measure_timepoints), 
                #                     desc=0))
                tokens.append(
                    self.__create_cp_token(
                        timepoint_async,
                        pos=self._find_position(note.start.t, measure_timepoints),
                        desc="Position",
                    )
                )

                # Duration
                duration_value = None
                try: # some parsing error of note without symbolic duration type?
                    if note.symbolic_duration['type'] == 'eighth':
                        duration_value = '8th'
                    else:
                        duration_value = note.symbolic_duration['type']
                    # events.append(Event(type='Duration', time=timepoint_async, value=duration_value, desc=0))
                except:
                    pass

                voice = None
                if self.feat_level:
                    # Voice
                    if note.voice in score_voices: # reduce the voices to 1, 2, 3...
                        voice = score_voices.index(note.voice)
                    else:
                        voice = len(score_voices)
                        score_voices.append(voice)
                    # events.append(Event(type='Voice', time=timepoint_async, value=voice, desc=f'{note.voice}'))
                
                # Note
                if type(note) == pt.score.Rest:
                    # events.append(Event(type='Rest', time=timepoint_async, value=None, desc=0))
                    tokens.append(
                        self.__create_cp_token(
                            timepoint_async, 
                            rest="Rest_None", 
                            dur=duration_value,
                            desc="Rest"
                        )
                    )
                else:
                    if (note.alter not in range(-2, 2)) or (note.alter_sign not in self.alters): # ignore some of the notes with double sharp (x)
                        continue
                    note_fullname = f"{note.step}{note.alter_sign}{note.octave}"
                    # events.append(Event(type='Note', time=timepoint_async, value=note_fullname, desc=note.end))
                    tokens.append(
                        self.__create_cp_token(
                            timepoint_async,
                            note=note_fullname,
                            voice=voice,
                            dur=duration_value,
                            desc="note",
                        )
                    )
                           
        
            if self.feat_level:
                # Signatures
                for sig in score_part.key_sigs:
                    sig.mode = 'minor' if sig.mode == 'minor'  else 'major'
                    # events.append(Event(type='Key-Sig', time=sig.start.t, value=f"{sig.name}.{sig.mode}", desc=0))
                    tokens.append(
                        self.__create_cp_token(
                            sig.start.t, key_sig=f"{sig.name}.{sig.mode}", desc="key_sig"
                        )
                    )
                for sig in score_part.time_sigs:
                    if (sig.beats in self.numerators) and (sig.beat_type in self.denominators): # less frequent signatures we just omit
                        # events.append(Event(type='Time-Sig', time=sig.start.t, value=f"{sig.beats}.{sig.beat_type}", desc=0))
                        tokens.append(
                            self.__create_cp_token(
                                sig.start.t, time_sig=f"{sig.beats}.{sig.beat_type}", desc="time_sig"
                            )
                        )

                # Articulations
                for dyn in score_part.dynamics:
                    if dyn.text in self.dynamics: # TODO: encorporate things like crescendo later 
                        # events.append(Event(type='Dynamics', time=dyn.start.t, value=dyn.text, desc=0))
                        tokens.append(
                            self.__create_cp_token(
                                dyn.start.t, dynamics=dyn.text, desc="dynamics"
                            )
                        )
                for atc in score_part.articulations:
                    if atc.text in self.articulations:
                        # events.append(Event(type='Articulation', time=atc.start.t, value=atc.text, desc=0))
                        tokens.append(
                            self.__create_cp_token(
                                atc.start.t, articulation=atc.text, desc="articulations"
                            )
                        )                

                for rep in score_part.repeats:
                    # events.append(Event(type='Repeat', time=rep.end.t, value=None, desc=0))
                    tokens.append(
                        self.__create_cp_token(
                            rep.end.t, repeat='Repeat_None', desc="repeat"
                        )
                    ) 

            if start_end_beat != None:
                """segmentation by beats"""
                start, end = start_end_beat
                tokens = [t for t in tokens if (score_part.beat_map(t[0].time) >= start and score_part.beat_map(t[0].time) < end)]

        tokens.sort(key=lambda x: x[0].time)

        # Convert the first element of each compound token from Event to int
        for cp_token in tokens:
            cp_token[0] = self.vocab[0].event_to_token[f"Family_{cp_token[0].value}"]

        return tokens

    def __create_cp_token(
        self,
        time: int,
        bar: str = False,
        pos: str = None,
        note: str = None,
        rest: str = None,
        voice: str = None,
        dur: str = None,
        repeat: str = None,
        time_sig: str = None,
        key_sig: str = None,
        dynamics: str = None,
        articulation: str = None,
        desc: str = "",
    ) -> List[Union[Event, int]]:
        r"""Create a CP Word token, with the following structure:
            (index. Token type)
            0. Family
            1. Bar/Position
            2. Note/Rest
            3. Voice
            4. Duration
            (5. Repeat) optional, 
            (6. Time-Sig) optional, f"Time-Sig_{numer}.{denom}"
            (7. Key-Sig) optional, f"Key-Sig_{tonic}.{mode}"
            (8. Dynamics) optional, f"Dynamics_{dyn}"
            (9. Articulation) optional, f"Articulation_{atc}"

        NOTE: the first Family token (first in list) will be given as an Event object to keep track
        of time easily so that other method can sort CP tokens afterwards.

        :return: The compound token as a list of integers
        """
        cp_token_template = [
            Event(type="Family", value="Metric", time=time, desc=desc),
            self.vocab[1].event_to_token["Ignore_None"],
            self.vocab[2].event_to_token["Ignore_None"],
            self.vocab[3].event_to_token["Ignore_None"],
            self.vocab[4].event_to_token["Ignore_None"],
            self.vocab[5].event_to_token["Ignore_None"],
        ]

        if bar:
            cp_token_template[1] = self.vocab[1].event_to_token["Bar_None"]
        elif pos is not None:
            cp_token_template[1] = self.vocab[1].event_to_token[f"Position_{pos}"]
        elif rest is not None:
            cp_token_template[2] = self.vocab[2].event_to_token["Rest_None"]
            if dur:
                cp_token_template[4] = self.vocab[4].event_to_token[f"Duration_{dur}"]
        elif note is not None:
            cp_token_template[0].value = "Note"
            cp_token_template[2] = self.vocab[2].event_to_token[f"Note_{note}"]
            if voice: # no voice info if feature level control
                cp_token_template[3] = self.vocab[3].event_to_token[f"Voice_{voice}"]
            cp_token_template[4] = self.vocab[4].event_to_token[f"Duration_{dur}"]
        elif repeat is not None:
            cp_token_template[5] = self.vocab[5].event_to_token["Repeat_None"]
        elif time_sig is not None:
            cp_token_template[5] = self.vocab[5].event_to_token[f"Time-Sig_{time_sig}"]
        elif key_sig is not None:
            cp_token_template[5] = self.vocab[5].event_to_token[f"Key-Sig_{key_sig}"]
        elif dynamics is not None:
            cp_token_template[5] = self.vocab[5].event_to_token[f"Dynamics_{dynamics}"]
        elif articulation is not None:
            cp_token_template[5] = self.vocab[5].event_to_token[f"Articulation_{articulation}"]

        return cp_token_template


    def tokens_to_track(self, tokens: List[int], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False), default_duration: int = None) \
            -> Tuple[Instrument, List[TempoChange]]:
        """ 
        ************* NOT IMPLEMENTED ************************
        Converts a sequence of tokens into a track object

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :param default_duration: default duration (in ticks) in case a Note On event occurs without its associated
                                note off event. Leave None to discard Note On with no Note Off event.
        :return: the miditoolkit instrument object and tempo changes
        """
        ticks_per_sample = time_division // max(self.beat_res.values())
        events = self.tokens_to_events(tokens)

        max_duration = self.durations[-1][0] * time_division + self.durations[-1][1] * (time_division //
                                                                                        self.durations[-1][2])
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(TEMPO, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        while ei < len(events):
            if events[ei].type == 'Note-On':
                try:
                    if events[ei + 1].type == 'Velocity':
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        duration = 0
                        for i in range(ei + 1, len(events)):
                            if events[i].type == 'Note-Off' and int(events[i].value) == pitch:
                                duration = offset_tick
                                break
                            elif events[i].type == 'Time-Shift':
                                offset_tick += self._token_duration_to_ticks(events[i].value, time_division)
                            elif events[ei].type == 'Rest':
                                beat, pos = map(int, events[ei].value.split('.'))
                                offset_tick += beat * time_division + pos * ticks_per_sample
                            if offset_tick > max_duration:  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration != 0:
                            instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        ei += 1
                except IndexError as _:
                    pass
            elif events[ei].type == 'Time-Shift':
                current_tick += self._token_duration_to_ticks(events[ei].value, time_division)
            elif events[ei].type == 'Rest':
                beat, pos = map(int, events[ei].value.split('.'))
                current_tick += beat * time_division + pos * ticks_per_sample
            elif events[ei].type == 'Tempo':
                tempo = int(events[ei].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_vocabulary(self, sos_eos_tokens: bool = None) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: DEPRECIATED, will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :return: the vocabulary object
        """
        if sos_eos_tokens is not None:
            print(f'\033[93msos_eos_tokens argument is depreciated and will be removed in a future update, '
                  f'_create_vocabulary now uses self._sos_eos attribute set a class init \033[0m')
        # vocab = Vocabulary({'PAD_None': 0}, sos_eos=self._sos_eos, mask=self._mask)

        vocab = [
            Vocabulary(
                pad=True, sos_eos=self._sos_eos, mask=self._mask, sep=self._sep
            )
            for _ in range(6)
        ]
        for v in vocab[1:]:
            v.add_event("Ignore_None")

        vocab[0].add_event("Family_Metric")
        vocab[0].add_event("Family_Note")

        # MEASURE and POSITION
        vocab[1].add_event('Bar_None')
        vocab[1].add_event(f'Position_{pos}' for pos in self.positions)

        # NOTE and REST
        vocab[2].add_event(f'Note_{i}' for i in self.full_note_names_octaves)
        vocab[2].add_event('Rest_None')

        # VOICE
        vocab[3].add_event(f'Voice_{i}' for i in self.voices)

        # DURATION - symbolic duration
        vocab[4].add_event(f'Duration_{i}' for i in self.symbolic_durations)

        # REPEATS
        vocab[5].add_event("Repeat_None")
        # SIGNATURES - time and key
        vocab[5].add_event(f"Time-Sig_{numer}.{denom}" for numer in self.numerators for denom in self.denominators)
        vocab[5].add_event(f"Key-Sig_{tonic}.{mode}" for tonic in self.full_note_names for mode in ['major', 'minor'])
        # MARKINGS - dynamic and articulation
        vocab[5].add_event(f"Dynamics_{dyn}" for dyn in self.dynamics)
        vocab[5].add_event(f"Articulation_{atc}" for atc in self.articulations)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        """ 
        ************* NOT IMPLEMENTED ************************
        Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.

        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['Note-On'] = ['Velocity']
        dic['Velocity'] = ['Note-On', 'Time-Shift']
        dic['Time-Shift'] = ['Note-Off', 'Note-On']
        dic['Note-Off'] = ['Note-Off', 'Note-On', 'Time-Shift']

        if self.additional_tokens['Chord']:
            dic['Chord'] = ['Note-On']
            dic['Time-Shift'] += ['Chord']
            dic['Note-Off'] += ['Chord']

        if self.additional_tokens['Tempo']:
            dic['Time-Shift'] += ['Tempo']
            dic['Tempo'] = ['Note-On', 'Time-Shift']
            if self.additional_tokens['Chord']:
                dic['Tempo'] += ['Chord']

        if self.additional_tokens['Rest']:
            dic['Rest'] = ['Rest', 'Note-On', 'Time-Shift']
            if self.additional_tokens['Chord']:
                dic['Rest'] += ['Chord']
            dic['Note-Off'] += ['Rest']

        # self._add_pad_type_to_graph(dic)
        self._add_special_tokens_to_types_graph(dic)
        return dic

    def token_types_errors(self, tokens: List[int], consider_pad: bool = False) -> float:
        """ 
        ************* NOT IMPLEMENTED ************************
        Checks if a sequence of tokens is constituted of good token types
        successions and returns the error ratio (lower is better).
        The Pitch and Position values are also analyzed:
            - a Note-On token should not be present if the same pitch is already being played
            - a Note-Off token should not be present the note is not being played

        :param tokens: sequence of tokens to check
        :param consider_pad: if True will continue the error detection after the first PAD token (default: False)
        :return: the error ratio (lower is better)
        """
        err = 0
        current_pitches = []
        max_duration = self.durations[-1][0] * max(self.beat_res.values())
        max_duration += self.durations[-1][1] * (max(self.beat_res.values()) // self.durations[-1][2])

        events = self.tokens_to_events(tokens)

        for i in range(1, len(events)):
            # Good token type
            if events[i].type in self.tokens_types_graph[events[i - 1].type]:
                if events[i].type == 'Note-On':
                    if int(events[i].value) in current_pitches:
                        err += 1  # pitch already being played
                        continue

                    current_pitches.append(int(events[i].value))
                    # look for an associated note off event to get duration
                    offset_sample = 0
                    for j in range(i + 1, len(events)):
                        if events[j].type == 'Note-Off' and int(events[j].value) == int(events[i].value):
                            break  # all good
                        elif events[j].type == 'Time-Shift':
                            offset_sample += self._token_duration_to_ticks(events[j].value, max(self.beat_res.values()))

                        if offset_sample > max_duration:  # will not look for Note Off beyond
                            err += 1
                            break
                elif events[i].type == 'Note-Off':
                    if int(events[i].value) not in current_pitches:
                        err += 1  # this pitch wasn't being played
                    else:
                        current_pitches.remove(int(events[i].value))
                elif not consider_pad and events[i].type == 'PAD':
                    break

            # Bad token type
            else:
                err += 1

        return err / len(tokens)

    def _find_position(self, t:int, measure_timepoints: list):
        measure = max(i for i in measure_timepoints if i <= t)
        next_measure = measure_timepoints[measure_timepoints.index(measure) + 1]
        position_fraction = (t - measure) / (next_measure - measure)

        position = self.positions[np.absolute(self.positions - position_fraction).argmin()]
        
        return position

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order

        :param x: event to get order index
        :return: an order int
        """
        if x.type == 'Key-Sig' or x.type == 'Time-Sig':
            return -1
        elif x.type == 'Bar':
            return 0
        elif x.type == 'Position':
            return 1
        elif x.type == 'Note' or x.type == 'Rest':
            return 2
        elif x.type == 'Voice':
            return 3
        elif x.type == "Duration":
            return 4
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 5


