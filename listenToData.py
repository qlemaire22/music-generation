import config
from music21 import converter, stream, instrument, note, chord
from tools import listenToMidi
import argparse

FILE_SUBPATH = "europa/misc/norge01.krn"


def KRNtoMIDI(rythm):
    print(config.PATH_DATA_FILES[:-11] + FILE_SUBPATH)

    s = converter.parse(config.PATH_DATA_FILES[:-11] + FILE_SUBPATH)
    s2 = instrument.partitionByInstrument(s)
    notes_to_parse = s2.parts[0].recurse()
    notes = []
    offset = 0
    for current_note in notes_to_parse:
        if isinstance(current_note, note.Note):
            new_note = current_note
            new_note.storedInstrument = instrument.Piano()
            if rythm == 0:
                new_note.offset = offset
            notes.append(new_note)
            offset += 0.5

    midi_stream = stream.Stream(notes)

    midi_stream.write('midi', fp='temp.mid')
    listenToMidi("temp.mid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--rythm', default=1,
                        help="int, 0 if you want all note to last 0.5 time.", type=int)

    args = parser.parse_args()

    KRNtoMIDI(args.rythm)
