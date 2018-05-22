from tqdm import tqdm
import glob
from music21 import converter, instrument, note, chord
import pickle
import config

def create_data():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in tqdm(glob.iglob(config.PATH_DATA_FILES, recursive=True)):

        midi = converter.parse(file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        ite += 1

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    print("lennote", len(notes))
    return notes
