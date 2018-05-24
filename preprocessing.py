from tqdm import tqdm
import glob
from music21 import converter, instrument, note, chord
import pickle
import config
import numpy as np
import argparse
import os

def create_data(nb_max):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """

    nb = 0

    if not os.path.exists("data"):
        print("Data folder created.")
        os.makedirs("data")

    filenames = []
    for filename in tqdm(glob.iglob(config.PATH_DATA_FILES, recursive=True)):
        filenames.append(filename)

    filenames = np.unique(filenames)



    for filename in tqdm(filenames):
        midi = converter.parse(filename)

        notes_to_parse = None

        if nb == nb_max:
            break

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        notes = []

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        np.save("data/" + str(nb) + file.replace(config.PATH_DATA_FILES[:-11], '').replace("/", '-').replace(".krn", '.npy'), notes)

        nb += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max', default=-1,
                        help="int, set a number to fix a max size for the dataset.", type=int)

    args = parser.parse_args()

    create_data(args.max)
