import numpy as np
from keras.utils import np_utils
import config
import glob
import os
import data
from tqdm import tqdm

SONG_PATH = "results/run4/han_china_output2.npy"

def evaluation():
    filenames_temp = []
    for filename in glob.glob("data/individual_songs/*"):
        filenames_temp.append(filename)

    filenames = sorted(filenames_temp)

    _, _, n_vocab, pitchnames, _ = data.prepare_sequences()

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    song = np.load(SONG_PATH)
    print(song)

    distances = []

    for i in tqdm(range(len(filenames))):
        list_notes = list(np.load(filenames[i]))
        list_notes = [note_to_int[char] for char in list_notes]
        distances.append(distance(song, list_notes))

    print("Closest song: " + filenames[distances.index(min(distances))] + " " + str(min(distances)))
    print(song)
    list_notes = list(np.load(filenames[distances.index(min(distances))]))
    list_notes = [note_to_int[char] for char in list_notes]
    print(list_notes)
    
def distance(song1, song2):
    if len(song2) > len(song1):
        song1, song2 = song2, song1

    n1 = len(song1)
    n2 = len(song2)

    distances = []

    for i in range(n1):
        i1 = i
        i2 = i + n2

        if i2 < n1:
            distances.append(sum(song1[i1:i2] - song2))
        else:
            temp_song = np.append(song1[i1:n1-1], song1[:i2-n1+1])
            distances.append(sum(temp_song - song2))

    return min(distances)

if __name__ == '__main__':
    evaluation()
