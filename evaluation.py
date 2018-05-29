import numpy as np
from keras.utils import np_utils
import config
import glob
import os
import data
from tqdm import tqdm
import copy

SONG_PATH = "results/run4/italia_output1.npy"

def evaluation():
    print("Evaluating: " + SONG_PATH)
    filenames_temp = []
    for filename in glob.glob("data/individual_songs/*"):
        filenames_temp.append(filename)

    filenames = sorted(filenames_temp)

    _, _, n_vocab, pitchnames, _ = data.prepare_sequences()

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    song = np.load(SONG_PATH)

    distances = []

    for i in tqdm(range(len(filenames))):
        list_notes = list(np.load(filenames[i]))
        list_notes = [note_to_int[char] for char in list_notes]
        if len(list_notes) > 0:
            distances.append(distance(song, list_notes))
        else:
            distances.append(1000)


    minimums = np.argsort(distances)

    for i in range(10):
        print("Song position " + str(i) + ": " + filenames[minimums[i]])

    print("Closest song: " + filenames[np.argmin(distances)] + " distance: " + str(np.min(distances)))

    print(song)
    list_notes = list(np.load(filenames[np.argmin(distances)]))
    list_notes = [note_to_int[char] for char in list_notes]
    print(list_notes)
    print(distance_debug(song, list_notes))

def distance(song1, song2):
    if len(song2) > len(song1):
        song1_temp = copy.deepcopy(song1)
        song2_temp = copy.deepcopy(song2)
        song1, song2 = song2_temp, song1_temp

    n1 = len(song1)
    n2 = len(song2)

    distances = []

    for i in range(n1):
        i1 = i
        i2 = i + n2

        if i2 < n1:
            temp = np.abs(song1[i1:i2] - song2)
            #temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
        else:
            temp_song = np.append(song1[i1:n1-1], song1[:i2-n1+1])
            temp = np.abs(temp_song - song2)
            #temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
    return min(distances)/n2

def distance_debug(song1, song2):
    if len(song2) > len(song1):
        song1_temp = copy.deepcopy(song1)
        song2_temp = copy.deepcopy(song2)
        song1 = song2_temp
        song2 = song1_temp

    n1 = len(song1)
    n2 = len(song2)

    distances = []

    for i in range(n1):
        i1 = i
        i2 = i + n2

        if i2 < n1:
            temp = np.abs(song1[i1:i2] - song2)
            temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
            if temp == 1:
                print("ok")
                print(song1[i1:i2])
                print(song2)
                print(np.abs(song1[i1:i2] - song2))

        else:
            temp_song = np.append(song1[i1:n1-1], song1[:i2-n1+1])
            temp = np.ceil(np.abs(temp_song - song2))
            temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
            if temp == 1:
                print("ok")
                print(temp_song)
                print(song2)
                print(np.abs(temp_song - song2))

    return min(distances)

if __name__ == '__main__':
    evaluation()
