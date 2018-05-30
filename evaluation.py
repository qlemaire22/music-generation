import numpy as np
from keras.utils import np_utils
import config
import glob
import os
import data
from tqdm import tqdm
import copy
import argparse

SONG_PATH = "results/han_china_output0.npy"
HISTO_PATH = "histograms/"
NB_SEMITONES = 12
REAL_NB_NOTES = 52
REAL_NB_NOTES_PLUS = 60


"""def evaluation():
    print("Evaluating: " + SONG_PATH)
    filenames_temp = []
    for filename in glob.glob("data/individual_songs/*"):
        filenames_temp.append(filename)

    filenames = sorted(filenames_temp)

    _, _, n_vocab, pitchnames, _ = data.prepare_sequences()

    # print(pitchnames)

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    # print(note_to_int)

    song = np.load(SONG_PATH)

    print(song)

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

    # print("Closest song: " + filenames[np.argmin(distances)] + " distance: " + str(np.min(distances)))

    # print(song)
    list_notes = list(np.load(filenames[np.argmin(distances)]))
    list_notes = [note_to_int[char] for char in list_notes]
    # print(list_notes)
    # print(distance_debug(song, list_notes))


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
            # temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
        else:
            temp_song = np.append(song1[i1:n1 - 1], song1[:i2 - n1 + 1])
            temp = np.abs(temp_song - song2)
            # temp = np.where(temp > 0, 1, 0)
            temp = np.sum(temp)
            distances.append(temp)
    return min(distances) / n2


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
            temp_song = np.append(song1[i1:n1 - 1], song1[:i2 - n1 + 1])
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
"""


def evaluation(loaddata):
    print("Evaluating: " + SONG_PATH)
    files_path = "data/individual_songs/"

    filenames_temp = []
    for filename in glob.glob(files_path + "*"):
        filenames_temp.append(filename)

    filenames = sorted(filenames_temp)

    filenames_short = []
    for filename in filenames:
        if filename.startswith(files_path):
            filenames_short.append(filename[len(files_path):])
        else:
            filenames_short.append(filename)

    if loaddata == 0:
        if not os.path.exists("histograms"):
            os.makedirs("histograms")
        distances = create_hist_distances(filenames_short, filenames)

    elif loaddata == 1:
        if not (os.path.exists("histograms")):
            print("There is histograms, under histogram directory. Please try computing "
                  "all the histograms first (parameter 1).")
            return
        else:
            distances = load_hist_distances(filenames_short)

    else:
        print("Wrong parameter for whole. Please enter 1 for computing the histogram of the whole dataset,"
              "or 0 for using preexisting data (if any)")
        return

    minimums = np.argsort(distances)

    for i in range(10):
        print("Song position " + str(i) + ": " + filenames[minimums[i]])

    print("Closest song: " + filenames[np.argmin(distances)] + " distance: " + str(np.min(distances)))
    best_histo = np.load(HISTO_PATH + filenames_short[np.argmin(distances)])
    print(best_histo)

    # print(song)
    #list_notes = list(np.load(filenames[np.argmin(distances)]))
    #list_notes = [note_to_int[char] for char in list_notes]
    # print(list_notes)
    # print(distance_debug(song, list_notes))


def load_hist_distances(filenames_short):
    song = np.load(SONG_PATH)
    song_histogram = interval_histogram(pitch_histogram(song), len(song))
    print(song_histogram)
    distances = []
    for i in tqdm(range(len(filenames_short))):
        path = HISTO_PATH + filenames_short[i]
        current_histogram = np.load(path)

        # because of nan in asian file 1028
        if np.isnan(current_histogram).any():
            current_histogram = np.array([0] * NB_SEMITONES)

        if current_histogram.size > 0:
            distances.append(compare_histograms(song_histogram, current_histogram))
        else:
            distances.append(1000)

    return distances


def create_hist_distances(filenames_short, filenames):
    song = np.load(SONG_PATH)
    song_histogram = interval_histogram(pitch_histogram(song), len(song))
    print(song_histogram)
    _, _, _, pitchnames, _ = data.prepare_sequences()
    distances = []

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    for i in tqdm(range(len(filenames))):
        list_notes = list(np.load(filenames[i]))
        list_notes = [note_to_int[char] for char in list_notes]
        current_histogram = interval_histogram(pitch_histogram(list_notes), len(list_notes))
        np.save(HISTO_PATH + filenames_short[i], current_histogram)
        if len(list_notes) > 0:
            distances.append(compare_histograms(song_histogram, current_histogram))
        else:
            distances.append(1000)

    return distances


# inputs : two lists of length 12
# output : sum of differences
# idea to modify the method : add a strong weight for a note which is 0 compared to a non zero note
def compare_histograms(hist1, hist2):
    result = 0
    for i in range(NB_SEMITONES):
        result += abs(hist1[i] - hist2[i])
    return result


def pitch_histogram(song):
    length = len(song)
    my_dict = note_to_mypitch_table()
    histogram = [0] * NB_SEMITONES
    for i in range(length):
        histogram[my_dict[song[i]] % NB_SEMITONES] += 1
    return np.asarray(histogram)


# input should be numpy array and length of whole song
# return normalized list
def interval_histogram(pitch_histo, length):
    result = [0] * NB_SEMITONES
    max_idx = np.argmax(pitch_histo)
    # count + normalize frequency of each note
    # as a relative to the most recurrent note
    for idx in range(NB_SEMITONES):
        result[idx - int(max_idx)] += pitch_histo[idx] / length

    return result


def note_to_mypitch_table():
    _, _, _, pitchnames, _ = data.prepare_sequences()

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    ordered_pitch = {'A2': 0, 'B-2': 1, 'B2': 2, 'C3': 3, 'C#3': 4,
                     'D3': 5, 'D#3': 6, 'E-3': 6, 'E3': 7, 'F3': 8, 'F#3': 9, 'G3': 10, 'G#3': 11,
                     'A-3': 11, 'A3': 12, 'A#3': 13, 'B-3': 13, 'B3': 14, 'C-4': 14, 'B#3': 15, 'C4': 15, 'C#4': 16,
                     'D-4': 16, 'D4': 17, 'D#4': 18, 'E-4': 18, 'E4': 19, 'F-4': 19, 'E#4': 20, 'F4': 20, 'F#4': 21,
                     'G-4': 21, 'G4': 22, 'G#4': 23, 'A-4': 23, 'A4': 24, 'A#4': 25, 'B-4': 25, 'B4': 26, 'C-5': 26,
                     'B#4': 27, 'C5': 27, 'C#5': 28, 'D-5': 28, 'D5': 29, 'D#5': 30, 'E-5': 30, 'E5': 31, 'F-5': 31,
                     'E#5': 32, 'F5': 32, 'F#5': 33, 'G-5': 33, 'G5': 34, 'G#5': 35, 'A-5': 35, 'A5': 36, 'A#5': 37,
                     'B-5': 37, 'B5': 38, 'B#5': 39, 'C6': 39, 'C#6': 40, 'D-6': 40, 'D6': 41, 'D#6': 42, 'E-6': 42,
                     'E6': 43, 'F6': 44, 'F#6': 45, 'G6': 46, 'G#6': 47, 'A6': 48, 'B-6': 49, 'B6': 50, 'C7': 51}

    result = {}
    for note in pitchnames:
        result[note_to_int[note]] = ordered_pitch[note]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--loaddata', default=0,
                        help="0 if you want to recompute the histograms for the whole dataset,"
                             "1 if you want to load pre-existing histograms.", type=int)

    args = parser.parse_args()

    evaluation(args.loaddata)
