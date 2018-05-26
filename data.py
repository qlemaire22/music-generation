import numpy as np
from keras.utils import np_utils
import config
import glob
import os


def prepare_sequences(conditions = []):
    """ Prepare the sequences used by the Neural Network """

    conditions_string = ""
    n = len(conditions)

    for i in range(n):
        conditions_string += conditions[i]
        if i != n-1:
            conditions_string += "_"

    if conditions_string == "":
        conditions_string = "all"

    if not os.path.exists("data/" + conditions_string):

        print("Create dir data/" + conditions_string + ".")

        sequence_length = config.SEQUENCE_LENGTH

        filenames_temp = []
        for filename in glob.glob("data/*"):
            filenames_temp.append(filename)

        filenames_sorted = sorted(filenames_temp)
        filenames = []

        for filename in filenames_sorted:
            cond = True
            for i in range(n):
                if not(conditions[i] in filename):
                    cond = False

            if cond:
                filenames.append(filename)

        notes = []
        for i in range(len(filenames)):
            notes += list(np.load(filenames[i]))

        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number)
                           for number, note in enumerate(pitchnames))

        unnormalized_network_input = []
        network_output = []


        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            unnormalized_network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(unnormalized_network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(
            unnormalized_network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = np_utils.to_categorical(network_output)

        pitchnames = sorted(set(item for item in notes))

        os.makedirs("data/" + conditions_string)

        np.save("data/" + conditions_string + "/network_input.npy", network_input)
        np.save("data/" + conditions_string + "/network_output.npy", network_output)
        np.save("data/" + conditions_string + "/n_vocab.npy", n_vocab)
        np.save("data/" + conditions_string + "/pitchnames.npy", pitchnames)
        np.save("data/" + conditions_string + "/unormalized.npy", unnormalized_network_input)
    else:
        network_input = np.load("data/" + conditions_string + "/network_input.npy")
        network_output = np.load("data/" + conditions_string + "/network_output.npy")
        n_vocab = int(np.load("data/" + conditions_string + "/n_vocab.npy"))
        pitchnames = np.load("data/" + conditions_string + "/pitchnames.npy")
        unnormalized_network_input = np.load("data/" + conditions_string + "/unormalized.npy")

    return network_input, network_output, n_vocab, pitchnames, unnormalized_network_input
