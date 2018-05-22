import numpy as np
from keras.utils import np_utils
import config
import glob

def prepare_sequences():
    """ Prepare the sequences used by the Neural Network """
    sequence_length = config.SEQUENCE_LENGTH

    filenames = []
    for filename in glob.glob("data/*"):
        filenames.append(filename)

    filenames = sorted(filenames)

    notes = []
    for i in range(len(filenames)):
        notes += list(np.load(filename))
    print(notes)

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

prepare_sequences()
