import network
import data
import preprocessing
import numpy as np
from keras.callbacks import ModelCheckpoint


def train_network():
    """ Train a Neural Network to generate music """
    notes = preprocessing.create_data()

    # get amount of pitch names
    n_vocab = len(set(notes))


    network_input, network_output = data.prepare_sequences(notes, n_vocab)


    net = network.Network(n_vocab)
    model = net.model

    train(model, network_input, network_output)

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
