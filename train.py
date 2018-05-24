import network
import data
import preprocessing
import numpy as np
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import config

def init():
    """ Train a Neural Network to generate music """

    network_input, network_output, n_vocab, _, _ = data.prepare_sequences()

    net = network.Network(n_vocab)
    model = net.model

    train(model, network_input, network_output)

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=config.NUMBER_EPOCHS, batch_size=config.BATCH_SIZE, callbacks=callbacks_list)

if __name__ == '__main__':
    init()
