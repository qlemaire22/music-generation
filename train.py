import network
import data
import preprocessing
import numpy as np
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import config
from keras.callbacks import CSVLogger
import os

def init():
    """ Train a Neural Network to generate music """

    network_input, network_output, n_vocab, _, _ = data.prepare_sequences()

    net = network.Network(n_vocab)
    model = net.model

    if not os.path.exists("outputs"):
        os.makedirs("outputs/")
        os.makedirs("outputs/weights")

    train(model, network_input, network_output)

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "outputs/weights/weights-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    csv_logger = CSVLogger('outputs/train_log.csv', append=True, separator=';')

    callbacks_list = [checkpoint, csv_logger]

    model.fit(network_input, network_output, epochs=config.NUMBER_EPOCHS, batch_size=config.BATCH_SIZE, callbacks=callbacks_list, validation_split=0.8)


if __name__ == '__main__':
    init()
