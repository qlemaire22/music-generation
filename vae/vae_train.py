import vae_network
import vae_config
import vae_data
import os
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
import numpy as np


def init():
    """ Train a Neural Network to generate music """

    network_input, network_output = vae_data.prepare_sequences()

    net = vae_network.VAE()
    model = net.model

    if not os.path.exists("outputs/vae_weights"):
        os.makedirs("outputs/vae_weights")

    train(model, network_input, network_output)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "outputs/vae_weights/vae_weights.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    csv_logger = CSVLogger('outputs/vae_train_log.csv',
                           append=True, separator=';')

    callbacks_list = [checkpoint, csv_logger]
    print(np.max(network_input))
    print(np.min(network_input))
    print(np.mean(network_input))

    model.fit(network_input, network_output, shuffle=True, epochs=vae_config.NUMBER_EPOCHS,
              batch_size=vae_config.BATCH_SIZE, callbacks=callbacks_list, validation_split=0.1)


if __name__ == '__main__':
    init()
