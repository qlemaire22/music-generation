import network
import data
import preprocessing
import numpy as np
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import config

def init():
    """ Train a Neural Network to generate music """

    network_input, network_output, n_vocab = data.prepare_sequences()

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

    for i in tqdm(range(config.NUMBER_EPOCHS)):
        j = -1
        ind = []
        while j < network_input.shape[0]-1:
            j += np.random.randint(1, 50)
            if j >=network_input.shape[0]:
                j = network_input.shape[0]-1
            ind.append(j)

        X = network_input[ind]
        Y = network_output[ind]
        model.fit(X, Y, epochs=1, batch_size=config.BATCH_SIZE, callbacks=callbacks_list)

if __name__ == '__main__':
    init()
