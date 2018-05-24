import vae_network
import vae_config
import os
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint


def init():
    """ Train a Neural Network to generate music """

    network_input, network_output = vae_data.prepare_sequences()

    net = vae_network.VAENetwork()
    model = net.model

    if not os.path.exists("outputs"):
        os.makedirs("outputs/")
        os.makedirs("outputs/weights")

    train(model, network_input, network_output)


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "outputs/weights/weights-{loss:.4f}.hdf5"  # need to modify path
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    csv_logger = CSVLogger('outputs/train_log.csv', append=True, separator=';')

    callbacks_list = [checkpoint, csv_logger]

    model.fit(network_input, network_output, shuffle=True, epochs=vae_config.NUMBER_EPOCHS,
              batch_size=vae_config.BATCH_SIZE, callbacks=callbacks_list, validation_split=0.8)

if __name__ == '__main__':
    init()