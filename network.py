from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
import numpy as np

class Network:
    def __init__(self, n_vocab, shape=(None, 1)):
        self.model = Sequential()
        self.model.add(LSTM(
            512,
            input_shape=shape,
            return_sequences=True
        ))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(512))
        self.model.add(Dense(256))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(n_vocab))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)
