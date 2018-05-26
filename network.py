from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model
import numpy as np
import keras.backend as K

class Network:
    def __init__(self, n_vocab):
        self.model = Sequential()
        self.model.add(LSTM(
            512,
            input_shape=(None, 1),
            return_sequences=True,
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


class NetworkGetStates:
    def __init__(self, n_vocab):

        input = Input(shape=(None, 1), dtype='float32', name='input')

        lstm1_out, lstm1_state_h, lstm1_state_c = LSTM(512, return_sequences=True, return_state=True)(input)

        lstm2_out, lstm2_state_h, lstm2_state_c = LSTM(512, return_sequences=True, return_state=True)(lstm1_out)

        lstm3_out, lstm3_state_h, lstm3_state_c = LSTM(512, return_state=True)(lstm2_out)

        dense1_out = Dense(256)(lstm3_out)

        dense2_out = Dense(n_vocab)(dense1_out)

        output = Activation("softmax")(dense2_out)

        self.model = Model(inputs=[input], outputs=[output, lstm1_state_h, lstm1_state_c, lstm2_state_h, lstm2_state_c, lstm3_state_h, lstm3_state_c])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)

class NetworkWithInit:
    def __init__(self, n_vocab, h1, c1, h2, c2, h3, c3):

        input = Input(shape=(None, 1), dtype='float32', name='input')

        lstm1_out = LSTM(512, return_sequences=True)(input)

        lstm2_out = LSTM(512, return_sequences=True)(lstm1_out)

        lstm3_out = LSTM(512)(lstm2_out)

        dense1_out = Dense(256)(lstm3_out)

        dense2_out = Dense(n_vocab)(dense1_out)

        output = Activation("softmax")(dense2_out)

        self.model = Model(inputs=[input], outputs=[output])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.model.layers[1].states[0] = K.variable(value=h1)
        self.model.layers[1].states[1] = K.variable(value=c1)

        self.model.layers[2].states[0] = K.variable(value=h2)
        self.model.layers[2].states[1] = K.variable(value=c2)

        self.model.layers[3].states[0] = K.variable(value=h3)
        self.model.layers[3].states[1] = K.variable(value=c3)

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)
