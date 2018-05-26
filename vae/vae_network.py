from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Input, Dense
from keras.models import Model

import vae_config


# Basic VAE with 1 intermediate layer
class VAENetwork:
    def __init__(self):
        # Layers of the basic VAE
        decoder_h = Dense(vae_config.INTER_DIM, activation='relu')
        decoder_mean = Dense(vae_config.ORIGINAL_DIM, activation='sigmoid')

        x = Input(batch_shape=(vae_config.BATCH_SIZE, vae_config.ORIGINAL_DIM))
        h = Dense(vae_config.INTER_DIM, activation='relu')(x)
        self.z_mean = Dense(vae_config.LATENT_DIM)(h)
        self.z_log_sigma = Dense(vae_config.LATENT_DIM)(h)

        # sample new similar points from the latent space
        z = Lambda(self.sampling)([self.z_mean, self.z_log_sigma])

        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # Encoder part of the model
        self.encoder = Model(x, self.z_mean)

        # Decoder/Generator part of the model
        decoder_input = Input(shape=(vae_config.LATENT_DIM,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(vae_config.BATCH_SIZE, vae_config.LATENT_DIM),
                                  mean=0., stddev=vae_config.EPSILON_STD)  # changed std to stddev because of error
        return z_mean + K.exp(z_log_sigma) * epsilon

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)


# Deep VAE with 3 intermediate layers
class VAEDeepNetwork3:
    def __init__(self):
        # Layers of the basic VAE
        dim1 = vae_config.INTER_DIM1
        dim2 = vae_config.INTER_DIM2
        dim3 = vae_config.INTER_DIM3
        origdim = vae_config.ORIGINAL_DIM
        latdim = vae_config.LATENT_DIM

        decoder_i1 = Dense(dim1, activation='relu')  # layer size 768
        decoder_i2 = Dense(dim2, activation='relu')  # layer size 192
        decoder_i3 = Dense(dim3, activation='relu')  # layer size 48
        decoder_mean = Dense(origdim, activation='sigmoid')

        x = Input(shape=(vae_config.ORIGINAL_DIM,))
        i1 = Dense(dim1, activation='relu')(x)
        i2 = Dense(dim2, activation='relu')(i1)
        i3 = Dense(dim3, activation='relu')(i2)
        self.z_mean = Dense(latdim)(i3)
        self.z_log_sigma = Dense(latdim)(i3)

        # sample new similar points from the latent space
        z = Lambda(self.sampling)([self.z_mean, self.z_log_sigma])

        i3_decoded = decoder_i3(z)
        i2_decoded = decoder_i2(i3_decoded)
        i1_decoded = decoder_i1(i2_decoded)
        x_decoded_mean = decoder_mean(i1_decoded)

        # Encoder part of the model
        self.encoder = Model(x, self.z_mean)

        # Decoder/Generator part of the model
        decoder_input = Input(shape=(latdim,))
        _i3_decoded = decoder_i3(decoder_input)
        _i2_decoded = decoder_i2(_i3_decoded)
        _i1_decoded = decoder_i1(_i2_decoded)
        _x_decoded_mean = decoder_mean(_i1_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(vae_config.LATENT_DIM,),
                                  mean=0., stddev=vae_config.EPSILON_STD)  # changed std to stddev because of error
        return z_mean + K.exp(z_log_sigma) * epsilon

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)
