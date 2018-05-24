from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Input, Dense
from keras.models import Model

import vae_config

class VAENetwork :

    def __init__(self):

        # Layers of the basic VAE
        self.decoder_h = Dense(vae_config.INTER_DIM, activation='relu')
        self.decoder_mean = Dense(vae_config., activation='sigmoid')

        x = Input(batch_shape=(vae_config.BATCH_SIZE, vae_config.ORIGINAL_DIM))
        h = Dense(vae_config.INTER_DIM, activation='relu')(x)
        z_mean = Dense(vae_config.LATENT_DIM)(h)
        z_log_sigma = Dense(vae_config.LATENT_DIM)(h)

        # sample new similar points from the latent space
        z = Lambda(self.sampling)([z_mean, z_log_sigma])

        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # Encoder part of the model
        self.encoder = Model(x, z_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss([z_mean, z_log_sigma]))

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(args, x, x_decoded_mean):
        z_mean, z_log_sigma = args
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(vae_config.BATCH_SIZE, vae_config.LATENT_DIM),
                                  mean=0., stddev=vae_config.EPSILON_STD)  # changed std to stddev because of error
        return z_mean + K.exp(z_log_sigma) * epsilon

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)
