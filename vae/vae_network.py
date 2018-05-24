from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Input, Dense
from keras.models import Model

import vae_config

class VAENetwork :
    BATCH_SIZE = 32
    ORIGINAL_DIM = 100
    INTER_DIM = 50
    LATENT_DIM = 4
    EPSILON_STD = 1.0
    NUMBER_EPOCHS = 100
    NB_VECT = BATCH_SIZE * 22  # 704

    def __init__(self):

        # Layers of the basic VAE
        self.decoder_h = Dense(vae_config.INTER_DIM, activation='relu')
        self.decoder_mean = Dense(vae_config., activation='sigmoid')

        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        # sample new similar points from the latent space
        z = Lambda(self.sampling)([z_mean, z_log_sigma])

        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # Encoder part of the model
        self.encoder = Model(x, z_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)  # changed std to stddev because of error
        return z_mean + K.exp(z_log_sigma) * epsilon


""""# encoder network, mapping inputs to our latent distribution parameters
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

# sample new similar points from the latent space
z = Lambda(sampling)([z_mean, z_log_sigma])

# map the sampled latent points back to reconstructed inputs
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Models
# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)"""
