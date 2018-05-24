import vae_network
import vae_config

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from keras import objectives

import numpy as np


def generate(g_input):
    vae = vae_network.VAENetwork()

    decoder_input = Input(shape=(vae.LATENT_DIM,))
    _h_decoded = vae.decoder_h(decoder_input)
    _x_decoded_mean = vae.decoder_mean(_h_decoded)

    generator = Model(decoder_input, _x_decoded_mean)

    # the input must have shape (latent_dim, ?)
    return generator.predict(g_input)


if __name__ == '__main__':
    g_input = np.random.rand(1, vae_config.LATENT_DIM)
    generate(g_input)
