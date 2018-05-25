from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from keras import objectives

import numpy as np

import vae_network
import vae_config


def generate(g_input):
    net = vae_network.VAEDeepNetwork3()
    net.load_weights('outputs/vae_weights/vae_weights-0.5627.hdf5')
    #model = net.model
    #model.load_weight('results/run1/weights-1.9942.hdf5')  # need to modify path

    generator = net.generator

    # the input must have shape (latent_dim, ?)
    return generator.predict(g_input)


if __name__ == '__main__':
    g_input = np.random.rand(1, vae_config.LATENT_DIM)
    generate(g_input)
