from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import argparse

import vae_network
import vae_config


def generate(g_input):
    net = vae_network.VAEDeepNetwork3()
    net.load_weights('outputs/vae_weights/vae_weights-0.5348.hdf5')

    generator = net.generator

    return generator.predict(g_input)


if __name__ == '__main__':
    g_input = np.random.rand(1, vae_config.LATENT_DIM)
    result = generate(g_input)

    parser = argparse.ArgumentParser()

    parser.add_argument('--print', default=0,
                        help="1 if you want to display the result", type=int)

    args = parser.parse_args()

    if args.print == 1:
        print("Generated vector: " + repr(result))
