import numpy as np

import argparse

try:
    import vae_network
    import vae_config
except:
    import vae.vae_network as vae_network
    import vae.vae_config as vae_config


def generate():
    g_input = np.random.rand(1, vae_config.LATENT_DIM)

    net = vae_network.VAE()
    net.load_weights('outputs/a_i_a_s/vae_weights/vae_weights.hdf5')

    generator = net.generator

    y = generator.predict(g_input)

    return y


if __name__ == '__main__':

    result = generate()

    parser = argparse.ArgumentParser()

    parser.add_argument('--print', default=0,
                        help="1 if you want to display the result", type=int)

    args = parser.parse_args()

    if args.print == 1:
        print("Generated vector: " + repr(result))
