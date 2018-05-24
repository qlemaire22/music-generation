import numpy as np
import vae_config

def prepare_sequences():
    x_train = np.random.rand(vae_config.NB_VECT, vae_config.ORIGINAL_DIM) / 2 + 0.5
    return x_train, x_train
