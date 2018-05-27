import numpy as np
import vae_config

DATA_FILE = "outputs/states/han_china_states.npy"

def prepare_sequences():
    x_train = np.load(DATA_FILE)
    return x_train, x_train
