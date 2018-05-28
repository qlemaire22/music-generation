import numpy as np
try:
    import vae_config
except:
    import vae.vae_config as vae_config

DATA_FILE = "outputs/states/arabic_india_ashsham_suomi_states.npy"

def prepare_sequences():
    x_train = np.load(DATA_FILE)
    print("Data loaded: " + DATA_FILE)
    return x_train, x_train
