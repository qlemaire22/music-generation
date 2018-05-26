import numpy as np
import vae_config


def prepare_sequences():
    x_train = rand_seq2()
    return x_train, x_train


# Random sequence with mean 0.75 and 1st and last values to 0
def rand_seq1():
    x_train = np.random.rand(vae_config.NB_VECT, vae_config.ORIGINAL_DIM) / 2 + 0.5
    return x_train


# Random sequence with
def rand_seq2():
    vect_leng = vae_config.ORIGINAL_DIM
    nb_vect = vae_config.NB_VECT
    array_train = []
    for l in range(nb_vect):
        vec = []
        # First element is 0
        vec.append(0)
        # First half is random between 0 and 1
        for j in range(int(vect_leng / 2)-1):
            vec.append(np.random.random())
        # Second half is centered around 0.75
        for j in range(int(vect_leng / 2)-1):
            vec.append(np.random.random() / 2 + 0.5)
        # Last element is 0
        vec.append(0)
        array_train.append(vec)

    x_train = np.asarray(array_train)
    return x_train
