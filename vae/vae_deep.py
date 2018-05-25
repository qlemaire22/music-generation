from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from keras import objectives

import numpy as np

# some parameters
batch_size = 32
original_dim = 512  # vector size 512*6=3072
inter_dim1 = int(original_dim/4)
inter_dim2 = int(inter_dim1/4)
latent_dim = 8
epsilon_std = 1
epochs = 100
nb_vect = 704
vect_leng = original_dim


# needed functions
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)  # changed std to stddev because of error
    return z_mean + K.exp(z_log_sigma) * epsilon


# custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss


# encoder network, mapping inputs to our latent distribution parameters
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(inter_dim1, activation='relu')(x)
i = Dense(inter_dim2, activation='relu')(h)  # activation relu ???
z_mean = Dense(latent_dim)(i)
z_log_sigma = Dense(latent_dim)(i)


# sample new similar points from the latent space
z = Lambda(sampling)([z_mean, z_log_sigma])

# map the sampled latent points back to reconstructed inputs
decoder_i = Dense(inter_dim2, activation='relu')
decoder_h = Dense(inter_dim1, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
i_decoded = decoder_i(z)
h_decoded = decoder_h(i_decoded)
x_decoded_mean = decoder_mean(h_decoded)

# Models
# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_i_decoded = decoder_i(decoder_input)
_h_decoded = decoder_h(_i_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# we train the model using the end-to-end model
vae.compile(optimizer='rmsprop', loss=vae_loss)

# input to the network : random vectors
x_train1 = np.random.rand(nb_vect, vect_leng) / 2 + 0.5
x_test1 = np.random.rand(nb_vect, vect_leng) / 2 + 0.5

def custom_rand_vec():
    array_train = []
    for l in range(nb_vect):
        vec = []
        for j in range(int(vect_leng/2)):
            vec.append(np.random.random())
        for j in range(int(vect_leng/2)):
            vec.append(np.random.random() / 4 + 0.75)
        array_train.append(vec)
    return array_train


x_train2 = np.asarray(custom_rand_vec())
x_test2 = np.asarray(custom_rand_vec())

for vect in x_train2:
    vect[0] = 0
    vect[-1] = 0
for vect in x_test2:
    vect[0] = 0
    vect[-1] = 0

print(x_train2)
print(x_test2)

vae.fit(x_train2, x_train2,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test2, x_test2))

#print(z_mean)

# prediction
input = np.random.rand(1, latent_dim)
print(input)
generated_vect = generator.predict(input)
print(generated_vect)
