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
original_dim = 100  # vector size
intermediate_dim = 50
latent_dim = 4
epsilon_std = 1
epochs = 100
nb_vect = 704
vect_leng = 100


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
generator = Model(decoder_input, _x_decoded_mean)

# we train the model using the end-to-end model
vae.compile(optimizer='rmsprop', loss=vae_loss)

# input to the network : random vectors
x_train = np.random.rand(nb_vect, vect_leng)
x_test = np.random.rand(nb_vect, vect_leng)

for vect in x_train:
    vect[vect_leng-1] = vect[0]
for vect in x_test:
    vect[vect_leng-1] = vect[0]

vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# prediction
input = np.random.rand(1, latent_dim)
generated_vect = generator.predict(input)
print(generated_vect)
