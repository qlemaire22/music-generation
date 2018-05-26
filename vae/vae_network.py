from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Input, Dense
from keras.models import Model

import vae_config

# Deep VAE with 3 intermediate layers


class VAEDeepNetwork3:
    def __init__(self):
        # Layers of the basic VAE
        dim1 = vae_config.INTER_DIM1
        dim2 = vae_config.INTER_DIM2
        dim3 = vae_config.INTER_DIM3
        origdim = vae_config.ORIGINAL_DIM
        latent_dim = vae_config.LATENT_DIM

        decoder_i1 = Dense(dim1, activation='tanh')  # layer size 768
        decoder_i2 = Dense(dim2, activation='tanh')  # layer size 192
        decoder_i3 = Dense(dim3, activation='tanh')  # layer size 48
        decoder_mean = Dense(origdim, activation='sigmoid')

        x = Input(shape=(vae_config.ORIGINAL_DIM,))
        i1 = Dense(dim1, activation='tanh')(x)
        i2 = Dense(dim2, activation='tanh')(i1)
        i3 = Dense(dim3, activation='tanh')(i2)

        self.z_mean = Dense(latent_dim)(i3)
        self.z_log_sigma = Dense(latent_dim)(i3)

        # sample new similar points from the latent space
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_sigma])

        i3_decoded = decoder_i3(z)
        i2_decoded = decoder_i2(i3_decoded)
        i1_decoded = decoder_i1(i2_decoded)
        x_decoded_mean = decoder_mean(i1_decoded)

        # Encoder part of the model
        self.encoder = Model(x, self.z_mean)

        # Decoder/Generator part of the model
        decoder_input = Input(shape=(latent_dim,))
        _i3_decoded = decoder_i3(decoder_input)
        _i2_decoded = decoder_i2(_i3_decoded)
        _i1_decoded = decoder_i1(_i2_decoded)
        _x_decoded_mean = decoder_mean(_i1_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='rmsprop', loss=self.vae_loss)

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(self, x, x_decoded_mean):
        reconstruction_loss = objectives.mse(x, x_decoded_mean)
        reconstruction_loss *= vae_config.ORIGINAL_DIM
        kl_loss = 1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def load_weights(self, path):
        self.model.load_weights(path)
        print("Weights loaded:", path)
