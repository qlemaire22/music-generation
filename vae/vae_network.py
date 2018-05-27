from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Input, Dense
from keras.models import Model

import vae_config


class VAE:
    def __init__(self):
        # Layers of the basic VAE
        dim1 = vae_config.INTER_DIM1
        dim2 = vae_config.INTER_DIM2
        dim3 = vae_config.INTER_DIM3
        dim4 = vae_config.INTER_DIM4
        dim5 = vae_config.INTER_DIM5
        origdim = vae_config.ORIGINAL_DIM
        latent_dim = vae_config.LATENT_DIM

        decoder_i1 = Dense(dim1, activation='relu')  # layer size 768
        decoder_i2 = Dense(dim2, activation='relu')  # layer size 192
        decoder_i3 = Dense(dim3, activation='relu')  # layer size 48
        decoder_i4 = Dense(dim4, activation='relu')  # layer size 48
        decoder_i5 = Dense(dim5, activation='relu')  # layer size 48
        decoder_mean = Dense(origdim)

        x = Input(shape=(vae_config.ORIGINAL_DIM,))
        i1 = Dense(dim1, activation='relu')(x)
        i2 = Dense(dim2, activation='relu')(i1)
        i3 = Dense(dim3, activation='relu')(i2)
        i4 = Dense(dim4, activation='relu')(i3)
        i5 = Dense(dim5, activation='relu')(i4)

        self.z_mean = Dense(latent_dim)(i5)
        self.z_log_sigma = Dense(latent_dim)(i5)

        # sample new similar points from the latent space
        z = Lambda(self.sampling, output_shape=(latent_dim,),
                   name='z')([self.z_mean, self.z_log_sigma])

        i5_decoded = decoder_i5(z)
        i4_decoded = decoder_i4(i5_decoded)
        i3_decoded = decoder_i3(i4_decoded)
        i2_decoded = decoder_i2(i3_decoded)
        i1_decoded = decoder_i1(i2_decoded)
        x_decoded_mean = decoder_mean(i1_decoded)

        # Encoder part of the model
        self.encoder = Model(x, self.z_mean)

        # Decoder/Generator part of the model
        decoder_input = Input(shape=(latent_dim,))
        _i5_decoded = decoder_i5(decoder_input)
        _i4_decoded = decoder_i4(_i5_decoded)
        _i3_decoded = decoder_i3(_i4_decoded)
        _i2_decoded = decoder_i2(_i3_decoded)
        _i1_decoded = decoder_i1(_i2_decoded)
        _x_decoded_mean = decoder_mean(_i1_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

        self.model = Model(x, x_decoded_mean)
        self.model.compile(optimizer='adam', loss=self.vae_loss)

    # custom loss function : the sum of a reconstruction term, and the KL divergence regularization term
    def vae_loss(self, x, x_decoded_mean):
        reconstruction_loss = objectives.mse(x, x_decoded_mean)
        reconstruction_loss *= vae_config.ORIGINAL_DIM
        kl_loss = 1 + self.z_log_sigma - \
            K.square(self.z_mean) - K.exp(self.z_log_sigma)
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
