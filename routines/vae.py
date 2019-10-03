from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, UpSampling2D, Activation, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model, model_from_json
from keras.losses import mse
import numpy as np
from keras import backend as K
import tensorflow as tf
import os

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE():
    def __init__(self,X_train):
        print("Initialize the autoencoder...")
        self.inputs = X_train
        self.latent_dim = 4

    def create_model(self):
        state_input = Input(shape=(8,))
        hidden_layer = Dense(6, activation='relu')(state_input)
        z_mean = Dense(self.latent_dim, name='z_mean')(hidden_layer)
        z_log_var = Dense(self.latent_dim, name='z_log')(hidden_layer)
        
        z = Lambda(sampling, output_shape=(self.latent_dim, ), name='z')([z_mean, z_log_var]) 

        encoder = Model(state_input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(6, activation='relu')(latent_inputs)
        decoded = Dense(8, activation='tanh')(x)
        
        decoder = Model(latent_inputs, decoded)
        decoder.summary()
        
        outputs = decoder(encoder(state_input)[2])
        vae = Model(state_input,outputs)
        vae.summary()

        reconstruction_loss = mse(K.flatten(state_input), K.flatten(outputs))
        reconstruction_loss *= 8
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        return vae


    def fit(self):
        self.model = self.create_model()
        self.model.fit(self.inputs, epochs=200, batch_size=256, shuffle=True)

        return self.model
    
    def save_model(self, path):
        self.model.save_weights(os.path.join(path, "vae_weights.h5"))
        with open(os.path.join(path, 'vae_architecture.json'),'w') as f:
            f.write(self.model.to_json())