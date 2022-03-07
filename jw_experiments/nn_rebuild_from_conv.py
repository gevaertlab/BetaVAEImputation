import os
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class NetworkBuilder():
    def __init__(self, latent_dim=2, input_shape=(28, 28, 1), network_architecture=None):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.network_architecture = network_architecture

    def create_encoder(self):
        n_hidden_recog_1 = self.network_architecture['n_hidden_recog_1']
        n_hidden_recog_2 = self.network_architecture['n_hidden_recog_2']
        encoder_inputs = keras.Input(shape=self.input_shape)
        x = layers.Dense(units=n_hidden_recog_1, activation="relu")(encoder_inputs)
        x = layers.Dense(units=n_hidden_recog_2)(x)
        # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def create_decoder(self):
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        n_hidden_gener_2 = self.network_architecture['n_hidden_gener_1']
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(n_hidden_gener_1, activation="relu")(latent_inputs)
        x = layers.Dense(n_hidden_gener_2, activation="relu")(x)
        # x = layers.Reshape((7, 7, 64))(x)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder_outputs = layers.Dense(self.input_shape)(x) # todo in the original implementation we define a distribution on the output
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_mean(
                    keras.losses.mean_squared_error(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
# mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
os.chdir('..')
with open("example_config_VAE.json") as f:
    config = json.load(f)

training_epochs = config["training_epochs"]  # 250
batch_size = config["batch_size"]  # 250
learning_rate = config["learning_rate"]  # 0.0005
latent_size = config["latent_size"]  # 200
hidden_size_1 = config["hidden_size_1"]
hidden_size_2 = config["hidden_size_2"]
beta = config["beta"]
data_path = config["data_path"]
corrupt_data_path = config["corrupt_data_path"]
save_root = config["save_rootpath"]


data = pd.read_csv(data_path).values
data_missing = pd.read_csv(corrupt_data_path).values
# How many genes do we have? ie. what is the dimensiontality of Yobs?
n_row = data_missing.shape[1]  # dimensionality of data space

network_architecture = \
    dict(n_hidden_recog_1=hidden_size_1,  # 1st layer encoder neurons
         n_hidden_recog_2=hidden_size_2,  # 2nd layer encoder neurons
         n_hidden_gener_1=hidden_size_2,  # 1st layer decoder neurons
         n_hidden_gener_2=hidden_size_1,  # 2nd layer decoder neurons
         n_input=n_row,  # data input size
         n_z=latent_size)  # dimensionality of latent space

# Store the index of each sample that is complete
non_missing_row_ind = np.where(np.isfinite(np.sum(data_missing, axis=1)))
# Store the rows and columns of every missing data point in your "data_missing" numpy array
na_ind = np.where(np.isnan(data_missing))

sc = StandardScaler()
# Create a new numpy array that is complete (subset of simulated data_missing)
data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
# Find scaling factors from the complete set of the simulated missing data
sc.fit(data_missing_complete)
data_missing[na_ind] = 0  # Assign zero values to missing value indicies
# Transform missing data by the scaling factors defined from all complete values
data_missing = sc.transform(data_missing)
# Re-assign the missing values to the same positions as before
data_missing[na_ind] = np.nan
del data_missing_complete

# Remove strings and metadata from first few columns in data
data = np.delete(data, np.s_[0:4], axis=1)
data = sc.transform(data)

network_builder  = NetworkBuilder(latent_dim=2, input_shape=n_row, network_architecture=network_architecture)
encoder = network_builder.create_encoder()
decoder = network_builder.create_decoder()
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001))
vae.fit(data_missing, epochs=30, batch_size=128)