import os
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import tensorflow_probability as tfp


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """


    mu, log_sigma_sq = ypreds
    sigma = K.sqrt(K.exp(log_sigma_sq))
    logsigma = K.log(sigma)
    n_dims = mu.shape[1]

    sse = -0.5 * K.sum(K.square((ytrue - mu) / sigma), axis=1) # divide by sigma instead of sigma squared because sigma is inside the square operation
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)
    log_likelihood = sse + sigma_trace + log2pi

    return K.mean(-log_likelihood)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class NetworkBuilder():
    def __init__(self, latent_dim=2, input_shape=(28, 28, 1), network_architecture=None, proba_output=True):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.network_architecture = network_architecture
        self.proba_output = proba_output

    def create_encoder(self):
        n_hidden_recog_1 = self.network_architecture['n_hidden_recog_1']
        n_hidden_recog_2 = self.network_architecture['n_hidden_recog_2']
        encoder_inputs = keras.Input(shape=self.input_shape)
        h1 = layers.Dense(units=n_hidden_recog_1, activation="relu")(encoder_inputs)
        h2 = layers.Dense(units=n_hidden_recog_2)(h1)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(h2)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(h2)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()
        return encoder

    def create_decoder(self):
        if self.proba_output:
            return self.create_probabalistic_decoder()
        else:
            return self.create_basic_decoder()

    def create_basic_decoder(self):
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        n_hidden_gener_2 = self.network_architecture['n_hidden_gener_1']
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h1 = layers.Dense(n_hidden_gener_1, activation="relu")(latent_inputs)
        h2 = layers.Dense(n_hidden_gener_2, activation="relu")(h1)
        decoder_outputs = layers.Dense(self.input_shape)(h2) # todo in the original implementation we define a distribution on the output
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def create_probabalistic_decoder(self):
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        n_hidden_gener_2 = self.network_architecture['n_hidden_gener_1']
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h1 = layers.Dense(n_hidden_gener_1, activation="relu", name='h1')(latent_inputs)
        h2 = layers.Dense(n_hidden_gener_2, activation="relu", name='h2')(h1)
        x_hat_mean = layers.Dense(self.input_shape, name='x_hat_mean')(h2)
        x_hat_log_sigma_sq = layers.Dense(self.input_shape, name='x_hat_log_sigma_sq')(h2)
        decoder = keras.Model(latent_inputs, [x_hat_mean, x_hat_log_sigma_sq], name="decoder")
        decoder.summary()
        return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, proba_output=True, beta=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.proba_output = proba_output
        self.beta = beta

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y =  data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            if self.proba_output:
                # output = self.decoder(z)
                # tf.print(output[0].shape)
                # tf.print(len(output))
                x_hat_mean, x_hat_log_sigma_sq = self.decoder(z)
                reconstruction_loss = gaussian_nll(y, (x_hat_mean, x_hat_log_sigma_sq))
                # reconstruction_loss = tfp.distributions.Normal(loc=x_hat_mean, scale=x_hat_log_sigma_sq).log_prob(y) # note that the scale parameter is sigma not sigma squared
            else:
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_mean(
                        keras.losses.mean_squared_error(y, reconstruction)
                    )
                )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) # identical form to the other implementation
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
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
    def predict(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_hat_mean, x_hat_log_sigma_sq = self.decoder(z_mean)
        return x_hat_mean

def evaluate_model_performance(model, missing_data, data, na_ind):
    preds = model.predict(missing_data)
    missing_preds = preds.numpy()[na_ind]
    true_values = data[na_ind]
    r2 = r2_score(true_values, missing_preds)
    return r2

os.chdir('..')
with open("example_config_VAE.json") as f:
    config = json.load(f)

training_epochs = config["training_epochs"]  # 250
batch_size = config["batch_size"]  # 250
learning_rate = config["learning_rate"]  # 0.0005
latent_size = config["latent_size"]  # 200
hidden_size_1 = 500 #config["hidden_size_1"]
hidden_size_2 = 200 #config["hidden_size_2"]
beta = config["beta"]
data_path = config["data_path"]
corrupt_data_path = config["corrupt_data_path"]
save_root = config["save_rootpath"]


data = pd.read_csv(data_path).values
data_missing = pd.read_csv(corrupt_data_path).values
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
# data_missing[na_ind] = np.nan
del data_missing_complete

# Remove strings and metadata from first few columns in data
data = np.delete(data, np.s_[0:4], axis=1)
data = sc.transform(data)

proba_output=True
tf.random.set_seed(13)
network_builder  = NetworkBuilder(latent_dim=2, input_shape=n_row,
                                  network_architecture=network_architecture, proba_output=proba_output)
load_model = False
encoder_path = 'output/encoder_model.keras'
decoder_path = 'output/decoder_model.keras'
if load_model:
    encoder = keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
else:
    encoder = network_builder.create_encoder()
    decoder = network_builder.create_decoder()
vae = VAE(encoder, decoder, proba_output=proba_output, beta=beta)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000005))

r_squared_on_missing = evaluate_model_performance(model=vae, missing_data=data_missing, data=data, na_ind=na_ind)
print(f'r-squared on the missing values: {r_squared_on_missing}')
vae.fit(x=data_missing,y=data, epochs=100, batch_size=batch_size)
vae.encoder.save(encoder_path)
vae.decoder.save(decoder_path)