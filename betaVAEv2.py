import datetime
import pickle
import numpy as np
import tensorflow as tf
# from tf.keras import layers

from lib.helper_functions import get_scaled_data

network_architecture = \
    dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
         n_hidden_recog_2=2000,  # 2nd layer encoder neurons
         n_hidden_gener_1=2000,  # 1st layer decoder neurons
         n_hidden_gener_2=6000,  # 2nd layer decoder neurons
         n_z=200,
         n_input=17175
         )  # dimensionality of latent space

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z (the latent representation)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoderV2(tf.keras.Model):
    def __init__(self, network_architecture=None, proba_output=True, beta=1,
                 pretrained_encoder=None, pretrained_decoder=None, **kwargs):
        super(VariationalAutoencoderV2, self).__init__(**kwargs)
        self.latent_dim = network_architecture['n_z']
        self.n_input_nodes = network_architecture['n_input']
        self.network_architecture = network_architecture
        self.proba_output = proba_output
        self.beta = beta
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = self.create_encoder()
        if pretrained_decoder is not None:
            self.decoder = pretrained_decoder
        else:
            self.decoder = self.create_decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    def create_encoder(self):
        n_hidden_recog_1 = self.network_architecture['n_hidden_recog_1']
        n_hidden_recog_2 = self.network_architecture['n_hidden_recog_2']
        encoder_inputs = tf.keras.Input(shape=self.n_input_nodes)
        h1 = tf.keras.layers.Dense(units=n_hidden_recog_1, activation="relu", name='h1')(encoder_inputs)
        n1 = tf.keras.layers.LayerNormalization(name='norm1')(h1)
        h2 = tf.keras.layers.Dense(units=n_hidden_recog_2, name='h2')(n1)
        n2 = tf.keras.layers.LayerNormalization(name='norm2')(h2)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(n2)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name="z_log_var")(h2)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
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
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        h1 = tf.keras.layers.Dense(n_hidden_gener_1, activation="relu")(latent_inputs)
        n1 = tf.keras.layers.LayerNormalization()(h1)
        h2 = tf.keras.layers.Dense(n_hidden_gener_2, activation="relu")(n1)
        n2 = tf.keras.layers.LayerNormalization()(h2)
        decoder_outputs = tf.keras.layers.Dense(self.n_input_nodes)(n2)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()
        return decoder

    def create_probabalistic_decoder(self):
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        n_hidden_gener_2 = self.network_architecture['n_hidden_gener_1']
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        h1 = tf.keras.layers.Dense(n_hidden_gener_1, activation="relu", name='h1')(latent_inputs)
        n1 = tf.keras.layers.LayerNormalization()(h1)
        h2 = tf.keras.layers.Dense(n_hidden_gener_2, activation="relu", name='h2')(n1)
        n2 = tf.keras.layers.LayerNormalization()(h2)
        x_hat_mean = tf.keras.layers.Dense(self.n_input_nodes, name='x_hat_mean')(n2)
        x_hat_log_sigma_sq = tf.keras.layers.Dense(self.n_input_nodes, name='x_hat_log_sigma_sq')(h2)
        decoder = tf.keras.Model(latent_inputs, [x_hat_mean, x_hat_log_sigma_sq], name="decoder")
        decoder.summary()
        return decoder

    def mvn_neg_ll(self, ytrue, ypreds):
        """Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
        This implementation implies diagonal covariance matrix.

        Parameters
        ----------
        ytrue: tf.tensor of shape [n_samples, n_dims]
            ground truth values
        ypreds: tuple of tf.tensors each of shape [n_samples, n_dims]
            predicted mu and logsigma values (e.g. by your neural network)

        Returns
        -------
        neg_log_likelihood: float
            negative loglikelihood averaged over samples

        This loss can then be used as a target loss for any keras model, e.g.:
            model.compile(loss=mvn_neg_ll, optimizer='Adam')
        """

        mu, log_sigma_sq = ypreds
        sigma = tf.keras.backend.sqrt(tf.keras.backend.exp(log_sigma_sq))
        logsigma = tf.keras.backend.log(sigma)
        n_dims = mu.shape[1]

        sse = -0.5 * tf.keras.backend.sum(tf.keras.backend.square((ytrue - mu) / sigma),
                           axis=1)  # divide by sigma instead of sigma squared because sigma is inside the square operation
        sigma_trace = -tf.keras.backend.sum(logsigma, axis=1)
        log2pi = -0.5 * n_dims * np.log(2 * np.pi)
        log_likelihood = sse + sigma_trace + log2pi

        return tf.keras.backend.mean(-log_likelihood)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            if self.proba_output:
                # output = self.decoder(z)
                # tf.print(z_mean[0])
                # tf.print(z_log_var[0])
                # tf.print(len(output))
                x_hat_mean, x_hat_log_sigma_sq = self.decoder(z)
                reconstruction_loss = self.mvn_neg_ll(y, (x_hat_mean, x_hat_log_sigma_sq))
                # reconstruction_loss = tfp.distributions.Normal(loc=x_hat_mean, scale=x_hat_log_sigma_sq).log_prob(y) # note that the scale parameter is sigma not sigma squared
            else:
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_mean(
                        tf.keras.losses.mean_squared_error(y, reconstruction)
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
        if self.proba_output:
            x_hat_mean, x_hat_log_sigma_sq = self.decoder(z_mean)
            return x_hat_mean
        else:
            return self.decoder(z_mean)

    def reconstruct(self, data, sample = 'mean'):
        z_mean, z_log_var, z = self.encoder(data)
        if sample == 'sample':
            x_hat_mu, x_hat_log_var = self.decoder(z)
        else:
            x_hat_mu, x_hat_log_var = self.decoder(z_mean)
        return x_hat_mu # todo consider whether it should sample from x_hat_mu here


    def evaluate_on_true(self, data_corrupt, data_complete, n_recycles=3, loss='RMSE'):
        losses = []
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt, axis=1)))
        data_miss_val = np.copy(data_corrupt[missing_row_ind[0], :])
        true_values_for_missing = data_complete[missing_row_ind[0], :]
        na_ind = np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0
        for i in range(n_recycles):
            data_reconstruct = self.reconstruct(data_miss_val)
            data_miss_val[na_ind] = data_reconstruct[na_ind]
            if loss == 'RMSE':
                losses.append(np.sqrt(((true_values_for_missing[na_ind] - data_reconstruct[na_ind])**2).mean()))
            elif loss == 'MAE':
                losses.append(np.abs(true_values_for_missing[na_ind] - data_reconstruct[na_ind]).mean())
        return losses

def load_model_v2(encoder_path='output/20220405-14:37:31_encoder.keras',
                  decoder_path='output/20220405-14:37:31_decoder.keras',
                  network_architecture = network_architecture, load_pretrained=True
                  ):

    if load_pretrained:
        encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
        decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    else:
        encoder = None
        decoder = None
    vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder)
    return vae

if __name__=="__main__":
    try: # this code block selects which GPU to run on (non essential)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[-1], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(logical_devices)
    except:
        pass
    data, data_missing = get_scaled_data()
    n_row = data.shape[1]
    network_architecture['n_input']=n_row  # data input size
    load_pretrained = True
    if load_pretrained:
        encoder_path =  'output/20220405-14:37:31_encoder.keras'
        decoder_path = 'output/20220405-14:37:31_decoder.keras'
        encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
        decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    else:
        encoder, decoder = None, None
    vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder, pretrained_decoder=decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0))
    history = vae.fit(x=data, y=data, epochs=100, batch_size=256) #  callbacks=[tensorboard_callback]
    decoder_save_path = f"output/{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_decoder.keras"
    encoder_save_path = f"output/{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_encoder.keras"
    vae.encoder.save(encoder_save_path)
    vae.decoder.save(decoder_save_path)
    with open('output/train_history_dict', 'wb') as file_handle:
        pickle.dump(history.history, file_handle)
