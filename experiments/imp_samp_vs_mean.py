import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from betaVAEv2 import VariationalAutoencoderV2, Sampling
from lib.helper_functions import get_scaled_data

network_architecture = \
    dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
         n_hidden_recog_2=2000,  # 2nd layer encoder neurons
         n_hidden_gener_1=2000,  # 1st layer decoder neurons
         n_hidden_gener_2=6000,  # 2nd layer decoder neurons
         n_z=200,
         n_input=17175
         )

class ModVariationalAutoencoder(VariationalAutoencoderV2):

    def get_weight_and_likelihood(self, z, data_miss_val, z_Distribution, z_prior, probability_mask):
        x_hat_mean, x_hat_log_sigma_sq = self.decoder.predict(z)
        x_hat_sigma = np.exp(0.5 * x_hat_log_sigma_sq)
        X_hat_distribution = tfp.distributions.Normal(loc=x_hat_mean, scale=x_hat_sigma)
        log_p_Yc_z = tf.reduce_sum(X_hat_distribution.log_prob(data_miss_val).numpy() * probability_mask,
                                   axis=1).numpy()
        log_p_z = tf.reduce_sum(z_prior.log_prob(z), axis=1).numpy()
        log_q_z_Y = tf.reduce_sum(z_Distribution.log_prob(z), axis=1).numpy()
        logr = log_p_Yc_z + log_p_z - log_q_z_Y
        return logr, log_p_Yc_z

    def impute_multiple(self, data_corrupt, max_iter=10, method='importance sampling2 modified'):
        missing_row_ind = np.where(np.isnan(data_corrupt).any(axis=1))
        data_miss_val = data_corrupt[missing_row_ind[0],:]
        na_ind = np.where(np.isnan(data_miss_val))
        compl_ind = np.where(np.isfinite(data_miss_val))
        data_miss_val[na_ind] = 0
        z_prior = tfp.distributions.Normal(loc=np.zeros([data_miss_val.shape[0], self.latent_dim]), scale=np.ones([data_miss_val.shape[0], self.latent_dim]))

        if method == "importance sampling2 modified":
            logr_counter = 0
            likelihood_counter = 0
            logweights = []
            z_sample_l = []
            z_mean, z_log_sigma_sq, z_samp = self.encoder.predict(data_miss_val)
            z_Distribution = tfp.distributions.Normal(loc=z_mean, scale=tf.sqrt(tf.exp(z_log_sigma_sq)))
            probability_mask = np.zeros(data_miss_val.shape)
            probability_mask[compl_ind] = 1
            indicies_changes = []
            for i in range(max_iter):
                if i == 0:
                    z_mean_logr, z_mean_log_p_Yc_z = self.get_weight_and_likelihood(z_mean, data_miss_val, z_Distribution, z_prior, probability_mask)
                else:
                    z_l = z_Distribution.sample().numpy()
                    logr, log_p_Yc_z = self.get_weight_and_likelihood(z_l, data_miss_val, z_Distribution, z_prior, probability_mask)
                    logweights.append(logr)
                    z_sample_l.append(z_l)
                    obs_higher_weight = sum(logr > z_mean_logr)
                    indicies_changes += list(np.where(logr > z_mean_logr)[0])
                    logr_counter += obs_higher_weight
                    print(f'n obs with higher weight in round {i}: {obs_higher_weight}')
                    obs_higher_likelihood = sum(log_p_Yc_z > z_mean_log_p_Yc_z)
                    print(f'n obs with higher likelihood in round {i}: {obs_higher_likelihood}')
                    likelihood_counter += obs_higher_likelihood
            plt.hist(indicies_changes, bins=133, range=[0,132])
            plt.savefig('output/index_changes_zmean')
            plt.show()



            print(f'{logr_counter} out of {max_iter*len(data_miss_val)} samples had a greater un-normalized weight than z-mean')
            print(f'{likelihood_counter} out of {max_iter*len(data_miss_val)} samples had a greater LIKELIHOOD (log_p_Yc_z) than z-mean')

        else:
            print('this class only works with the `importance sampling2` method')

if __name__=="__main__":
    model_dir = '../output/non_masked_beta50_lr1e-05/epoch_1000_loss_14374.0/'
    encoder_path = model_dir + 'encoder.keras'
    decoder_path = model_dir +'decoder.keras'
    encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    model = ModVariationalAutoencoder(network_architecture=network_architecture, beta=50, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder, dropout=False)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    model.impute_multiple(data_corrupt=data_missing, max_iter=1000, method = 'importance sampling2 modified')