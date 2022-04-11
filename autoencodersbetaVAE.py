import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.python.ops.gen_math_ops import exp
Normal = tf.contrib.distributions.Normal
np.random.seed(0)
tf.set_random_seed(0)

def calculate_losses(true, preds):
    return {
        "RMSE": np.sqrt(((true - preds) ** 2).mean()),
        "MAE": np.abs(true - preds).mean(),
        "r2_score": r2_score(true, preds)

    }

class VariationalAutoencoder(object):
#"VAE implementation is based on the implementation from  McCoy, J.T.,et al."
#https://www-sciencedirect-com.stanford.idm.oclc.org/science/article/pii/S2405896318320949"

    def __init__(self, network_architecture, transfer_fct=tf.nn.relu, 
                 learning_rate=0.001, batch_size=100, istrain=True, restore_path=None, beta=1):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta=beta
        
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        
        if istrain:
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
        else:
            self.sess=tf.Session()            
            self.saver.restore(self.sess, restore_path)
    
    def _create_network(self):

        network_weights = self._initialize_weights(**self.network_architecture)

        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        eps = tf.random_normal(tf.shape(self.z_mean), 0, 1, 
                               dtype=tf.float32)

        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.x_hat_mean, self.x_hat_log_sigma_sq = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, 
                            n_hidden_gener_1,  n_hidden_gener_2, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)
    
    def _generator_network(self, weights, biases):

        # self.z
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_hat_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        x_hat_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        
        return (x_hat_mean, x_hat_log_sigma_sq)
            
    def _create_loss_optimizer(self):

        # 1. Reconstruction loss - the negative log probability of of the input under the reconstructed Gaussian distribution
        # induced by the decoder in the latent space 
        # Note: if we want to use all observed data, instead of splitting into training and testing, this reconstruction loss/xhat distribution would only
        # consider observed/non missing data - in 119 replace log prob with zero for missing indices?
        X_hat_distribution = Normal(loc=self.x_hat_mean,
                                    scale=tf.sqrt(tf.exp(self.x_hat_log_sigma_sq))) # not taking square root here
        reconstr_loss = \
            -tf.reduce_sum(X_hat_distribution.log_prob(self.x), 1)
          
        # 2. Latent loss - KL divergence between between the latent space distribution induced by the encoder on the data
        # And some prior
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + self.beta *latent_loss)   # average over batch
        self.latent_cost=self.beta *latent_loss
        
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
                
    def fit(self, data):

        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: data})
        return cost
    
    def inspect_latent_cost (self, data):

        lc = self.sess.run(self.latent_cost, 
                                  feed_dict={self.x: data})
        return lc
    
    def transform_feature(self, data):

        return self.sess.run(self.z_mean, feed_dict={self.x: data})
    
    def reconstruct(self, data, sample = 'mean'):

        if sample == 'sample':
            x_hat_mu, x_hat_logsigsq = self.sess.run((self.x_hat_mean, self.x_hat_log_sigma_sq), 
                             feed_dict={self.x: data})
        
            eps = tf.random_normal(tf.shape(data), 1, 
                               dtype=tf.float32)

            x_hat = tf.add(x_hat_mu, 
                        tf.multiply(tf.sqrt(tf.exp(x_hat_logsigsq)), eps))
            x_hat = x_hat.eval()
        else:
            ## Here is where we are extracting the mean from the reconstructed data
            # Need to update this, wherever it's calling the mean from, we need to pull a sample from the distribution of z
            x_hat_mu = self.sess.run(self.x_hat_mean, 
                             feed_dict={self.x: data})
            x_hat = x_hat_mu
        
        return x_hat
    
    def impute(self, data_corrupt, max_iter = 10):
        """ Use VAE to impute missing values in data_corrupt. Missing values
            are indicated by a NaN.
        """
        # Find all missing row indices (samples/patients with any missing values)
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt,axis=1)))
        # copy to a new dataframe so that the original numpy array doesn't get changed
        data_miss_val = np.copy(data_corrupt[missing_row_ind[0],:]) 
        
        # Set all missing values at each location to zero before imputation begins
        na_ind = np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0
        convergence = []
        # Run through 10 iterations of computing latent space and reconstructing data and then feeding that back through the trained VAE
        for i in range(max_iter):
        
            data_reconstruct = self.reconstruct(data_miss_val)
            if i != 0:
                print(data_reconstruct[na_ind] - data_miss_val[na_ind])
            # Take average of absolute values across all values different between reconstructed data from previous step
            vals = np.abs(data_reconstruct[na_ind] - data_miss_val[na_ind])
            convergence.append(np.mean(vals))

            # l2 
            #vals_l2 = (data_reconstruct[na_ind] - data_miss_val[na_ind])**2
            #convergence_l2.append(np.sqrt(np.mean(vals_l2)))
            # Replace values in data_miss_val with reconstructed values at NA indices
            data_miss_val[na_ind] = data_reconstruct[na_ind]
        
        
        data_corrupt[missing_row_ind,:] = data_miss_val
        data_imputed = data_corrupt
        # Here i think we need a np.copy statement s.t. it doesn't change OG input

        return data_imputed, convergence


    def impute_multiple(self, data_corrupt, max_iter = 10):
        """
        Return a random sample from the decoder given a random sample from the latent distribution conditioned on data_corrupt
        """
        # Find all missing row indices (samples/patients with any missing values)
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt,axis=1)))
        data_miss_val = data_corrupt[missing_row_ind[0],:]

        # Set all missing values at each location to zero before imputation begins
        na_ind = np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0
        convergence = []
        convergence_loglik = []
        largest_imp_val = []
        avg_imp_val = []
        z_s_minus_1 = []
        x_hat_mean_s_minus_1 = []
        x_hat_log_sigma_sq_s_minus_1 = []
        # Run through 10 iterations of computing latent space and reconstructing data and then feeding that back through the trained VAE
        for i in range(max_iter):
            print("Running imputation iteration", i+1)
            ## Here - need a function which passes z_sample into the decoder and obtains the distribution of x_hat
            # Once x_hat distribution is generated, we need a random sample from this distribution
            # Then take all values at na_ind and replace them in the original data_miss_val with the sampled values from x_hat
            
            # calling x_hat_mean and x_hat_log_sigma_sq actually pulls a random sample from z via re-parametrization trick and then feeds it through the decoder 
            # And then computes the distribution and pulls a sample from this
            # Obtain a random sample from z, x_hat_mean and x_hat_log_sigma_sq given our corrupt data
            x_hat_mean, x_hat_log_sigma_sq, z_samp, z_mean, z_log_sigma_sq = self.sess.run([self.x_hat_mean,
                                                                                            self.x_hat_log_sigma_sq,
                                                                                            self.z, self.z_mean, self.z_log_sigma_sq],  feed_dict={self.x: data_miss_val})
            X_hat_distribution = Normal(loc=x_hat_mean,
                                    scale=tf.sqrt(tf.exp(x_hat_log_sigma_sq)))

            x_hat_sample = self.sess.run(X_hat_distribution.sample())
            # Take average of absolute values across all values different between reconstructed data from previous step
            convergence.append(np.mean(np.abs(x_hat_sample[na_ind] - data_miss_val[na_ind])))

            # monitor log-likelihood for Ymis across iterations
            # evaluate density of X_hat_distrubution at imputed values
            # first extract mean and stdev of x_hat at na_ind and compute X_hat_distribution
            X_hat_distribution_na = Normal(loc = x_hat_mean[na_ind], scale = tf.sqrt(tf.exp(x_hat_log_sigma_sq[na_ind])))

            # Compute negative log likelihood of X hat distribution from data_miss_val compared to _hat_sample
            sum_log_likl = self.sess.run(-tf.reduce_sum(X_hat_distribution_na.log_prob(x_hat_sample[na_ind])))

            convergence_loglik.append(sum_log_likl)

            # Store the largest value imputed at the NA indices
            largest_imp_val.append(np.amax(np.abs(x_hat_sample[na_ind])))

            # Store average absolute value of all NA indices (compared to baseline zero)
            avg_imp_val.append(np.mean(np.abs(x_hat_sample[na_ind])))

            ## To calculate acceptance probability, we need to calculate various metrics at each imputation iteration
            # Store z_samp from this iteration
            if i < 3:
                # First iteration you must set z_samp to the first sampling
                z_s_minus_1 = z_samp
                x_hat_mean_s_minus_1 = x_hat_mean
                x_hat_log_sigma_sq_s_minus_1 = x_hat_log_sigma_sq

                # Replace na_ind with x_hat_sample from first sampling
                data_miss_val[na_ind] = x_hat_sample[na_ind]
            else:
                # Define distributions
                z_Distribution = Normal(loc = z_mean, scale = tf.sqrt(tf.exp(z_log_sigma_sq)))
                z_prior = Normal(loc = np.zeros(z_mean.shape), scale = np.ones(z_mean.shape))
                X_hat_distr_s_minus_1 = Normal(loc = x_hat_mean_s_minus_1, scale = tf.sqrt(tf.exp(x_hat_log_sigma_sq_s_minus_1)))


                # Calculate log likelihood for previous and new sample to calculate acceptance probability with the following
                log_q_z_star, log_q_z_s_minus_1, log_p_z_star, log_p_z_s_minus_1, log_p_Y_z_star, log_p_Y_z_s_minus_1, =\
                                    self.sess.run(
                                    [tf.reduce_sum(z_Distribution.log_prob(z_samp)), # log_q_z_star
                                    tf.reduce_sum(z_Distribution.log_prob(z_s_minus_1)), # log_q_z_s_minus_1
                                    tf.reduce_sum(z_prior.log_prob(z_samp)), # log_p_z_star
                                    tf.reduce_sum(z_prior.log_prob(z_s_minus_1)), # log_p_z_s_minus_1
                                    tf.reduce_sum(X_hat_distribution.log_prob(data_miss_val)), # log_p_Y_z_star
                                    tf.reduce_sum(X_hat_distr_s_minus_1.log_prob(data_miss_val))]
                                    ) # log_p_Y_z_s_minus_1

                # Acceptance probability of sample z_star
                a_prob = np.exp(log_p_Y_z_star + log_p_z_star + log_q_z_s_minus_1 - (log_p_Y_z_s_minus_1 + log_p_z_s_minus_1 + log_q_z_star))
                # If we accept the new sample, set (s-1) z-sample as the new previous sampling
                if np.random.uniform() < a_prob:
                    print("new sample accepted with acceptance probability", a_prob)
                    z_s_minus_1 = z_samp
                    x_hat_mean_s_minus_1 = x_hat_mean
                    x_hat_log_sigma_sq_s_minus_1 = x_hat_log_sigma_sq

                    # Replace na_ind with x_hat_sample from this z_sampling
                    data_miss_val[na_ind] = x_hat_sample[na_ind] # Otherwise retain Ymis(s-1)
                else:
                    print("new sample rejected with acceptance probability", a_prob)


        # after the iterations have run through, you will have 1 of m plausible MI datasets
        data_corrupt[missing_row_ind,:] = data_miss_val
        data_imputed = data_corrupt

        return data_imputed, convergence, convergence_loglik, largest_imp_val, avg_imp_val

    def test_sampling(self, data_corrupt, max_iter = 10):
        """
        Return a random sample from the decoder given a random sample from the latent distribution conditioned on data_corrupt
        Test what this is actually returning, make sure the shape is correct (should be 667 x 17175)
        """
        # Find all missing row indices (samples/patients with any missing values)
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt,axis=1)))
        data_miss_val = data_corrupt[missing_row_ind[0],:]

        # Set all missing values at each location to zero before imputation begins
        na_ind = np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0

        # Run through one iteration
        # Obtain a random sample from z, x_hat_mean and x_hat_log_sigma_sq given our corrupt data
        z_sample, x_hat_mean, x_hat_log_sigma_sq = self.sess.run([self.z, self.x_hat_mean, self.x_hat_log_sigma_sq],
                             feed_dict={self.x: data_miss_val}) 

        X_hat_distribution = Normal(loc=x_hat_mean,
                                    scale=tf.sqrt(tf.exp(x_hat_log_sigma_sq)))

        # X_hat_distribution is the tensor associated with this order of operations, so we need to run it to get the values
        x_hat_sample = self.sess.run(X_hat_distribution.sample())
        
        return x_hat_sample


    def get_z_distribution(self, data):
        """
        Return latent distribution so we can run inspect it
        """
        z_space_comp = self.sess.run(self.z, 
                             feed_dict={self.x: data})
        return z_space_comp

    
    def train(self, data, training_epochs=10, display_step=10):

        
        missing_row_ind = np.where(np.isnan(np.sum(data,axis=1)))
        n_samples = np.size(data, 0) - missing_row_ind[0].shape[0]
        
        losshistory = []
        losshistory_epoch = []
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / self.batch_size)
            #random.seed(training_epochs)
            for i in range(total_batch):
                # Do we need to set a seed based on total_batch?
                # pass into next_batch() function s.t. data gets randomly partitioned the same way each time?
                batch_xs = next_batch(data,self.batch_size)
                cost = self.fit(batch_xs)
                lc = self.inspect_latent_cost(batch_xs)
                avg_cost += cost / n_samples * self.batch_size
               
            if epoch % display_step == 0:
                losshistory_epoch.append(epoch)
                losshistory.append(-avg_cost)
                print(f'Epoch: {epoch+1:.4f} Cost= {avg_cost:.9f}')
                #print (lc)
        self.losshistory = losshistory
        self.losshistory_epoch = losshistory_epoch
        return self

    def evaluate_on_true(self, data_corrupt, data_complete, n_recycles=3, loss='RMSE', scaler=None):
        """
        data_corrupt and data_complete should both be scaled to use this function
        """
        losses = []
        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt, axis=1)))[0]
        data_miss_val = np.copy(data_corrupt[missing_row_ind, :])
        true_values_for_missing = data_complete[missing_row_ind, :]
        na_ind = np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0 # todo should the zero be imputed after the scaling is already done?
        for i in range(n_recycles):
            data_reconstruct = self.reconstruct(data_miss_val)
            data_miss_val[na_ind] = data_reconstruct[na_ind]
            if scaler is not None:
                predictions = np.copy(scaler.inverse_transform(data_reconstruct)[na_ind])
                target_values = np.copy(scaler.inverse_transform(true_values_for_missing)[na_ind])
            else:
                predictions = np.copy(data_reconstruct[na_ind])
                target_values = np.copy(true_values_for_missing[na_ind])

            if loss == 'RMSE':
                losses.append(np.sqrt(((target_values - predictions)**2).mean()))
            elif loss == 'MAE':
                losses.append(np.abs(target_values - predictions).mean())
            elif loss =='all':
                multi_loss_dict = calculate_losses(target_values, predictions)
                losses.append(multi_loss_dict)
        return losses

def next_batch(data,batch_size):

    non_missing_row_ind = np.where(np.isfinite(np.sum(data,axis=1)))
    sample_ind = random.sample(list(non_missing_row_ind[0]),batch_size)
    data_sample = np.copy(data[sample_ind,:])
    if len(np.where(np.isnan(data_sample))[0]) > 0:
        print("Batch contains NAs")
        breakpoint = True
    return data_sample

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
