import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


Normal = tfp.distributions.Normal
np.random.seed(0)
tf.set_random_seed(0)

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
        
        self.x = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        self._create_network()
        
        self._create_loss_optimizer()
        
        self.saver = tf.compat.v1.train.Saver()

        init = tf.global_variables_initializer()
        
        if istrain:
            self.sess = tf.compat.v1.InteractiveSession()
            self.sess.run(init)
        else:
            self.sess=tf.compat.v1.Session()            
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
        
        X_hat_distribution = Normal(loc=self.x_hat_mean,
                                    scale=tf.exp(self.x_hat_log_sigma_sq))
        reconstr_loss = \
            -tf.reduce_sum(X_hat_distribution.log_prob(self.x), 1)
          

        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + self.beta *latent_loss)   # average over batch
        self.latent_cost=self.beta *latent_loss
        
        self.optimizer = \
            tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
                
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
            x_hat_mu = self.sess.run(self.x_hat_mean, 
                             feed_dict={self.x: data})
            x_hat = x_hat_mu
        
        return x_hat
    
    def impute(self, data_corrupt, max_iter = 10):
        """ Use VAE to impute missing values in data_corrupt. Missing values
            are indicated by a NaN.
        """

        missing_row_ind = np.where(np.isnan(np.sum(data_corrupt,axis=1)))
        data_miss_val = data_corrupt[missing_row_ind[0],:]
        
        na_ind= np.where(np.isnan(data_miss_val))
        data_miss_val[na_ind] = 0
        
        for i in range(max_iter):
        
            data_reconstruct = self.reconstruct(data_miss_val)
            data_miss_val[na_ind] = data_reconstruct[na_ind]
        
        data_corrupt[missing_row_ind,:] = data_miss_val
        data_imputed = data_corrupt

        return data_imputed
    
    def train(self, data, training_epochs=10, display_step=10):

        
        missing_row_ind = np.where(np.isnan(np.sum(data,axis=1)))
        n_samples = np.size(data, 0) - missing_row_ind[0].shape[0]
        
        losshistory = []
        losshistory_epoch = []
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / self.batch_size)
            for i in range(total_batch):
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
        
        log_dir = "./logs/train/"
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, graph=self.sess.graph_def)

        return self

def next_batch(data,batch_size):

    non_missing_row_ind = np.where(np.isfinite(np.sum(data,axis=1)))
    sample_ind = random.sample(list(non_missing_row_ind[0]),batch_size)
    data_sample = np.copy(data[sample_ind,:])
    
    return data_sample

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
