import random
import numpy as np
import tensorflow as tf
Normal = tf.contrib.distributions.Normal
np.random.seed(0)
tf.set_random_seed(0)

"""VAE implementation is based on the implementation from  McCoy, J.T.,et al."""
#https://www-sciencedirect-com.stanford.idm.oclc.org/science/article/pii/S2405896318320949"

def next_batch(Xdata,batch_size, MissingVals = False):

        ObsRowIndex = np.where(np.isfinite(np.sum(Xdata,axis=1)))
        X_indices = random.sample(list(ObsRowIndex[0]),batch_size)
        Xdata_sample = np.copy(Xdata[X_indices,:])
    
    return Xdata_sample

def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):

    def __init__(self, network_architecture, transfer_fct=tf.nn.relu, 
                 learning_rate=0.001, batch_size=100, istrain=True, restore_path=None, beta=1):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta=beta
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        self.saver = tf.train.Saver()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()
        
        # Launch the session
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
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
                
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def inspect_latent_cost (self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        lc = self.sess.run(self.latent_cost, 
                                  feed_dict={self.x: X})
        return lc
    
    def transform_feature(self, X):
        """Transform data by mapping it into the latent space."""

        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def reconstruct(self, X, sample = 'mean'):

        if sample == 'sample':
            x_hat_mu, x_hat_logsigsq = self.sess.run((self.x_hat_mean, self.x_hat_log_sigma_sq), 
                             feed_dict={self.x: X})
        
            eps = tf.random_normal(tf.shape(X), 0, 1, 
                               dtype=tf.float32)

            x_hat = tf.add(x_hat_mu, 
                        tf.multiply(tf.sqrt(tf.exp(x_hat_logsigsq)), eps))
            x_hat = x_hat.eval()
        else:
            x_hat_mu = self.sess.run(self.x_hat_mean, 
                             feed_dict={self.x: X})
            x_hat = x_hat_mu
        
        return x_hat
    
    def impute(self, X_corrupt, max_iter = 10):
        """ Use VAE to impute missing values in X_corrupt. Missing values
            are indicated by a NaN.
        """
        # Select the rows of the datset which have one or more missing values:
        NanRowIndex = np.where(np.isnan(np.sum(X_corrupt,axis=1)))
        x_miss_val = X_corrupt[NanRowIndex[0],:]
        
        # initialise missing values with arbitrary value
        NanIndex = np.where(np.isnan(x_miss_val))
        x_miss_val[NanIndex] = 0
        
        MissVal = np.zeros([max_iter,len(NanIndex[0])], dtype=np.float32)
        
        for i in range(max_iter):
            MissVal[i,:] = x_miss_val[NanIndex]
            
            # reconstruct the inputs, using the mean:
            x_reconstruct = self.reconstruct(x_miss_val)
            x_miss_val[NanIndex] = x_reconstruct[NanIndex]
        
        X_corrupt[NanRowIndex,:] = x_miss_val
        X_imputed = X_corrupt
        self.MissVal = MissVal
        
        return X_imputed
    
    def train(self, XData, training_epochs=10, display_step=10):
        """ Train VAE in a loop, using numerical data"""
        
        # number of rows with complete entries in XData
        NanRowIndex = np.where(np.isnan(np.sum(XData,axis=1)))
        n_samples = np.size(XData, 0) - NanRowIndex[0].shape[0]
        
        losshistory = []
        losshistory_epoch = []
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = next_batch(XData,self.batch_size, MissingVals = False)
                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                lc = self.inspect_latent_cost(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * self.batch_size
               
            # Display logs per epoch step
            if epoch % display_step == 0:
                losshistory_epoch.append(epoch)
                losshistory.append(-avg_cost)
                print(f'Epoch: {epoch+1:.4f} Cost= {avg_cost:.9f}')
                print (lc)
        self.losshistory = losshistory
        self.losshistory_epoch = losshistory_epoch
        return self
