import tensorflow as tf

"""
Attempting to rebuild the VariationalAutoencoder class from autoencodersbetaVAE
but without the dependencies on tensorflow version 1
"""

class VariationalAutoencoder():
    def __init__(self, network_architecture, transfer_fct=tf.nn.relu,
                 learning_rate=0.001, batch_size=100, istrain=True, restore_path=None, beta=1):

        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = beta

        # self.x = tf.compat.v1.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self.x_dim = network_architecture["n_input"]

        self._create_network()

        self._create_loss_optimizer()

        self.saver = tf.compat.v1.train.Saver()

        init = tf.compat.v1.global_variables_initializer()

        if istrain:
            self.sess = tf.compat.v1.InteractiveSession()
            self.sess.run(init)
        else:
            self.sess = tf.compat.v1.Session()
            self.saver.restore(self.sess, restore_path)

    def _create_network(self):
        pass

    def _create_loss_optimizer(self):
        X_hat_distribution = Normal(loc=self.x_hat_mean,
                                    scale=tf.exp(self.x_hat_log_sigma_sq))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)