from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from betaVAEv2 import VariationalAutoencoderV2
from lib import get_scaled_data
"""
The goal of this experiment is to determine if a VAE
trained with input dropout does better at reconstructing
unseen missing data samples - than one which is trained
just reconstructing complete data
"""

def train_test_split(data, data_missing):
    test_indices

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[-1], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
print(logical_devices)
data, data_missing = get_scaled_data()
n_row = data.shape[1]
network_architecture = \
    dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
         n_hidden_recog_2=2000,  # 2nd layer encoder neurons
         n_hidden_gener_1=2000,  # 1st layer decoder neurons
         n_hidden_gener_2=6000,  # 2nd layer decoder neurons
         n_input=n_row,  # data input size
         n_z=200)  # dimensionality of latent space

vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=100, input_dropout=True)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0))
history = vae.fit(x=data, y=data, epochs=1000, batch_size=256)  # callbacks=[tensorboard_callback]

kf = KFold(n_splits=5)
for train_index, test_index in kf.split(data):
    x_train, x_test = data[train_index], data[test_index]