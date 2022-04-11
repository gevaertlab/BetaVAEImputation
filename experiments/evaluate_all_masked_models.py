import os
import tensorflow as tf
from lib.helper_functions import get_scaled_data
from betaVAEv2 import VariationalAutoencoderV2, Sampling, network_architecture

results = {}
os.chdir('..')
data, data_missing, sc = get_scaled_data(return_scaler=True, put_nans_back=True)
for dir in sorted(os.listdir('output/'), reverse=True):
    if not os.path.isdir('output/' + dir) or 'epoch' not in dir:
        continue
    encoder_path = 'output/' + dir + '/encoder_masked.keras'
    decoder_path = 'output/' + dir + '/decoder_masked.keras'
    epochs = dir.split('_')[-1]
    loss = int(dir.split('_')[1][4:])
    encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})

    model = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder)

    losses = model.evaluate_on_true(data_missing, data, n_recycles=6, loss='all', scaler=sc)
    mae = losses[-1]['MAE']
    results[epochs] = {'mae': mae, 'loss':loss}
    print(epochs, mae)

bp=True