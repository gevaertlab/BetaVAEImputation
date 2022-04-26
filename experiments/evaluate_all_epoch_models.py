import os
import tensorflow as tf
from lib.helper_functions import get_scaled_data
from betaVAEv2 import VariationalAutoencoderV2, Sampling, network_architecture
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
results = {}
os.chdir('..')
data, data_missing, sc = get_scaled_data(return_scaler=True, put_nans_back=True)
root_dir = 'output/dropout_rate0.05_beta50_lr1e-05/'
for dir in sorted(os.listdir(root_dir), reverse=True):
    if not os.path.isdir(root_dir + dir) or 'epoch' not in dir:
        continue
    encoder_path = root_dir + dir + '/encoder.keras'
    decoder_path = root_dir + dir + '/decoder.keras'
    epochs = int(dir.split('_')[1])
    try:
        loss = int(dir.split('_')[-1][4:])
    except:
        loss = None
    encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})

    model = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder)

    losses = model.evaluate_on_true(data_missing, data, n_recycles=6, loss='all', scaler=sc)
    mae = losses[-1]['MAE']
    results[epochs] = {'mae': mae, 'loss':loss}
    print(epochs, mae)

bp=True
plt.plot([int(e) for e in sorted(results.keys())], [results[e]['mae'] for e in sorted(results)])
plt.title('loss over epochs')
plt.xlabel('epochs')
plt.ylabel('MAE')
plt.savefig(root_dir+'plotted_loss_over_epochs')
plt.show()
