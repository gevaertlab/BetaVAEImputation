import datetime
import numpy as np
import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from lib.helper_functions import get_scaled_data
from betaVAEv2 import VariationalAutoencoderV2, Sampling
from lib.masked_data_generator import DataGenerator

if __name__=="__main__":
    try: # this code block selects which GPU to run on (non essential)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[-1], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(logical_devices)
    except:
        pass
    data, data_missing = get_scaled_data(put_nans_back=True)
    complete_rows_idx = np.where(np.isfinite(data_missing).all(axis=1))[0]
    data_complete = data_missing[complete_rows_idx]
    training_generator = DataGenerator(x_train=data_complete, y_train=np.copy(data_complete), batchSize=250,
                                       prop_missing_patients=0.9, prop_missing_features=0.1)
    n_col = data_missing.shape[1]
    network_architecture = \
        dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
             n_hidden_recog_2=2000,  # 2nd layer encoder neurons
             n_hidden_gener_1=2000,  # 1st layer decoder neurons
             n_hidden_gener_2=6000,  # 2nd layer decoder neurons
             n_input=n_col,  # data input size
             n_z=200)  # dimensionality of latent space
    load_pretrained = False

    if load_pretrained:
        model_dir = ''
        encoder_path = model_dir + 'encoder.keras'
        decoder_path = model_dir + 'decoder.keras'
        encoder = keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
        decoder = keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    else:
        encoder, decoder = None, None
    beta = 50
    lr = 0.00001
    vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=beta, pretrained_encoder=encoder, pretrained_decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
    model_savepath = f'output/masking_generator_0p9_0p1_beta{beta}_lr{lr}/'
    os.makedirs(model_savepath, exist_ok=True)
    epochs = 200
    for i in range(14):
        history = vae.fit(training_generator, epochs=epochs, batch_size=256) #  callbacks=[tensorboard_callback]
        if i > 2:
            loss = int(round(history.history['loss'][-1] , 0))
            outdir = model_savepath + f"epoch_{(i+1)*epochs}_loss_{loss}/"
            decoder_save_path = f"{outdir}decoder.keras"
            encoder_save_path = f"{outdir}encoder.keras"
            vae.encoder.save(encoder_save_path)
            vae.decoder.save(decoder_save_path)