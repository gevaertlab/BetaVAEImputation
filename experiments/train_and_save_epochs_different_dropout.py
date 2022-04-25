import datetime
import pickle
import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
from lib.helper_functions import get_scaled_data
from betaVAEv2 import VariationalAutoencoderV2, Sampling

if __name__=="__main__":

    try: # this code block selects which GPU to run on (non essential)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[-1], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        print(logical_devices)
    except:
        pass
    data, data_missing = get_scaled_data(put_nans_back=False)
    n_col = data.shape[1]
    for dropout_rate in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        network_architecture = \
            dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
                 n_hidden_recog_2=2000,  # 2nd layer encoder neurons
                 n_hidden_gener_1=2000,  # 1st layer decoder neurons
                 n_hidden_gener_2=6000,  # 2nd layer decoder neurons
                 n_input=n_col,  # data input size
                 n_z=200, # dimensionality of latent space
                 dropout_rate=dropout_rate,
                 )
        load_pretrained = False

        if load_pretrained:
            model_dir = '/home/jwells/Documents/BetaVAEImputation/output/non_masked_beta10_lr1e-05/epoch_450_loss_9893/'
            encoder_path = model_dir + 'encoder.keras'
            decoder_path = model_dir + 'decoder.keras'
            encoder = keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
            decoder = keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
        else:
            encoder, decoder = None, None
        beta = 50
        lr = 0.00001
        vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=beta, dropout=True,
                                       pretrained_encoder=encoder, pretrained_decoder=decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
        model_savepath = f'output/dropout_rate{dropout_rate}_beta{beta}_lr{lr}/'
        os.makedirs(model_savepath, exist_ok=True)
        epochs = 100
        for i in range(50):
            history = vae.fit(x=data_missing, y=data_missing, epochs=epochs, batch_size=256) #  callbacks=[tensorboard_callback]
            if i > 3 and loss < 19_000:
                loss = int(round(history.history['loss'][-1] , 0))
                outdir = model_savepath + f"epoch_{(i+1)*epochs}_loss_{loss}/"
                decoder_save_path = f"{outdir}decoder.keras"
                encoder_save_path = f"{outdir}encoder.keras"
                vae.encoder.save(encoder_save_path)
                vae.decoder.save(decoder_save_path)
                if loss < 10000:
                    break
                with open(outdir + 'train_history_dict.pickle', 'wb') as file_handle:
                    pickle.dump(history.history, file_handle)