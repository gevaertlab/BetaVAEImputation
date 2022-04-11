import datetime
import tensorflow as tf
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
    data, data_missing = get_scaled_data()
    n_row = data.shape[1]
    network_architecture = \
        dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
             n_hidden_recog_2=2000,  # 2nd layer encoder neurons
             n_hidden_gener_1=2000,  # 1st layer decoder neurons
             n_hidden_gener_2=6000,  # 2nd layer decoder neurons
             n_input=n_row,  # data input size
             n_z=200)  # dimensionality of latent space
    load_pretrained = False
    if load_pretrained:
        encoder_path = ''
        decoder_path = ''
        encoder = keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
        decoder = keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    else:
        encoder, decoder = None, None
    vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=100, pretrained_encoder=encoder, pretrained_decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0))
    history = vae.fit(x=data, y=data, epochs=100, batch_size=256) #  callbacks=[tensorboard_callback]
    decoder_save_path = f"output/{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_decoder.keras"
    encoder_save_path = f"output/{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_encoder.keras"
    vae.encoder.save(encoder_save_path)
    vae.decoder.save(decoder_save_path)