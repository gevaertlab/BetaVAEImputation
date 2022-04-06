import tensorflow as tf
from betaVAEv2 import VariationalAutoencoderV2, Sampling, network_architecture
from lib.helper_functions import get_scaled_data


if __name__=="__main__":

    encoder_path = 'output/20220405-14:37:31_encoder.keras'
    decoder_path = 'output/20220405-14:37:31_decoder.keras'
    encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    data, data_missing = get_scaled_data()
    n_row = data.shape[1]
    network_architecture['n_input']=n_row
    vae = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder)