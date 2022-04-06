import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from betaVAEv2 import VariationalAutoencoderV2, Sampling, network_architecture
from lib.helper_functions import get_scaled_data

def evaluate_model_v2(model):
    data = pd.read_csv('../data/data_complete.csv').values
    data = np.array(np.copy(data[:, 4:]), dtype='float64')
    data_missing = pd.read_csv('../data/LGGGBM_missing_10perc_trial1.csv').values
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    data_missing[na_ind] = 0
    data_missing = sc.transform(data_missing)
    data = sc.transform(data)
    data_missing[na_ind] = np.nan
    del data_missing_complete
    losses = model.evaluate_on_true(data_missing, data, n_recycles=6, loss='RMSE', scaler=sc)
    bp=True

if __name__=="__main__":

    encoder_path = '../output/masked_20220406-07:32:09_encoder.keras'
    decoder_path = '../output/masked_20220406-07:32:09_decoder.keras'
    encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling})
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})

    model = VariationalAutoencoderV2(network_architecture=network_architecture, beta=1, pretrained_encoder=encoder,
                                   pretrained_decoder=decoder)

    model.compile()
    evaluate_model_v2(model)