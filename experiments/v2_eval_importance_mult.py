import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data

if __name__=="__main__":
    model_dir = '../output/non_masked_beta50_lr1e-05/epoch_1000_loss_14374.0/'
    encoder_path = model_dir + 'encoder.keras'
    decoder_path = model_dir +'decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    model.impute_multiple(data_corrupt=data_missing, max_iter=10, m = 4, method = 'importance sampling2')
    bp=True
