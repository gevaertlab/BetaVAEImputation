import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data

if __name__=="__main__":
    decoder_path = '../output/non_masked_beta100_lr1e-05/epoch_2000_loss_15207.0/decoder.keras'
    encoder_path = '../output/non_masked_beta100_lr1e-05/epoch_2000_loss_15207.0/encoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 40
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    multi_imputes = []
    for i in range(m_datasets):
        index_changes, missing_imputed = model.impute_multiple(data_corrupt=data_missing, max_iter=200)
        # plt.hist(index_changes, range=[0,134], bins=133)
        # plt.show()
        # plt.savefig('1000_iterations_n_changes_per_index')
        multi_imputes.append(missing_imputed[na_ind])
    output_dir = 'output/non_masked_beta100_lr1e-05/'
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir +'multi_impute.pickle', 'wb') as filehandle:
        pickle.dump(multi_imputes, filehandle)