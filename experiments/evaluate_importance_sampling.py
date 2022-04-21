import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, evaluate_coverage

if __name__=="__main__":
    model_dir = '../output/non_masked_beta50_lr1e-05/epoch_1000_loss_14374.0/'
    encoder_path = model_dir + 'encoder.keras'
    decoder_path = model_dir +'decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    mult_imp_datasets, effective_sample = model.impute_multiple(data_corrupt=data_missing, max_iter=10_000, m = 40, method = 'importance sampling2')
    print(f'Effective sample size: {round(np.mean(effective_sample), 2)}')
    multi_imputes_missing = []
    missing_row_ind = np.where(np.isnan(data_missing).any(axis=1))[0]
    data_miss_val = data_missing[missing_row_ind, :]
    na_ind = np.where(np.isnan(data_miss_val))
    for imputed_dataset in mult_imp_datasets:
        multi_imputes_missing.append(imputed_dataset[na_ind])
    evaluate_coverage(multi_imputes_missing, data, data_missing, scaler)
    bp=True
