import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data

def evaluate_coverage(multi_imputes=None, data=None, data_missing=None, scaler=None):
    if multi_imputes is None:
        # '../output/non_masked_beta100_lr1e-05/multi_impute.pickle'
        with open('../output/non_masked_beta50_lr1e-05/multi_impute.pickle', 'rb') as filehandle:
            multi_imputes = np.array(pickle.load(filehandle))
    if data is None:
        data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    na_ind = np.where(np.isnan(data_missing))
    means = np.mean(multi_imputes, axis=0)
    unscaled_st_devs = np.std(multi_imputes, axis=0)
    unscaled_differences = np.abs(data[na_ind] - means)
    n_deviations = unscaled_differences / unscaled_st_devs
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    print('prop 90:', prop_90)
    print('prop 95:', prop_95)
    print('prop 99:', prop_99)
    data = scaler.inverse_transform(data)
    data_missing[na_ind] = means
    data_missing = scaler.inverse_transform(data_missing)
    differences = np.abs(data[na_ind] - data_missing[na_ind])
    print('average absolute error:', np.mean(differences))



if __name__=="__main__":
    model_dir = './output/epoch_1000_loss_14374.0/'
    output_dir = '/output/results/'
    encoder_path = './output/20220413-13_00_29_encoder.keras'
    decoder_path = './output/20220413-13_00_25_decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 10
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    #multi_imputes = []
    multi_imputes = model.impute_multiple(data_corrupt=data_missing, max_iter=15, m = m_datasets, method = 'importance sampling2')
    print("pause here")
    
    
    for i in range(m_datasets):
        index_changes, missing_imputed = model.impute_multiple(data_corrupt=data_missing, max_iter=15, m = m_datasets, method = 'importance sampling2')
        multi_imputes.append(missing_imputed[na_ind])
    plt.hist(index_changes, bins=133, range=[0,132])
    plt.show()
    evaluate_coverage(multi_imputes, data, data_missing, scaler)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + 'multi_impute.pickle', 'wb') as filehandle:
        pickle.dump(multi_imputes, filehandle)

