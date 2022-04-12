import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, get_accuracy_metrics

def get_average_imputed_error(imputed_vals=None):
    """
    Use this function if the multiple imputations have already been
    computed and saved to disk as a pickle file.
    """
    if imputed_vals is None:
        with open('../output/imputed_beta100/multi_impute.pickle', 'rb') as filehandle:
            imputed_vals = np.array(pickle.load(filehandle))
    average_imputed = imputed_vals.mean(axis=0)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    mae = get_accuracy_metrics(data, data_missing, average_imputed, scaler)
    print('MAE of the average multiply imputed value:', mae)

if __name__=="__main__":
    decoder_path = '../output/non_masked_beta_100_lr1e-05/epoch_160/decoder.keras'
    encoder_path = '../output/non_masked_beta_100_lr1e-05/epoch_160/encoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 10
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    multi_imputes = []
    for i in range(m_datasets):
        index_changes, missing_imputed = model.impute_multiple(data_corrupt=data_missing, max_iter=100)
        # plt.hist(index_changes, range=[0,134], bins=133)
        # plt.show()
        # plt.savefig('1000_iterations_n_changes_per_index')
        multi_imputes.append(missing_imputed[na_ind])
    average_imputed  = np.array(multi_imputes).mean(axis=0)
    get_accuracy_metrics(data, data_missing, average_imputed, scaler)
    with open('output/imputed_beta100/multi_impute.pickle', 'wb') as filehandle:
        pickle.dump(multi_imputes, filehandle)