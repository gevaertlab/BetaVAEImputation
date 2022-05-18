import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model
from lib.helper_functions import get_scaled_data, evaluate_coverage



if __name__=="__main__":
    model_dir = '/Users/judewells/Documents/dataScienceProgramming/BetaVAEImputation/output/new_trained_model/epoch_1000/'
    output_dir = '/'.join(model_dir.split('/')[:-2]) + '/'
    model = load_model(model_dir)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 5
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    imputed_datasets = []
    multi_imputes_missing = []
    for i in range(m_datasets):
        data_missing_preserved = np.copy(data_missing)
        missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing_preserved, max_iter=5,
                                                               method="Metropolis-within-Gibbs")
        imputed_datasets.append(np.copy(missing_imputed))

    for imp_data in imputed_datasets:
        multi_imputes_missing.append(imp_data[na_ind])

    evaluate_coverage(multi_imputes_missing, data, data_missing, scaler)
    # os.makedirs(output_dir, exist_ok=True)
    # with open(output_dir + 'multi_impute.pickle', 'wb') as filehandle:
    #     pickle.dump(multi_imputes, filehandle)