import os
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, evaluate_coverage

if __name__=="__main__":

    try:
        # If on eddie
        os.chdir("/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/")
        model_dir = '/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/'
        encoder_path = model_dir + '20220516-15:46:27_encoder.keras'
        decoder_path = model_dir +'20220516-15:46:27_decoder.keras'
        m_datasets = 40
        max_iter = 1000
    except FileNotFoundError:
        # If local
        model_dir = 'output/epochs_125_beta12_lr0.00001_loss16913/'
        encoder_path = model_dir + 'encoder.keras'
        decoder_path = model_dir + 'decoder.keras'
        m_datasets = 3
        max_iter = 5
    output_dir = model_dir + 'importance_sampling/'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path, beta=12)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
   
    # impute by importance sampling
    # output will be a list of all m datasets imputed by importance sampling (missing observations only)
    missing_imputed, ess = model.impute_multiple(data_corrupt=data_missing, max_iter=max_iter, 
                                                                m = m_datasets, beta = 12, 
                                                                method="importance sampling2")
    missing_imputed = np.array(missing_imputed)

    # export output of m-th dataset
    data = scaler.inverse_transform(data)
    truevals_data_missing = data[missing_rows]
    os.makedirs(output_dir, exist_ok=True)
    for i in range(m_datasets):
        outname = 'plaus_dataset_' + str(i+1)
        missing_imputed[i] = scaler.inverse_transform(missing_imputed[i])
        # export NA indices values
        na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[i][na_ind]})
        na_indices.to_csv(output_dir+'NA_imputed_values_' + outname + '.csv')
        # export each imputed dataset
        np.savetxt(output_dir + outname + ".csv", missing_imputed[i], delimiter=",")
        print("Mean Absolute Error:", sum(((missing_imputed[i][na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0])) 

    np.savetxt(output_dir + 'importance_sampling_ESS_' + str(len(ess)) + '_samples''.csv', np.array(ess), delimiter=',')
