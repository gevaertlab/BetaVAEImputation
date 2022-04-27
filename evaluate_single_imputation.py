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

    model_dir = 'output/epoch_1000_loss_14374.0/'
    output_dir = model_dir + 'single_imputation/'
    outname = 'single_imputed_dataset'
    print(outname)
    encoder_path = model_dir + 'encoder.keras'
    decoder_path = model_dir +'decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
   
    # impute by metropolis-within-Gibbs 
    missing_imputed, conv = model.impute_single(data_corrupt=data_missing, data_complete = data, n_recycles=5)

    # export output of m-th dataset
    data = scaler.inverse_transform(data)
    missing_imputed = scaler.inverse_transform(missing_imputed)

    truevals_data_missing = data[missing_rows]
    na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[na_ind]})

    os.makedirs(output_dir, exist_ok=True)

    na_indices.to_csv(output_dir+'NA_imputed_values_' + outname + '.csv')
    np.savetxt(output_dir + outname + ".csv", missing_imputed, delimiter=",")

    print("Mean Absolute Error:", sum(((missing_imputed[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))
