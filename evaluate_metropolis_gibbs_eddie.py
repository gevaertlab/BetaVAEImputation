import os
import argparse
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, evaluate_coverage

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='1', help='m-th dataset you are generating via MI')

if __name__=="__main__":
    args = parser.parse_args()

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
        model_dir = 'output/epoch_1000_loss_14374.0/'
        encoder_path = model_dir + 'encoder.keras'
        decoder_path = model_dir + 'decoder.keras'
        m_datasets = 3
        max_iter = 5

    model_dir = '/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/'
    output_dir = model_dir + 'Metropolis-within-Gibbs/'
    outname = 'plaus_dataset_' + args.dataset
    print(outname)
    encoder_path = model_dir + '20220423-14:22:36_encoder.keras'
    decoder_path = model_dir +'20220423-14:22:36_decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path, beta=50)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
   
    # impute by metropolis-within-Gibbs 
    missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing, max_iter=max_iter,
                                                               beta = 12, method="Metropolis-within-Gibbs")

    # export output of m-th dataset
    data = scaler.inverse_transform(data)
    missing_imputed = scaler.inverse_transform(missing_imputed)

    truevals_data_missing = data[missing_rows]
    na_indices = pd.DataFrame({'true_values': truevals_data_missing[na_ind], outname: missing_imputed[na_ind]})

    os.makedirs(output_dir, exist_ok=True)

    na_indices.to_csv(output_dir+'NA_imputed_values_' + outname + '.csv')
    np.savetxt(output_dir + outname + ".csv", missing_imputed, delimiter=",")
    np.savetxt(output_dir + 'loglikelihood_across_iterations' + outname + '.csv', np.array(convergence_loglik), delimiter=',')

    print("Mean Absolute Error:", sum(((missing_imputed[na_ind] - truevals_data_missing[na_ind])**2)**0.5)/len(na_ind[0]))

