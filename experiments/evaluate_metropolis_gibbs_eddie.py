import os
import argparse
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

    model_dir = '/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/'
    output_dir = model_dir + 'Metropolis-within-Gibbs/'
    encoder_path = model_dir + '20220421-15:49:01_encoder.keras'
    decoder_path = model_dir +'20220421-15:49:01_decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 3
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    imputed_datasets = []
    multi_imputes_missing = []
    for i in range(m_datasets):
        missing_imputed, convergence_loglik = model.impute_multiple(data_corrupt=data_missing, max_iter=5,
                                                               method="Metropolis-within-Gibbs")
        imputed_datasets.append(missing_imputed)

    for imp_data in imputed_datasets:
        multi_imputes_missing.append(imp_data[na_ind])

    evaluate_coverage(multi_imputes_missing, data, data_missing, scaler)
    os.makedirs(output_dir, exist_ok=True)
    # with open(output_dir + 'multi_impute.pickle', 'wb') as filehandle:
    #     pickle.dump(multi_imputes, filehandle)
