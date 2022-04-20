import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pickle
import matplotlib.pyplot as plt
from betaVAEv2 import load_model_v2
from lib.helper_functions import get_scaled_data, evaluate_coverage



if __name__=="__main__":
    model_dir = '/home/jwells/Documents/BetaVAEImputation/output/non_masked_beta50_lr1e-05/epoch_1000_loss_14374.0/'
    output_dir = '/'.join(model_dir.split('/')[:-2]) + '/'
    encoder_path = model_dir + 'encoder.keras'
    decoder_path = model_dir +'decoder.keras'
    model = load_model_v2(encoder_path=encoder_path, decoder_path=decoder_path)
    data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    m_datasets = 40
    np.isnan(data_missing).any(axis=0)
    missing_rows = np.where(np.isnan(data_missing).any(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing[missing_rows]))
    multi_imputes = []
    for i in range(m_datasets):
        index_changes, missing_imputed = model.impute_multiple(data_corrupt=data_missing, max_iter=700,
                                                               method="Metropolis-within-Gibbs")
        multi_imputes.append(missing_imputed)
    plt.hist(index_changes, bins=133, range=[0,132])
    plt.show()
    evaluate_coverage(multi_imputes, data, data_missing, scaler)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + 'multi_impute.pickle', 'wb') as filehandle:
        pickle.dump(multi_imputes, filehandle)