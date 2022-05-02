import time
import sys
import os
import numpy as np
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow import keras
try:
    os.chdir('/home/jwells/BetaVAEImputation')
    sys.path.append('/home/jwells/BetaVAEImputation')
except:
    pass
from lib.helper_functions import get_scaled_data, evaluate_coverage
from betaVAEv2 import VariationalAutoencoderV2, Sampling

def create_lock(path='lock.txt'):
    with open(path, 'w') as filehandle:
        filehandle.write('temp lock')

def remove_lock(path='lock.txt'):
    os.remove(path)

def evaluate_variance(model, missing_w_nans, na_ind):
    missing_w_zeros = np.nan_to_num(missing_w_nans)
    x_hat_mean, x_hat_log_sigma_sq = model.predict(missing_w_zeros)
    return np.mean(x_hat_log_sigma_sq.numpy()[na_ind])

def generate_multiple_and_evaluate_coverage(model, missing_w_nans, missing_complete, na_ind, scaler):
    multi_imputes_missing =[]
    m_datasets = 50
    for i in range(m_datasets):
        missing_imputed, convergence_loglik = model.impute_multiple(missing_w_nans, max_iter=40, method = "Metropolis-within-Gibbs")
        multi_imputes_missing.append(missing_imputed[na_ind])
    results  = evaluate_coverage(multi_imputes_missing, missing_complete, missing_w_nans, scaler)
    return results


def evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler):
    coverage_results = generate_multiple_and_evaluate_coverage(model, np.copy(missing_w_nans), missing_complete, na_ind, scaler)
    _, _, all_mae = model.impute_single(np.copy(missing_w_nans), missing_complete, n_recycles=6, loss='MAE', scaler=scaler, return_losses=True)
    results = dict(
    mae = all_mae[-1],
    average_variance = evaluate_variance(model, missing_w_nans, na_ind)
    )
    for k,v in coverage_results.items():
        results[k] = v
    return results

def save_results(results, epoch, dropout, results_path='dropout_analysis.csv', lock_path='lock.txt'):
    if not os.path.exists(results_path):
        with open(results_path, 'w') as filehandle:
            filehandle.write('dropout,epoch,mae,multi_mae,average_variance,prop_90,prop_95,prop_99\n')
    while os.path.exists(lock_path):
        print('sleeping due to file lock')
        time.sleep(2)
    create_lock()
    df = pd.read_csv(results_path)
    results['epoch'] = epoch
    results['dropout'] = dropout
    df  = df.append(results, ignore_index=True)
    df.to_csv(results_path, index=False)
    bp = True


if __name__=="__main__":
    args = sys.argv
    d_index = int(args[1]) -1
    data, data_missing_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    data_complete = np.copy(data)
    missing_row_ind = np.where(np.isnan(data_missing_nan).any(axis=1))[0]
    data_w_missingness = data_missing_nan[missing_row_ind]
    na_ind = np.where(np.isnan(data_w_missingness))
    data_missing = np.nan_to_num(data_missing_nan)
    n_col = data.shape[1]
    dropout_rates = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    dropout_rate = dropout_rates[d_index]
    if dropout_rate == 0:
        dropout = False
    else:
        dropout = True
    network_architecture = \
        dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
             n_hidden_recog_2=2000,  # 2nd layer encoder neurons
             n_hidden_gener_1=2000,  # 1st layer decoder neurons
             n_hidden_gener_2=6000,  # 2nd layer decoder neurons
             n_input=n_col,  # data input size
             n_z=200, # dimensionality of latent space
             dropout_rate=dropout_rate,
             )
    encoder, decoder = None, None
    beta = 50
    lr = 0.00001
    model = VariationalAutoencoderV2(network_architecture=network_architecture, beta=beta, dropout=dropout,
                                   pretrained_encoder=encoder, pretrained_decoder=decoder)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
    # model_savepath = f'output/dropout_rate{dropout_rate}_beta{beta}_lr{lr}/'
    # os.makedirs(model_savepath, exist_ok=True)
    epochs = 125

    for i in range(25):
        full_w_zeros = np.copy(data_missing) # 667 obs
        full_complete = np.copy(data_complete) #667 obs
        missing_w_nans = np.copy(data_w_missingness)
        missing_complete = np.copy(data_complete[missing_row_ind])
        history = model.fit(x=full_w_zeros, y=full_w_zeros, epochs=epochs, batch_size=256)
        loss = int(round(history.history['loss'][-1] , 0))#  callbacks=[tensorboard_callback]
        if loss < 11_000:
            break

        results = evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler)
        completed_epochs = (i + 1) * epochs
        save_results(results, completed_epochs, dropout_rate)
        remove_lock()
