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

from experiments.array_dropout_analysis import remove_lock, evaluate_model, create_lock
from experiments.early_stopping_validation_analysis import get_additional_masked_data

def save_results(results, epoch, beta, results_path='beta_analysis.csv', lock_path='lock.txt'):
    if not os.path.exists(results_path):
        with open(results_path, 'w') as filehandle:
            filehandle.write('beta,epoch,mae,multi_mae,average_variance,prop_90,prop_95,prop_99\n')
    while os.path.exists(lock_path):
        print('sleeping due to file lock')
        time.sleep(2)
    create_lock()
    df = pd.read_csv(results_path)
    results['epoch'] = epoch
    results['beta'] = beta
    df  = df.append(results, ignore_index=True)
    df.to_csv(results_path, index=False)

if __name__=="__main__":
    args = sys.argv
    d_index = int(args[1]) -1
    data, data_missing_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    validation_w_nan, validation_complete, val_na_ind = get_additional_masked_data(data_missing_nan)
    other_missing_row_ind = np.where(np.isnan(data_missing_nan).any(axis=1))[0]
    training_input = np.append(validation_w_nan, data_missing_nan[other_missing_row_ind], axis=0)
    training_input = np.nan_to_num(training_input)
    n_col = data.shape[1]
    beta_rates = [4, 6, 8, 12, 16, 24, 32, 64, 100]
    beta = beta_rates[d_index]
    dropout = False

    model_settings = \
        dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
             n_hidden_recog_2=2000,  # 2nd layer encoder neurons
             n_hidden_gener_1=2000,  # 1st layer decoder neurons
             n_hidden_gener_2=6000,  # 2nd layer decoder neurons
             n_input=n_col,  # data input size
             n_z=200, # dimensionality of latent space
             )
    model_settings['beta'] = beta

    lr = 0.00005
    model = VariationalAutoencoderV2(model_settings=model_settings)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
    # model_savepath = f'output/dropout_rate{dropout_rate}_beta{beta}_lr{lr}/'
    # os.makedirs(model_savepath, exist_ok=True)
    epoch_granularity = {4:60, 6:60, 8:60, 12:60, 16:60, 24:60, 32:70, 50:120, 64:150, 100:250}
    n_epochs_dict = {4:1200,  6:2000, 8:2000, 12:2500, 16:2600, 24:2800, 32:3600, 50:4000, 64:4500, 100:4500}
    epochs = epoch_granularity[beta]
    rounds = int(n_epochs_dict[beta] / epochs) + 1
    for i in range(rounds):
        training_w_zeros = np.copy(training_input) # 667 obs
        validation_w_nan_cp = np.copy(validation_w_nan)
        history = model.fit(x=training_w_zeros, y=training_w_zeros, epochs=epochs, batch_size=256)
        loss = int(round(history.history['loss'][-1] , 0))#  callbacks=[tensorboard_callback]
        results = evaluate_model(model, validation_w_nan_cp, validation_complete, val_na_ind, scaler)
        completed_epochs = (i + 1) * epochs
        save_results(results, completed_epochs, beta, results_path='beta_analysis4.csv')
        remove_lock()


