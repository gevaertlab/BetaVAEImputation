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
    data_complete = np.copy(data)
    missing_row_ind = np.where(np.isnan(data_missing_nan).any(axis=1))[0]
    data_w_missingness = data_missing_nan[missing_row_ind]
    na_ind = np.where(np.isnan(data_w_missingness))
    data_missing = np.nan_to_num(data_missing_nan)
    n_col = data.shape[1]
    beta = 50
    dropout = False

    network_architecture = \
        dict(n_hidden_recog_1=6000,  # 1st layer encoder neurons
             n_hidden_recog_2=2000,  # 2nd layer encoder neurons
             n_hidden_gener_1=2000,  # 1st layer decoder neurons
             n_hidden_gener_2=6000,  # 2nd layer decoder neurons
             n_input=n_col,  # data input size
             n_z=200, # dimensionality of latent space
             dropout_rate=0,
             )
    encoder, decoder = None, None
    beta = beta
    lr = 0.00001
    model = VariationalAutoencoderV2(network_architecture=network_architecture, beta=beta, dropout=dropout,
                                   pretrained_encoder=encoder, pretrained_decoder=decoder)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
    # model_savepath = f'output/dropout_rate{dropout_rate}_beta{beta}_lr{lr}/'
    # os.makedirs(model_savepath, exist_ok=True)
    epochs = 20
    for i in range(100):
        full_w_zeros = np.copy(data_missing) # 667 obs
        full_complete = np.copy(data_complete) #667 obs
        missing_w_nans = np.copy(data_w_missingness)
        missing_complete = np.copy(data_complete[missing_row_ind])
        history = model.fit(x=full_w_zeros, y=full_w_zeros, epochs=epochs, batch_size=256)
        loss = int(round(history.history['loss'][-1] , 0))#  callbacks=[tensorboard_callback]
        if loss < 1000:
            break
        results = evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler)
        completed_epochs = (i + 1) * epochs
        save_results(results, completed_epochs, beta, results_path='beta_analysis.csv')
        remove_lock()
