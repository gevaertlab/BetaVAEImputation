import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import json
try:
    with open("example_config_VAE.json") as f:
        config = json.load(f)
except:
    with open("../example_config_VAE.json") as f:
        config = json.load(f)
data_path = config["data_path"]
corrupt_data_path = config["corrupt_data_path"]
from sklearn.preprocessing import StandardScaler

def get_scaled_data():
    for _ in range(3):
        if os.getcwd().split('/')[-1] == 'BetaVAEImputation':
            break
        os.chdir('..')
    data = pd.read_csv(data_path).values
    data_missing = pd.read_csv(corrupt_data_path).values
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    data_missing[na_ind] = 0
    data_missing = sc.transform(data_missing)
    del data_missing_complete
    data = np.array(np.copy(data[:,4:]),dtype='float64')
    data = sc.transform(data)
    return data, data_missing


def apply_scaler(data, data_missing, return_scaler=False):
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    data_missing[na_ind] = 0
    # Scale the testing data with model's trianing data mean and variance
    data_missing = sc.transform(data_missing)
    data_missing[na_ind] = np.nan
    del data_missing_complete
    data = sc.transform(data)
    if return_scaler:
        return data, data_missing, sc
    else:
        return data, data_missing

def load_saved_model(config_path = 'JW_config_VAE.json'):
    from autoencodersbetaVAE import VariationalAutoencoder
    n_col = 17175
    running_directory = os.getcwd()
    if running_directory.split('/')[-1] != 'BetaVAEImputation':
        os.chdir('..')
    with open(config_path) as f:
        config = json.load(f)
    training_epochs = config["training_epochs"]  # 250
    batch_size = config["batch_size"]  # 250
    learning_rate = config["learning_rate"]  # 0.0005
    latent_size = config["latent_size"]  # 200
    hidden_size_1 = config["hidden_size_1"]
    hidden_size_2 = config["hidden_size_2"]
    restore_root = config["save_rootpath"]
    beta = config["beta"]
    Decoder_hidden1 = hidden_size_1  # 6000
    Decoder_hidden2 = hidden_size_2  # 2000
    Encoder_hidden1 = hidden_size_2  # 2000
    Encoder_hidden2 = hidden_size_1  # 6000

    network_architecture = \
        dict(n_hidden_recog_1=Encoder_hidden1,  # 1st layer encoder neurons
             n_hidden_recog_2=Encoder_hidden2,  # 2nd layer encoder neurons
             n_hidden_gener_1=Decoder_hidden1,  # 1st layer decoder neurons
             n_hidden_gener_2=Decoder_hidden2,  # 2nd layer decoder neurons
             n_input=n_col,  # data input size
             n_z=latent_size)  # dimensionality of latent space

    rp = restore_root + "ep" + str(training_epochs) + "_bs" + str(batch_size) + "_lr" + str(
        learning_rate) + "_bn" + str(latent_size) + "_opADAM" + "_beta" + str(beta) + "_betaVAE" + ".ckpt"
    print("restore path: ", rp)
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size, istrain=False, restore_path=rp, beta=beta)
    os.chdir(running_directory)
    return vae


def add_zero_mask(x, col_prop=0.1, row_prop=0.2):
    """
    Changes a randomly selected proportion of the data to zero to simulate missing values
    """
    n_miss_cols = int(x.shape[1]*col_prop)
    n_miss_rows = int(x.shape[0]*row_prop)
    miss_cols = np.array([np.random.choice(x.shape[1], size=n_miss_cols, replace=False) for _ in range(n_miss_rows)]).reshape(-1)
    miss_rows = np.repeat(np.random.choice(x.shape[0], size=n_miss_rows, replace=False), repeats=n_miss_cols)
    x[(miss_cols, miss_rows)] = 0
    return x