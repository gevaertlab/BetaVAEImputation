import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from autoencodersbetaVAE import VariationalAutoencoder

def apply_scaler(data, data_missing):
    non_missing_row_ind = np.where(np.isfinite(np.sum(data_missing, axis=1)))
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
    return data, data_missing

def load_saved_model(config_path = 'JW_config_VAE.json'):
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