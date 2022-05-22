import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
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


def evaluate_coverage(multi_imputes=None, data=None, data_missing=None, scaler=None):
    assert data_missing.shape == data.shape
    if multi_imputes is None:
        # '../output/non_masked_beta100_lr1e-05/multi_impute.pickle'
        with open('../output/non_masked_beta50_lr1e-05/multi_impute.pickle', 'rb') as filehandle:
            multi_imputes = np.array(pickle.load(filehandle))
    if data is None:
        data, data_missing, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
    na_ind = np.where(np.isnan(data_missing))
    means = np.mean(multi_imputes, axis=0)
    unscaled_st_devs = np.std(multi_imputes, axis=0)
    unscaled_differences = np.abs(data[na_ind] - means)
    n_deviations = unscaled_differences / unscaled_st_devs
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    results = {
        'prop_90': prop_90,
        'prop_95': prop_95,
        'prop_99': prop_99
    }
    for k, v in results.items():
        print(k,':', v)
    data = scaler.inverse_transform(data)
    data_missing[na_ind] = means
    data_missing = scaler.inverse_transform(data_missing)
    differences = np.abs(data[na_ind] - data_missing[na_ind])
    MAE = np.mean(differences)
    results['multi_mae'] = MAE
    print('average absolute error:', MAE)
    return results

def get_scaled_data(return_scaler=False, put_nans_back=False):
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
    del data_missing_complete
    data_missing[na_ind] = 0
    data_missing = sc.transform(data_missing)
    data = np.array(np.copy(data[:,4:]),dtype='float64')
    data = sc.transform(data)
    if put_nans_back:
        data_missing[na_ind] = np.nan
    if return_scaler:
        return data, data_missing, sc
    else:
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

class DataMissingMaker:
    def __init__(self, complete_only, prop_miss_rows=1, prop_miss_col=0.1):
        self.data = complete_only
        self.n_col = self.data.shape[1]
        self.prop_miss_rows = prop_miss_rows
        self.prop_miss_col = prop_miss_col
        self.n_rows_to_null = int(len(complete_only) * prop_miss_rows)


    def get_random_col_selection(self):
        n_cols_to_null = np.random.binomial(n=self.n_col, p=self.prop_miss_col)
        return np.random.choice(range(self.n_col), n_cols_to_null, replace=False)

    def generate_missing_data(self):
        random_rows = np.random.choice(range(len(self.data)), self.n_rows_to_null, replace=False)
        null_col_indexes = [self.get_random_col_selection() for _ in range(self.n_rows_to_null)]
        null_row_indexes = [np.repeat(row, repeats=len(null_col_indexes[i])) for i, row in enumerate(random_rows)]
        null_col_indexes = np.array([inner[j] for inner in null_col_indexes for j in range(len(inner))]) # flatten the nested arrays
        null_row_indexes = np.array([inner[j] for inner in null_row_indexes for j in range(len(inner))]) # flatten the nested arrays
        new_masked_x = np.copy(self.data)
        new_masked_x[null_row_indexes, null_col_indexes] = np.nan
        return new_masked_x



if __name__=="__main__":
    evaluate_coverage()


