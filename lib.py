import numpy as np
import pandas as pd
import json
with open("example_config_VAE.json") as f:
    config = json.load(f)
data_path = config["data_path"]
corrupt_data_path = config["corrupt_data_path"]
from sklearn.preprocessing import StandardScaler

def get_scaled_data():
    data = pd.read_csv(data_path).values
    data_missing = pd.read_csv(corrupt_data_path).values
    non_missing_row_ind = np.where(np.isfinite(np.sum(data_missing, axis=1)))
    na_ind = np.where(np.isnan(data_missing))
    sc = StandardScaler()
    data_missing_complete = np.copy(data_missing[non_missing_row_ind[0], :])
    sc.fit(data_missing_complete)
    data_missing[na_ind] = 0
    data_missing = sc.transform(data_missing)
    del data_missing_complete
    # data = np.delete(data, np.s_[0:4], axis=1)
    data = np.array(np.copy(data[:,4:]),dtype='float64')
    return data, data_missing


