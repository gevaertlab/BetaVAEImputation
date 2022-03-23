import numpy as np
import pandas as pd

data_path =  "./data/data_complete.csv",
corrupt_data_path = "./data/LGGGBM_missing_10perc_trial1.csv"

data = pd.read_csv(data_path).values
data_missing = pd.read_csv(corrupt_data_path).values

prop_missing_patients = 0.2
prop_missing_features = 0.1

n_splits = 10
n_rows_to_add_null = int(len(data) * prop_missing_patients)
n_cols_to_null = int(data.shape[1] * prop_missing_features)
random_rows  = np.random.choice(range(len(data)), n_rows_to_add_null, replace=False)
random_cols = np.random.choice(range(data.shape[1]), n_cols_to_null, replace=False)
