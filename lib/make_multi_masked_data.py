import numpy as np
import pandas as pd

"""
This module creates multiple repeats of the same dataset - each one with 
different random missing data.
This is a very inefficient way to do this and a much better way would be 
to create a pre-processing step that masks a random proportion of the data 
with each batch.
"""
data_path =  "../data/data_complete.csv"
# corrupt_data_path = "./data/LGGGBM_missing_10perc_trial1.csv"

data = pd.read_csv(data_path).values
data = np.delete(data, np.s_[0:4], axis=1)

prop_missing_patients = 0.2
prop_missing_features = 0.1

n_splits = 10
n_cols = data.shape[1]
n_rows_to_add_null = int(len(data) * prop_missing_patients)
n_cols_to_null = int(n_cols * prop_missing_features)

def get_random_col_selection():
    return np.random.choice(range(n_cols), n_cols_to_null, replace=False)

train_data = []
for _ in range(n_splits):
    random_rows  = np.random.choice(range(len(data)), n_rows_to_add_null, replace=False)
    null_row_indexes = np.array([np.repeat(i, repeats=n_cols_to_null) for i in random_rows]).flatten()
    null_col_indexes = np.array([get_random_col_selection() for _ in range(n_rows_to_add_null)]).flatten()
    masked_data = np.copy(data)
    masked_data[null_row_indexes, null_col_indexes] = 0
    train_data.append(masked_data)


reshaped_train = np.array(train_data)
new_n_rows = reshaped_train.shape[1] * n_splits
reshaped_train = reshaped_train.reshape((new_n_rows, n_cols))
pd.DataFrame(reshaped_train).to_csv('../data/repeated_masked.csv', index=False)


