import numpy as np
from betaVAEv2 import load_model
from array_dropout_analysis import evaluate_model
from lib.helper_functions import get_scaled_data

model_dir = '../output/new_trained_model/epoch_1000/'
model = load_model(model_dir)

full_complete, full_w_nan, scaler = get_scaled_data(put_nans_back=True, return_scaler=True)
missing_row_ind = np.where(np.isnan(full_w_nan).any(axis=1))[0]
missing_w_nans = full_w_nan[missing_row_ind]
missing_complete = full_complete[missing_row_ind]
na_ind = np.where(np.isnan(missing_w_nans))

results = evaluate_model(model, missing_w_nans, missing_complete, na_ind, scaler)
bp=True
