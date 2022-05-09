import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def evaluate_coverage(truevals=None, impvals = None):
    means = np.mean(impvals, axis=1)
    st_devs = np.std(impvals, axis=1)
    differences = np.abs(truevals - means)
    n_deviations = differences / st_devs
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576
    prop_90 = sum(n_deviations < ci_90) / len(n_deviations)
    prop_95 = sum(n_deviations < ci_95) / len(n_deviations)
    prop_99 = sum(n_deviations < ci_99) / len(n_deviations)
    print('prop 90:', prop_90)
    print('prop 95:', prop_95)
    print('prop 99:', prop_99)
    
    differences = np.abs(truevals - means)
    mae = np.mean(differences)
    print('average absolute error:', mae)


if __name__=="__main__":
    from sklearn.preprocessing import StandardScaler

    res = pd.read_csv('/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/pseudo-Gibbs/compiled_NA_indices.csv').values
    
    # Assign first column of values to new variable and then remove it from res
    grtruth = res[:,0]
    imp_dat = res[:,1:]

    evaluate_coverage(truevals=grtruth, impvals = imp_dat)


