from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from fancyimpute import IterativeSVD
import os
import pandas as pd
import numpy as np
import random  
import sys
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')



if __name__ == '__main__':
    
        args = parser.parse_args()
        with open(args.config) as f:
            config = json.load(f)

        data_path =   config["data_path"]     #Ground truth data
        corrupt_data_path = config["corrupt_data_path"] #Data containing missing values
        rank = config["rank"]
        trial_ind = config["trial_ind"]

       # LOAD DATA
        data= pd.read_csv(data_path).values
        data_missing = pd.read_csv(corrupt_data_path).values

        
        n_row = data_missing.shape[1] # dimensionality of data space
        non_missing_row_ind= np.where(np.isfinite(np.sum(data_missing,axis=1)))
        na_ind = np.where(np.isnan(data_missing))
        na_count= len(na_ind[0])
      
        data_impute_SVD=IterativeSVD(rank=rank,convergence_threshold=0.0005,max_iters=16).fit_transform(data_missing)
        ReconstructionErrorSVD = sum(((data_impute_SVD[na_ind] - data[na_ind])**2)**0.5)/na_count
        print('Reconstruction error (VAE):')
        print(ReconstructionErrorSVD) 
            
        np.savetxt("./imputed_data_trial_"+str(trial_ind)+"_SVD.csv", data_impute_SVD, delimiter=",")  
    