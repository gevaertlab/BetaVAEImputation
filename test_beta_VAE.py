import os
try:
        os.chdir("git_repository/BetaVAEImputation")
except FileNotFoundError:
        pass

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencodersbetaVAE import VariationalAutoencoder
import pandas as pd
import random
import tensorflow as tf
import sys
import pickle
from sklearn.decomposition import KernelPCA
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='example_config_VAE.json', help='configuration json file')
tf.reset_default_graph()



if __name__ == '__main__':
    
       
        args = parser.parse_args()
        with open(args.config) as f:
            config = json.load(f)

        #with open("example_config_VAE.json") as f:
        #    config = json.load(f)
    
        training_epochs=config["training_epochs"] #250
        batch_size=config["batch_size"] #250
        learning_rate=config["learning_rate"] #0.0005
        latent_size = config["latent_size"] #200    
        hidden_size_1=config["hidden_size_1"]
        hidden_size_2=config["hidden_size_2"]
        beta=config["beta"]   
            
        data_path =   config["data_path"]     
        corrupt_data_path = config["corrupt_data_path"]
        restore_root = config["save_rootpath"]
        trial_ind = config ["trial_ind"]
        rp=restore_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"_betaVAE"+".ckpt"
        
        print("restore path: ", rp)
        
        # LOAD DATA
        data= pd.read_csv(data_path).values
        data_missing = pd.read_csv(corrupt_data_path).values

        
        n_row = data_missing.shape[1] # dimensionality of data space
        non_missing_row_ind= np.where(np.isfinite(np.sum(data_missing,axis=1)))
        na_ind = np.where(np.isnan(data_missing))
        na_count= len(na_ind[0])
       
        sc = StandardScaler()
        data_missing_complete = np.copy(data_missing[non_missing_row_ind[0],:])
        sc.fit(data_missing_complete)
        data_missing[na_ind] = 0
        #Scale the testing data with model's trianing data mean and variance
        data_missing = sc.transform(data_missing)
        data_missing[na_ind] = np.nan
        del data_missing_complete
        
        # Remove strings and metadata from first few columns in data
        data = np.delete(data,np.s_[0:4], axis=1)
        data = sc.transform(data)
        
        
        # VAE network size:
        Decoder_hidden1 = hidden_size_1 #6000
        Decoder_hidden2 = hidden_size_2 #2000
        Encoder_hidden1 = hidden_size_2 #2000
        Encoder_hidden2 = hidden_size_1 #6000

        # specify number of imputation iterations:
        ImputeIter = 3
        
        # define dict for network structure:
        network_architecture = \
            dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
                 n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
                 n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
                 n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
                 n_input=n_row, # data input size
                 n_z=latent_size)  # dimensionality of latent space
        
        # initialise VAE:
        vae = VariationalAutoencoder(network_architecture,
                                     learning_rate=learning_rate, 
                                     batch_size=batch_size,istrain=False,restore_path=rp,beta=beta)


        # Wrote a function within autoencodersbetaVAE.py to extract the z space, let's see what it does
        tmp = vae.get_z_distribution
        # attempt to extract z from read-in vae object
        z_space = vae.z
        z_mean = vae.z_mean
        z_log_sigma_sq = vae.z_log_sigma_sq

        #initialize the variable
        init = tf.compat.v1.global_variables_initializer()

        #run the graph
        with tf.compat.v1.InteractiveSession() as sess:
            sess.run(init) #execute init
            #print the random values that we sample
            print(sess.run(z_space, 
                             feed_dict={'x': data}))


        ## Now we go into autoencodersbetaVAE.py and try and deconstruct the impute function
        max_iter = ImputeIter

        data_impute, convergence = vae.impute(data_corrupt = data_missing, max_iter = ImputeIter)
        
        data = sc.inverse_transform(data)
        data_impute = sc.inverse_transform(data_impute)
        ReconstructionError = sum(((data_impute[na_ind] - data[na_ind])**2)**0.5)/na_count
        print('Reconstruction error (VAE):')
        print(ReconstructionError)
        np.savetxt("./imputed_data_trial_"+str(trial_ind)+"_VAE.csv", data_impute, delimiter=",")
        # np.savetxt("./imputed_data_trial_"+str(trial_ind)+"_VAE.csv", data_impute, delimiter=",")

        
    
        
