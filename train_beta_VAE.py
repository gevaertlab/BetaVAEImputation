import os
os.chdir("git_repository/BetaVAEImputation")
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencodersbetaVAE import VariationalAutoencoder
import pandas as pd
import random
import tensorflow as tf
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

if __name__ == '__main__':
    
        #args = parser.parse_args()
        #with open(args.config) as f:
        #    config = json.load(f)

        with open("example_config_VAE.json") as f:
            config = json.load(f)

    
        training_epochs=config["training_epochs"] #250
        batch_size=config["batch_size"] #250
        learning_rate=config["learning_rate"] #0.0005
        latent_size = config["latent_size"] #200    
        hidden_size_1=config["hidden_size_1"]
        hidden_size_2=config["hidden_size_2"]
        beta=config["beta"]            
        data_path =   config["data_path"]     
        corrupt_data_path = config["corrupt_data_path"]
        save_root = config["save_rootpath"]
        
        # Read in complete data
        data = pd.read_csv(data_path).values
        # Read in simulated missing dataset
        data_missing = pd.read_csv(corrupt_data_path).values

        # How many genes do we have? ie. what is the dimensiontality of Yobs?
        n_row = data_missing.shape[1] # dimensionality of data space
        # Store the index of each sample that is complete
        non_missing_row_ind= np.where(np.isfinite(np.sum(data_missing,axis=1)))
        # Store the rows and columns of every missing data point in your "data_missing" numpy array
        na_ind = np.where(np.isnan(data_missing))

        sc = StandardScaler()
        # Create a new numpy array that is complete (subset of simulated data_missing)
        data_missing_complete = np.copy(data_missing[non_missing_row_ind[0],:])
        # Find scaling factors from the complete set of the simulated missing data
        sc.fit(data_missing_complete)
        data_missing[na_ind] = 0 # Assign zero values to missing value indicies
        # Transform missing data by the scaling factors defined from all complete values
        data_missing = sc.transform(data_missing)
        # Re-assign the missing values to the same positions as before
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
                                     batch_size=batch_size,istrain=True,restore_path=None,
                                     beta=beta)
        
        # train VAE on corrupted data:
        vae = vae.train(data=data_missing,
                        training_epochs=training_epochs)

        # What does x_hat_distribution look like?
        tmp = vae.test_sampling(data_corrupt = data_missing)
        # Looks like this is a Tensor, not a dataframe

        # Test updates to VAE object class
        # specify number of imputation iterations:
        ImputeIter = 3

        data_imputed_sample = vae.impute_multiple(data_corrupt = data_missing, max_iter = ImputeIter)
        
        saver = tf.train.Saver()
        save_path = saver.save(vae.sess, save_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"_betaVAE"+".ckpt")
        
        
        
        
        
        
        
        
        
        
        
        
        
