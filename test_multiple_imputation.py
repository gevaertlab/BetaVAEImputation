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
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
tf.reset_default_graph()
config_path = parser.parse_args().config

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

        max_iter = config["n_iterations"]
            
        data_path = config["data_path"]     
        corrupt_data_path = config["corrupt_data_path"]
        restore_root = config["save_rootpath"]
        trial_ind = config ["trial_ind"]
        rp=restore_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"_betaVAE"+".ckpt"
        print("restore path: ", rp)

        # Define output directories for convergence metrics, imputed datasets and NA index statistics
        imp_out=restore_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"/multiple_imputation/imputed_datasets/"
        conv_out=restore_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"/multiple_imputation/convergence_plots/"
        na_out=restore_root+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_opADAM"+"_beta"+str(beta)+"/multiple_imputation/NA_indices/"
        
	    # if these directories don't exist, make them
        os.makedirs(imp_out, exist_ok=True)
        os.makedirs(conv_out, exist_ok=True)
        os.makedirs(na_out, exist_ok=True)

        # LOAD DATA
        print("Loading data in...")
        data= pd.read_csv(data_path).values
        data_missing = pd.read_csv(corrupt_data_path).values

        
        n_col = data_missing.shape[1] # dimensionality of data space
        non_missing_row_ind= np.where(np.isfinite(np.sum(data_missing,axis=1)))
        na_ind = np.where(np.isnan(data_missing))
        na_count= len(na_ind[0])
       
        print("Scaling data...")
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
        
        data_missing2 = np.copy(data_missing)
        
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
                 n_input=n_col, # data input size
                 n_z=latent_size)  # dimensionality of latent space

        
        # Generate m plausible datasets via impute_multiple() function

        print("Reading in trained model...")
        print("Training epochs:", training_epochs, "batch size:", batch_size, "learning rate:", learning_rate, "beta:", beta, "corrupt data path:", corrupt_data_path)
        # Let's do the same with multiple imputation
        vae_mult = VariationalAutoencoder(network_architecture,
                                     learning_rate=learning_rate, 
                                     batch_size=batch_size,istrain=False,restore_path=rp,beta=beta)

        # Let's run a for loop where we copy data_missing2 at the beginning and feed that into impute_multiple()
        m = int(1) # number of imputed datasets
        mult_imp_datasets = []
        mult_convs = []
        mult_convs_lik = []
        mult_largest_imp_vals = []
        mult_avg_imp_vals = []
        print("Beginning imputation of", m, "plausible dataset(s) with", max_iter, "imputation iterations")
        for i in range(m):
            print("Generating plausible dataset", i+1)

            data_missing_mult = np.copy(data_missing2)
            mult_imputed_data, mult_conv, mult_conv_lik, mult_largest_imp_val, mult_avg_imp_val, mean_sigma_sq = \
                vae_mult.impute_multiple(data_corrupt = data_missing_mult, max_iter = max_iter)

            # Add to list
            mult_imp_datasets.append(np.copy(mult_imputed_data))
            mult_convs.append(np.copy(mult_conv))
            mult_convs_lik.append(np.copy(mult_conv_lik))
            mult_largest_imp_vals.append(np.copy(mult_largest_imp_val))
            mult_avg_imp_vals.append(np.copy(mult_avg_imp_val))

        # Check each plausible dataset is unique
        mult_imp_datasets[0] == mult_imp_datasets[1] # good!

        print("Difference between previous iteration NA values and current:", mult_convs) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        for i in range(m):
            iter = list(range(1,max_iter+1))
            plt.plot(iter, mult_convs[i], 'ro')
            plt.axis([0,max_iter+1,0,max(mult_convs[i])+0.1])
            plt.ylabel('convergence')
            plt.savefig(conv_out+'Difference_in_previous_iteration'+'_dataset_'+str(i+1)+'.png')
            plt.close()

        print("Likelihood at NA indices:", mult_convs_lik) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        for i in range(m):
            iter = list(range(1,max_iter+1))
            plt.plot(iter, mult_convs_lik[i], 'ro')
            plt.axis([0,max_iter+1,0,max(mult_convs_lik[i])+0.1])
            plt.ylabel('Negative log likelihood')
            plt.savefig(conv_out+'Negative_log_likelihood'+'_dataset_'+str(i+1)+'.png') 
            plt.close()

        print("Largest imputed values at each iteration:", mult_largest_imp_vals) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        for i in range(m):
            iter = list(range(1,max_iter+1))
            plt.plot(iter, mult_largest_imp_vals[i], 'ro')
            plt.axis([0,max_iter+1,0,max(mult_largest_imp_vals[i])+0.1])
            plt.ylabel('largest imputed value (abs val)')
            plt.savefig(conv_out+'Largest_imputed_value'+'_dataset_'+str(i+1)+'.png')
            plt.close()
        

        print("Average imputed values at each iteration:", mult_avg_imp_vals) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        for i in range(m):
            iter = list(range(1,max_iter+1))
            plt.plot(iter, mult_avg_imp_vals[i], 'ro')
            plt.axis([0,max_iter+1,0,max(mult_avg_imp_vals[i])+0.1])
            plt.ylabel('average imputed value (abs val)')
            plt.savefig(conv_out+'Average_imputed_value'+'_dataset_'+str(i+1)+'.png')
            plt.close()

        if max_iter==100:
            np.savetxt('./MI_dataset_100_iterations_LGGGBM_trial1_10prctmissing.csv', mult_imp_datasets[0], delimiter=",")

        # Multiple imputation setting
        # First of all we need to inverse transform all plausible datasets and then compute an average across for our final dataset
        # Can also plot the reconstruction error of each in a boxplot?
        data = sc.inverse_transform(data)
        reconstr_error = []
        for i in range(m):
            print("Transforming imputed dataset", i+1)
            mult_imp_datasets[i] = sc.inverse_transform(mult_imp_datasets[i]) 
            print("Computing reconstruction error for dataset", i+1)
            reconstr_error.append(sum(((mult_imp_datasets[i][na_ind] - data[na_ind])**2)**0.5)/na_count)

        plt.boxplot(reconstr_error)
        plt.ylabel("reconstruction error")
        plt.savefig(na_out+'Reconstruction_error_boxplot.png')
        plt.close()

        for i in range(m):
            print("Saving plausible dataset", i+1, "to", imp_out)
            np.savetxt(imp_out+"./mult_imputed_data_trial_"+str(trial_ind)+"_dataset_"+str(i+1)+"_VAE.csv", mult_imp_datasets[i], delimiter=",")  

        for i in range(m):
            print('Reconstruction error on multiple imputation (VAE) for dataset', i+1, reconstr_error[i])

        np.savetxt(na_out+'reconstruction_error.csv', reconstr_error, delimiter=',')

        # Compute confidence intervals for each NA index
        na_indices = pd.DataFrame(
            {'plausible_dataset_1': mult_imp_datasets[0][na_ind],
             'plausible_dataset_2': mult_imp_datasets[1][na_ind],
             'plausible_dataset_3': mult_imp_datasets[2][na_ind],
             'plausible_dataset_4': mult_imp_datasets[3][na_ind],
             'plausible_dataset_5': mult_imp_datasets[4][na_ind],
             'plausible_dataset_6': mult_imp_datasets[5][na_ind],
             'plausible_dataset_7': mult_imp_datasets[6][na_ind],
             'plausible_dataset_8': mult_imp_datasets[7][na_ind],
             'plausible_dataset_9': mult_imp_datasets[8][na_ind],
             'plausible_dataset_10': mult_imp_datasets[9][na_ind],
             'actual_values': data[na_ind]
            })

        na_indices.to_csv(na_out+'NA_imputed_values_m_datasets.csv')

