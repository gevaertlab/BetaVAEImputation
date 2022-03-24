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
import pickle
from sklearn.decomposition import KernelPCA
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')
tf.reset_default_graph()


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
        
        data_missing2 = np.copy(data_missing)
        
        # VAE network size:
        Decoder_hidden1 = hidden_size_1 #6000
        Decoder_hidden2 = hidden_size_2 #2000
        Encoder_hidden1 = hidden_size_2 #2000
        Encoder_hidden2 = hidden_size_1 #6000

        # specify number of imputation iterations:
        ImputeIter = 100 # looks like both strategies converge around 4 iterations
        
        # define dict for network structure:
        network_architecture = \
            dict(n_hidden_recog_1=Encoder_hidden1, # 1st layer encoder neurons
                 n_hidden_recog_2=Encoder_hidden2, # 2nd layer encoder neurons
                 n_hidden_gener_1=Decoder_hidden1, # 1st layer decoder neurons
                 n_hidden_gener_2=Decoder_hidden2, # 2nd layer decoder neurons
                 n_input=n_row, # data input size
                 n_z=latent_size)  # dimensionality of latent space
        
        # initialise VAE:
        #vae = VariationalAutoencoder(network_architecture,
        #                             learning_rate=learning_rate, 
        #                             batch_size=batch_size,istrain=False,restore_path=rp,beta=beta)

        
        ## Now we go into autoencodersbetaVAE.py and try and deconstruct the impute function
        max_iter = ImputeIter

        # How can we look at the convergence of these random samples?
        # What we can do is look at the location of every single na index for each "iteration"
        # compute the difference between current and previous iteration, take the absolute value and then average across all na indices
        # To do this, we need to add into the function impute_multiple a way to store this value ^ into a list and then plot
        #imputed_data, conv = vae.impute(data_corrupt = data_missing, max_iter = max_iter)

        #print(conv) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        #iter = list(range(1,max_iter+1))
        #plt.plot(iter, conv, 'ro')
        #plt.axis([0,max_iter+1,0,max(conv)+0.05])
        #plt.ylabel('convergence')
        #plt.show()

        # Generate m plausible datasets via impute_multiple() function

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
        for i in range(m):
            print("Generating plausible dataset", i+1)

            data_missing_mult = np.copy(data_missing2)
            mult_imputed_data, mult_conv, mult_conv_lik, mult_largest_imp_val, mult_avg_imp_val = vae_mult.impute_multiple(data_corrupt = data_missing_mult, max_iter = max_iter)

            # Add to list
            mult_imp_datasets.append(np.copy(mult_imputed_data))
            mult_convs.append(np.copy(mult_conv))
            mult_convs_lik.append(np.copy(mult_conv_lik))
            mult_largest_imp_vals.append(np.copy(mult_largest_imp_val))
            mult_avg_imp_vals.append(np.copy(mult_avg_imp_val))

        # Check each plausible dataset is unique
        mult_imp_datasets[0] == mult_imp_datasets[1] # good!

        print(mult_convs) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        iter = list(range(1,max_iter+1))
        plt.plot(iter, mult_convs[0], 'ro')
        plt.axis([0,max_iter+1,0,max(mult_conv[0])+0.1])
        plt.ylabel('convergence')
        plt.show()

        print(mult_convs_lik) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        iter = list(range(1,max_iter+1))
        plt.plot(iter, mult_convs_lik[0], 'ro')
        plt.axis([0,max_iter+1,0,max(mult_conv_lik[0])+0.1])
        plt.ylabel('negative log likelihood')
        plt.show() 

        print(mult_largest_imp_vals) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        iter = list(range(1,max_iter+1))
        plt.plot(iter, mult_largest_imp_vals[0], 'ro')
        plt.axis([0,max_iter+1,0,max(mult_largest_imp_vals[0])+0.1])
        plt.ylabel('largest imputed value (abs val)')
        plt.show()  

        print(mult_avg_imp_vals) # Looks right! and looks like it is going down. Let's see what this looks like in graph form.
        iter = list(range(1,max_iter+1))
        plt.plot(iter, mult_avg_imp_vals[0], 'ro')
        plt.axis([0,max_iter+1,0,max(mult_avg_imp_vals[0])+0.1])
        plt.ylabel('largest imputed value (abs val)')
        plt.show()    

        if max_iter==100:
            np.savetxt('./MI_dataset_100_iterations_LGGGBM_trial1_10prctmissing.csv', mult_imp_datasets[0], delimiter=",")
        
        # Single imputation setting
        #data = sc.inverse_transform(data)
        #imputed_data = sc.inverse_transform(imputed_data)
        #ReconstructionError = sum(((imputed_data[na_ind] - data[na_ind])**2)**0.5)/na_count
        #print('Reconstruction error on single imputation (VAE):')
        #print(ReconstructionError)
        #np.savetxt("./imputed_data_trial_"+str(trial_ind)+"_VAE.csv", imputed_data, delimiter=",") 

        # Export results from single imputation
        #na_indices_single = pd.DataFrame(
        #    {'imputed_dataset': imputed_data[na_ind],
        #     'actual_values': data[na_ind]
        #    })

        #na_indices_single.to_csv('NA_imputed_values_single_imputation.csv')

        # Multiple imputation setting
        # First of all we need to inverse transform all plausible datasets and then compute an average across for our final dataset
        # Can also plot the reconstruction error of each in a boxplot?
        reconstr_error = []
        for i in range(m):
            print("Transforming imputed dataset", i+1)
            mult_imp_datasets[i] = sc.inverse_transform(mult_imp_datasets[i]) 
            print("Computing reconstruction error for dataset", i+1)
            reconstr_error.append(sum(((mult_imp_datasets[i][na_ind] - data[na_ind])**2)**0.5)/na_count)

        plt.boxplot(reconstr_error)
        plt.ylabel("reconstruction error")
        plt.show()

        for i in range(m):
            print("Saving plausible dataset", i+1)
            np.savetxt("./mult_imputed_data_trial_"+str(trial_ind)+"_dataset_"+str(m)+"_VAE.csv", mult_imp_datasets[i], delimiter=",")  

        for i in range(m):
            print('Reconstruction error on multiple imputation (VAE) for dataset', i+1, reconstr_error[i])

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

        na_indices.to_csv('NA_imputed_values_m_datasets.csv')

        




        