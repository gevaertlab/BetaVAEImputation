import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencodersbetaVAE import VariationalAutoencoder
import pandas as pd
import random
import tensorflow as tf
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json', help='configuration json file')

if __name__ == '__main__':
    
        args = parser.parse_args()
        with open(args.config) as f:
            config = json.load(f)
    
        training_epochs=config["traning_epochs"] #250
        batch_size=config["batch_size"] #250
        learning_rate=config["learning_rate"] #0.0005
        latent_size = config["latent_size"] #200    
        hidden_size_1=config["latent_size_1"]
        hidden_size_2=config["latent_size_2"]
        beta=config["beta"]   
            
        DataPath =   config["data_path"]     
        CorruptDataPath = config["corrupt_data_path"]
        SaveRoot = config["save_rootpath"]
        
        # LOAD DATA
        Xdata_df = pd.read_csv(DataPath)
        Xdata = Xdata_df.values
        del Xdata_df
        
        # Load data with missing values from a csv for analysis:
        Xdata_df = pd.read_csv(CorruptDataPath)
        Xdata_Missing = Xdata_df.values
        del Xdata_df
        
        # Properties of data:
        n_x = Xdata_Missing.shape[1] # dimensionality of data space
        ObsRowInd = np.where(np.isfinite(np.sum(Xdata_Missing,axis=1)))
        NanRowInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=1)))
        NanIndex = np.where(np.isnan(Xdata_Missing))

        sc = StandardScaler()
        Xdata_Missing_complete = np.copy(Xdata_Missing[ObsRowInd[0],:])
        sc.fit(Xdata_Missing_complete)
        Xdata_Missing[NanIndex] = 0
        Xdata_Missing = sc.transform(Xdata_Missing)
        Xdata_Missing[NanIndex] = np.nan
        del Xdata_Missing_complete
        Xdata = sc.transform(Xdata)

       
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
                 n_input=n_x, # data input size
                 n_z=latent_size)  # dimensionality of latent space
        
        # initialise VAE:
        vae = TFVariationalAutoencoder(network_architecture,
                                     learning_rate=learning_rate, 
                                     batch_size=batch_size,istrain=True,restore_path=None,
                                     beta=beta)
        
        # train VAE on corrupted data:
        vae = vae.train(XData=Xdata_Missing,
                        training_epochs=training_epochs)
                
        saver = tf.train.Saver()
        save_path = saver.save(vae.sess, SaveRoot+"ep"+str(training_epochs)+"_bs"+str(batch_size)+"_lr"+str(learning_rate)+"_bn"+str(latent_size)+"_cl"+corrupt_level+"_opADAM"+"_beta"+str(beta)+"_betaVAE"+".ckpt")
        
        
        
        
        
        
        
        
        
        
        
        
        
