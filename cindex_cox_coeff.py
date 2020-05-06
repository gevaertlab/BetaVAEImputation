import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from autoencodersVAE import TFVariationalAutoencoder
import pandas as pd
import random
import tensorflow as tf
import sys
from scipy.stats import spearmanr

#####Alternative look at all genes
def CIndex(pred, true):
    concord = 0.
    total = 0.
    pred=np.asarray(pred)
    true=np.asarray(true)
    N_test = true.shape[0]
    for i in range(N_test):
        if i %1000 ==0:
            print (i)
        for j in range(N_test):
            if true[j] > true[i]:
                total = total + 1
                if pred[j] > pred[i]: concord = concord + 1
                elif pred[j] == pred[i]: concord = concord + 0.5

    return(concord/total)  

if __name__ == '__main__':
    
        start=1
        end= 10
        
        crandom_list=[]
        cknn_list=[]
        csvd_list=[]
        cvae_list=[]
        
        for trial_ind in range(start, end):
            trial_ind=str(trial_ind)
            print("Trial", trial_ind)

            coef_true=np.asarray(pd.read_csv("./coxph_coef_trial_"+trial_ind+"_groundtruth.csv")).reshape((-1))
            coef_KNN=np.asarray(pd.read_csv("./coxph_coef_trial_"+trial_ind+"_KNN.csv")).reshape((-1))
            coef_SVD=np.asarray(pd.read_csv("./coxph_coef_trial_"+trial_ind+"_SVD.csv")).reshape((-1))
            coef_VAE=np.asarray(pd.read_csv("./coxph_coef_trial_"+trial_ind+"_VAE.csv")).reshape((-1))
            coef_Random=np.asarray(pd.read_csv("./coxph_coef_trial_"+trial_ind+"_Random.csv")).reshape((-1))
            missing_col_ind= np.asarray(pd.read_csv("./missing_colind_trail_"+trial_ind+".csv")).reshape((-1))
            
            coef_true=coef_true[missing_col_ind-1]
            coef_KNN=coef_KNN[missing_col_ind-1]
            coef_SVD=coef_SVD[missing_col_ind-1]
            coef_VAE=coef_VAE[missing_col_ind-1]
            coef_Random=coef_Random[missing_col_ind-1]
                        
            diff_SVD=coef_SVD-coef_true
            diff_KNN=coef_KNN-coef_true
            diff_VAE=coef_VAE-coef_true 
            diff_Random=coef_Random-coef_true 
            diff_SVD[diff_SVD>1]=0
            diff_SVD[diff_SVD<(-1)]=0
            diff_KNN[diff_KNN>1]=0
            diff_KNN[diff_KNN<(-1)]=0
            diff_VAE[diff_VAE>1]=0
            diff_VAE[diff_VAE<(-1)]=0      
            diff_Random[diff_Random>1]=0
            diff_Random[diff_Random<(-1)]=0                                 
                
            crandom=CIndex(coef_Random, coef_true)
            cknn=CIndex(coef_KNN, coef_true)
            csvd=CIndex(coef_SVD, coef_true)
            cvae=CIndex(coef_VAE, coef_true)

            crandom_list.append(crandom)
            cknn_list.append(cknn)
            csvd_list.append(csvd)
            cvae_list.append(cvae)
            
            np.savetxt("./coeffcox_cindex_trial_"+str(start)+"-"+str(end-1)+"_Random.csv", np.asarray(crandom_list), delimiter=",")  
            np.savetxt("./coeffcox_cindex_trial_"+str(start)+"-"+str(end-1)+"_KNN.csv", np.asarray(cknn_list), delimiter=",")  
            np.savetxt("./coeffcox_cindex_trial_"+str(start)+"-"+str(end-1)+"_SVD.csv", np.asarray(csvd_list), delimiter=",")  
            np.savetxt("./coeffcox_cindex_trial_"+str(start)+"-"+str(end-1)+"_VAE.csv",np.asarray(cvae_list), delimiter=",")              

            np.savetxt("./coeffcox_diff_trial_"+str(start)+"-"+str(end-1)+"_Random.csv",np.asarray(diff_Random), delimiter=",")  
            np.savetxt("./coeffcox_diff_trial_"+str(start)+"-"+str(end-1)+"_KNN.csv", np.asarray(diff_KNN), delimiter=",")            
            np.savetxt("./coeffcox_diff_trial_"+str(start)+"-"+str(end-1)+"_SVD.csv", np.asarray(diff_SVD), delimiter=",")  
            np.savetxt("./coeffcox_diff_trial_"+str(start)+"-"+str(end-1)+"_VAE.csv", np.asarray(diff_VAE), delimiter=",")  
            np.savetxt("./coeffcox_diff_trial_"+str(start)+"-"+str(end-1)+"_true_coeff.csv", np.asarray(coef_true), delimiter=",")  

            
            