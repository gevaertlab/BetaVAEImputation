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


        DataPath = "./data_complete.csv"
        ClinDataPath="./clinical.csv"
        Xdata= pd.read_csv(DataPath).values
        clin=pd.read_csv(ClinDataPath)

        
        crandom_list=[]
        cknn_list=[]
        csvd_list=[]
        cvae_list=[]
        
        for trial_ind in range(start, end):
            trial_ind=str(trial_ind)
            print("Trial", trial_ind)
                   
            SVDImputedDataPath="./imputed_data_trial_"+tiral_ind+"_SVD.csv"
            KNNImputedDataPath="./imputed_data_trial_"+tiral_ind+"_KNN.csv"
            VAEImputedDataPath="./imputed_data_trial_"+tiral_ind+"_VAE.csv"
            RandomImputedDataPath="./imputed_data_trial_"+tiral_ind+"_Random.csv"    
            CorruptDataPath = "./corrupted_data_trial_"+ trial_ind+".csv"
            
            
            # LOAD DATA
            # Load data from a csv for analysis:
            Xdata_SVDImputed=np.loadtxt(fname=SVDImputedDataPath,delimiter=",")
            Xdata_KNNImputed=np.loadtxt(fname=KNNImputedDataPath,delimiter=",")
            Xdata_VAEImputed=np.loadtxt(fname=VAEImputedDataPath,delimiter=",")
            Xdata_RandomImputed=np.loadtxt(fname=RandomImputedDataPath,delimiter=",")
            Xdata_Missing = pd.read_csv(CorruptDataPath).values
            
           
            ObsRowInd = np.where(np.isfinite(np.sum(Xdata_Missing,axis=1)))
            NanRowInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=1)))
            NanColInd = np.where(np.isnan(np.sum(Xdata_Missing,axis=0)))[0]
           
            NoHistoRowInd=np.where((clin.Grade!='G2')&(clin.Grade!='G3')&(clin.Grade!='G4'))
            TestInd=np.asarray([e for e in NanRowInd[0] if e not in NoHistoRowInd[0]])
            TrainInd=np.asarray([e for e in ObsRowInd[0] if e not in NoHistoRowInd[0]])
    
            train=Xdata_Missing[TrainInd,]
            test=Xdata[TestInd,]
            KNN_test=Xdata_KNNImputed[TestInd,]
            SVD_test=Xdata_SVDImputed[TestInd,]
            VAE_test=Xdata_VAEImputed[TestInd,]
            Random_test=Xdata_RandomImputed[TestInd,]
    
            clin_train=clin.iloc[TrainInd,]
            clin_test=clin.iloc[TestInd,]
            
            
            grade_ordinal_train=np.zeros(clin_train.shape[0])
            grade_ordinal_train[np.where(clin_train.Grade=='G2')]=2
            grade_ordinal_train[np.where(clin_train.Grade=='G3')]=3
            grade_ordinal_train[np.where(clin_train.Grade=='G4')]=4
            
            grade_ordinal_test=np.zeros(clin_test.shape[0])
            grade_ordinal_test[np.where(clin_test.Grade=='G2')]=2
            grade_ordinal_test[np.where(clin_test.Grade=='G3')]=3
            grade_ordinal_test[np.where(clin_test.Grade=='G4')]=4
            
                
            
            coef_list1=[]
            p_list1=[]
    
            for i in range(test.shape[1]):
            #for i in range(5):
                coef, p = spearmanr(test[:,i], grade_ordinal_test)
                coef_list1.append(coef)
                p_list1.append(p)
                if i%5000==0:
                    print (i)
            
            coef_list2=[]
            p_list2=[]     
            
            for i in range(test.shape[1]):
            #for i in range(5):
                coef, p = spearmanr(SVD_test[:,i], grade_ordinal_test)
                coef_list2.append(coef)
                p_list2.append(p)
                if i%5000==0:
                    print (i)
            
            coef_list3=[]
            p_list3=[]   
            for i in range(test.shape[1]):
            #for i in range(5):
                coef, p = spearmanr(KNN_test[:,i], grade_ordinal_test)
                coef_list3.append(coef)
                p_list3.append(p)
                if i%5000==0:
                    print (i)
    
            coef_list4=[]
            p_list4=[]   
            for i in range(test.shape[1]):
            #for i in range(5):
                coef, p = spearmanr(VAE_test[:,i], grade_ordinal_test)
                coef_list4.append(coef)
                p_list4.append(p)
                if i%5000==0:
                    print (i)
                    
                    
            coef_list5=[]
            p_list5=[]     
            
            for i in range(test.shape[1]):
            #for i in range(5):
                coef, p = spearmanr(Random_test[:,i], grade_ordinal_test)
                coef_list5.append(coef)
                p_list5.append(p)
                if i%5000==0:
                    print (i)
    
    
            coef_list1_arr=np.asarray(coef_list1)[NanColInd.astype(int) ]
            coef_list2_arr=np.asarray(coef_list2)[NanColInd.astype(int) ]
            coef_list3_arr=np.asarray(coef_list3)[NanColInd.astype(int) ]
            coef_list4_arr=np.asarray(coef_list4)[NanColInd.astype(int) ]
            coef_list5_arr=np.asarray(coef_list5)[NanColInd.astype(int) ]
            
            diff_SVD=np.asarray(coef_list2_arr)-np.asarray(coef_list1_arr)
            diff_KNN=np.asarray(coef_list3_arr)-np.asarray(coef_list1_arr)
            diff_VAE=np.asarray(coef_list4_arr)-np.asarray(coef_list1_arr)
            diff_Random=np.asarray(coef_list5_arr)-np.asarray(coef_list1_arr)
            true_coeff=np.asarray(coef_list1_arr)
            
            crandom=CIndex(coef_list2_arr, coef_list1_arr)
            cknn=CIndex(coef_list3_arr, coef_list1_arr)
            csvd=CIndex(coef_list2_arr, coef_list1_arr)
            cvae=CIndex(coef_list4_arr, coef_list1_arr)

            crandom_list.append(crandom)
            cknn_list.append(cknn)
            csvd_list.append(csvd)
            cvae_list.append(cvae)
            
            np.savetxt("./spearmanr_cindex_trial_"+str(start)+"-"+str(end-1)+"_Random.csv", np.asarray(crandom_list), delimiter=",")  
            np.savetxt("./spearmanr_cindex_trial_"+str(start)+"-"+str(end-1)+"_KNN.csv", np.asarray(cknn_list), delimiter=",")  
            np.savetxt("./spearmanr_cindex_trial_"+str(start)+"-"+str(end-1)+"_SVD.csv", np.asarray(csvd_list), delimiter=",")  
            np.savetxt("./spearmanr_cindex_trial_"+str(start)+"-"+str(end-1)+"_VAE.csv",np.asarray(cvae_list), delimiter=",")              

            np.savetxt("./spearmanr_diff_trial_"+str(start)+"-"+str(end-1)+"_Random.csv",np.asarray(diff_Random), delimiter=",")  
            np.savetxt("./spearmanr_diff_trial_"+str(start)+"-"+str(end-1)+"_KNN.csv", np.asarray(diff_KNN), delimiter=",")            
            np.savetxt("./spearmanr_diff_trial_"+str(start)+"-"+str(end-1)+"_SVD.csv", np.asarray(diff_SVD), delimiter=",")  
            np.savetxt("./spearmanr_diff_trial_"+str(start)+"-"+str(end-1)+"_VAE.csv", np.asarray(diff_VAE), delimiter=",")  
            np.savetxt("./spearmanr_diff_trial_"+str(start)+"-"+str(end-1)+"_true_coeff.csv", np.asarray(true_coeff), delimiter=",")  

            
            