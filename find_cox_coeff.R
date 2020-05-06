library(data.table)
library(survival)
library(glmnet)
library(survcomp)
library(RTCGA)
library(RTCGA.clinical)


for (trial_ind in c(1:10)){  
  DataPath = "./data_complete.csv"
  TimePath="./survival_times.csv"
  StatusPath="./survival_ystatus.csv"
  SVDImputedDataPath=paste0("./imputed_data_trial_",tiral_ind,"_SVD.csv")
  KNNImputedDataPath=paste0("./imputed_data_trial_" ,tiral_ind,"_KNN.csv")
  VAEImputedDataPath=paste0("./imputed_data_trial_", tiral_ind,"_VAE.csv")
  RandomImputedDataPath= paste0("./imputed_data_trial_", tiral_ind,"_Random.csv")
  CorruptDataPath =paste0("./corrupted_data_trial_", trial_ind,".csv")

  
  
  Xdata_SVDImputed<-fread(SVDImputedDataPath)
  Xdata_KNNImputed<-fread(KNNImputedDataPath)
  Xdata_VAEImputed<-fread(VAEImputedDataPath)
  Xdata_RandomImputed<-fread(RandomImputedDataPath)
  Xdata<-fread(DataPath)
  Xdata_Missing<-fread(CorruptDataPath)
  ytimes=fread(TimePath)
  ystatus=fread(StatusPath)
  
  
  TestInd<-which(is.na(rowSums(Xdata_Missing)))
  
  missing_test=as.data.frame(Xdata_Missing)[TestInd,]
  ground_truth_test=as.data.frame(Xdata)[TestInd,]
  KNN_test=as.data.frame(Xdata_KNNImputed)[TestInd,]
  SVD_test=as.data.frame(Xdata_SVDImputed)[TestInd,]
  VAE_test=as.data.frame(Xdata_VAEImputed)[TestInd,]
  Random_test=as.data.frame(Xdata_RandomImputed)[TestInd,]
  ytimes_test=as.data.frame(ytimes)[TestInd,]
  ystatus_test=as.data.frame(ystatus)[TestInd,]
  
  
  colnames(ground_truth_test)<-colnames(KNN_test)
  covariates <- colnames(ground_truth_test)
  
  
  univ_formulas <- sapply(covariates,
                          function(x) as.formula(paste('Surv(ytimes_test, ystatus_test)~', x)))
  
  
  #####
  univ_models <- lapply(univ_formulas, function(x){coxph(x, data = ground_truth_test)})
  feature_size=length(univ_models)
  coef_list=list()
  for (i in c(1:feature_size)){
    coef_list[i]<-univ_models[[i]]$coef  
  }
  write.csv(coef_list, paste0("./coxph_coef_trial_",trial_ind,"_groundtruth.csv"), row.names=FALSE)
  
  
  #####
  univ_models <- lapply(univ_formulas, function(x){coxph(x, data = KNN_test)})
  feature_size=length(univ_models)
  coef_list1=list()
  for (i in c(1:feature_size)){
    coef_list1[i]<-univ_models[[i]]$coef  
  }
  write.csv(coef_list1, paste0("./coxph_coef_trial_",trial_ind,"_KNN.csv"), row.names=FALSE)
  
  #####
  univ_models <- lapply(univ_formulas, function(x){coxph(x, data = SVD_test)})
  feature_size=length(univ_models)
  coef_list2=list()
  for (i in c(1:feature_size)){
    coef_list2[i]<-univ_models[[i]]$coef  
  }
  write.csv(coef_list2, paste0("./coxph_coef_trial_",trial_ind,"_SVD.csv"), row.names=FALSE)
  
  #####
  univ_models <- lapply(univ_formulas, function(x){coxph(x, data = VAE_test)})
  feature_size=length(univ_models)
  coef_list3=list()
  for (i in c(1:feature_size)){
    coef_list3[i]<-univ_models[[i]]$coef  
  }
  write.csv(coef_list3, paste0("./coxph_coef_trial_",trial_ind,"_VAE.csv"), row.names=FALSE)
  
  #####
  univ_models <- lapply(univ_formulas, function(x){coxph(x, data = Random_test)})
  feature_size=length(univ_models)
  coef_list4=list()
  for (i in c(1:feature_size)){
    coef_list4[i]<-univ_models[[i]]$coef  
  }
  write.csv(coef_list4, paste0("./coxph_coef_trial_",trial_ind,"_Random.csv"), row.names=FALSE)                          
  
  #####
  missing_col_ind<-which(is.na(colSums(Xdata_Missing)))
  write.csv(missing_col_ind, paste0("./missing_colind_trail_",trial_ind,".csv"), row.names=FALSE)
  
  
}