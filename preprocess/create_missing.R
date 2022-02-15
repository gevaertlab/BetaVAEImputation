library(data.table)
library(survival)
library(glmnet)
library(survcomp)
library(RTCGA)
library(RTCGA.clinical)
library(plyr)

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/Qiu_et_al/analysis/simulated_missing")
data_raw=fread("../pancan_survdata.csv",header=T)
data_raw<-data_raw[-which(data_raw$time<=0),]
dfSummary <- ddply(data_raw, "admin.disease_code", summarise, count=length(times))

data<- data_raw[which(data_raw$admin.disease_code=="lgg"|data_raw$admin.disease_code=="gbm"),]
fwrite(data, "../data_complete.csv")

##################################
# all values of 5% genes missing
##################################

data<- data_raw[which(data_raw$admin.disease_code=="lgg"|data_raw$admin.disease_code=="gbm"),]
missing_row_perc=0.2
missing_gene_perc=0.05

clin=data[,1:4]
feature=data[,5:17179]
sample_n=dim(feature)[1]
feature_size=dim(feature)[2]
feature_gene<-colnames(feature)

for (SEED in c(1:10)){
  
  set.seed(SEED)
  missing_rowind=sample(1:sample_n,sample_n*missing_row_perc )
  
  set.seed(SEED)
  missing_geng_ind<- sample(length(feature_gene), as.integer(missing_gene_perc*length(feature_gene)))
  target_gene<-feature_gene[missing_geng_ind]
  
  feature_corrupted=feature
  feature_subset=feature[missing_rowind,]
  
  col_ind=which(feature_gene%in%target_gene)
  print(length(col_ind))
  missing_row_missing_ones=matrix(0, length(missing_rowind), feature_size)
  missing_row_missing_ones[,col_ind]=1
  missing_row_nonmissing_ones=1-missing_row_missing_ones
  
  corrupt_subset=missing_row_nonmissing_ones*feature_subset
  feature_corrupted[missing_rowind,]<- corrupt_subset
  feature_corrupted[feature_corrupted==0]<-NA
  
  if (!file.exists("allvalue5percgene")) {
    dir.create("allvalue5percgene", recursive = TRUE)
  }
  
  fwrite(feature_corrupted, paste0("allvalue5percgene/LGGGBM_missing_allvalue5percgene_trial",SEED,".csv"))
}

##################################
# half of lowest 10% genes missing
##################################

data<- data_raw[which(data_raw$admin.disease_code=="lgg"|data_raw$admin.disease_code=="gbm"),]
lowest_percent = 0.1
percent_of_lowest_missing=0.5
missing_row_perc=0.2

clin=data[,1:4]
feature=data[,5:17179]
sample_n=dim(feature)[1]
feature_size=dim(feature)[2]
THRES=quantile(as.matrix(feature), probs = lowest_percent, names=FALSE)

for (SEED in c(1:10)){
  
  set.seed(SEED)
  missing_rowind=sample(1:sample_n,sample_n*missing_row_perc )
  
  feature_corrupted=feature
  feature_subset=feature[missing_rowind,]
  
  set.seed(SEED)
  missing_row_lowest_missing_ones=1*(feature_subset<THRES)
  missing_row_missingones=matrix(rbinom(length(missing_rowind)*feature_size,1,percent_of_lowest_missing), nrow = length(missing_rowind))
  missing_row_final_missing_ones=missing_row_lowest_missing_ones*missing_row_missingones
  missing_row_final_nonmissing_ones=1-missing_row_final_missing_ones
  
  corrupt_subset=missing_row_final_nonmissing_ones*feature_subset
  feature_corrupted[missing_rowind,]<- corrupt_subset
  feature_corrupted[feature_corrupted==0]<-NA
  
  if (!file.exists("missing_halflowest5perc")) {
    dir.create("missing_halflowest5perc", recursive = TRUE)
  }
  
  fwrite(feature_corrupted, paste0("missing_halflowest5perc/LGGGBM_missing_halflowest5perc_trial",SEED,".csv"))
}
##################################
# half of 10% highest GC values missing
##################################

data<- data_raw[which(data_raw$admin.disease_code=="lgg"|data_raw$admin.disease_code=="gbm"),]
highest_percent=0.9
percent_of_highest_missing=0.5
missing_row_perc=0.2

clin=data[,1:4]
feature=data[,5:17179]
sample_n=dim(feature)[1]
feature_size=dim(feature)[2]
gc<- read.csv("../../git_repository/BetaVAEImputation/data/Average_GC.csv", sep = "\t", row.names = 1)
GC_thres=quantile(as.matrix(gc$AV_GC_PCT),probs=highest_percent)
gc_high_gene=gc[gc$AV_GC_PCT>GC_thres,2]
strpfun<-function(x){
  out=strsplit(x, split="\\|")[[1]][1]
  return(out)}
feature_gene<-lapply(colnames(feature), strpfun)


for (SEED in c(1:10)){
  
  set.seed(SEED)
  missing_rowind=sample(1:sample_n,sample_n*missing_row_perc )
  
  feature_corrupted=feature
  feature_subset=feature[missing_rowind,]
  
  set.seed(SEED)
  col_ind=which(feature_gene%in%gc_high_gene)
  missing_row_highest_missing_ones=matrix(0, length(missing_rowind), feature_size)
  missing_row_highest_missing_ones[,col_ind]=1
  missing_row_missingones=matrix(rbinom(length(missing_rowind)*feature_size,1,percent_of_highest_missing), nrow = length(missing_rowind))
  missing_row_final_missing_ones=missing_row_highest_missing_ones*missing_row_missingones
  missing_row_final_nonmissing_ones=1-missing_row_final_missing_ones
  
  corrupt_subset=missing_row_final_nonmissing_ones*feature_subset
  feature_corrupted[missing_rowind,]<- corrupt_subset
  feature_corrupted[feature_corrupted==0]<-NA
  
  if (!file.exists("missing_halfgchighest10perc")) {
    dir.create("missing_halfgchighest10perc", recursive = TRUE)
  }

  fwrite(feature_corrupted,paste0("missing_halfgchighest10perc/LGGGBM_missing_halfgchighest10perc_trial",SEED,".csv"))
}

##################################
#Random missing
#10% random missing
##################################

data<- data_raw[which(data_raw$admin.disease_code=="lgg"|data_raw$admin.disease_code=="gbm"),]
nonmissing_percent=0.9
missing_row_perc=0.2

clin=data[,1:4]
feature=data[,5:17179]
sample_n=dim(feature)[1]
feature_size=dim(feature)[2]

for (SEED in c(1:10)){
  
  set.seed(SEED)
  missing_rowind=sample(1:sample_n,sample_n*missing_row_perc )
  missing_rowind
  
  set.seed(SEED)
  missing_row_nonmissingones=matrix(rbinom(length(missing_rowind)*feature_size,1,nonmissing_percent), nrow = length(missing_rowind))
  
  feature_corrupted=feature
  feature_subset=feature[missing_rowind,]
  corrupt_subset=missing_row_nonmissingones*feature_subset
  feature_corrupted[missing_rowind,]<- corrupt_subset
  feature_corrupted[feature_corrupted==0]<-NA
  
  sum(is.na(rowSums(feature_corrupted)))/sample_n
  
  if (!file.exists("missing_10perc")) {
    dir.create("missing_10perc", recursive = TRUE)
  }

  fwrite(feature_corrupted, paste0("missing_10perc/LGGGBM_missing_10perc_trial",SEED,".csv"))
}
