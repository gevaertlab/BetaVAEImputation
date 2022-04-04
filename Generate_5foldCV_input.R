# Generate 5-fold cross validation to look at overfitting
library(data.table)

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/Qiu_et_al/analysis/simulated_missing/5foldCV")

# Read in data
missdat <- read.csv("../missing_10perc/LGGGBM_missing_10perc_trial1.csv")

Yobs <- missdat[which(!is.na(rowSums(missdat))),]
Ymis <- missdat[which(is.na(rowSums(missdat))),]

# Generate 5 datasets split where 4/5 is complete and 1/5 has missing values
nonmissing_percent=0.9
missing_row_perc=0.2

feature=Yobs
sample_n=dim(feature)[1]
feature_size=dim(feature)[2]

simDats <- list()
# First 4 groups will have 107 samples, last group will have 106

for (SEED in c(1:5)){
  
  if (SEED < 5) {
    missing_rowind <- 1:107 + (SEED-1)*107
  } else {
    missing_rowind <- 429:534
  }
  
  set.seed(SEED)
  missing_row_nonmissingones=matrix(rbinom(length(missing_rowind)*feature_size,1,nonmissing_percent), nrow = length(missing_rowind))
  
  feature_corrupted=feature
  feature_subset=feature[missing_rowind,]
  corrupt_subset=missing_row_nonmissingones*feature_subset
  feature_corrupted[missing_rowind,] <- corrupt_subset
  feature_corrupted[feature_corrupted==0]<-NA
  
  sum(is.na(rowSums(feature_corrupted)))/sample_n
  
  simDats[[SEED]] <- feature_corrupted
  
  fwrite(feature_corrupted, paste0("LGGGBM_missing_10perc_trial_1_Yobs_CV",SEED,".csv"))
}

fwrite(Yobs, "LGGGBM_missing_10perc_trial_1_Yobs.csv")

