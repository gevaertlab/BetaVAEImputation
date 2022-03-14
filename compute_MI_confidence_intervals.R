## Look at confidence intervals across all NA indices
# In Multiple imputation setting
library(ggplot2)
library(dplyr)

setwd('/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/Qiu_et_al/analysis/multiple_imputation/')

# Read in data
na_vals <- read.csv("../../git_repository/BetaVAEImputation/NA_imputed_values_m_datasets.csv", 
                    stringsAsFactors = F)

na_vals$X <- paste("NA", na_vals$X + 1, sep = "_")
colnames(na_vals)[1] <- "NA_index"

# Compute mean and standard deviation across all NA indices
na_vals$mean_imputed_vals <- rowMeans(na_vals[,2:11])
na_vals$stdev_imputed_vals <- apply(na_vals[,2:11],1, sd)

# Compute 95% confidence intervals for each NA index
CIs <- sapply(1:nrow(na_vals), function(x) {
  n <- 10
  subdat <- na_vals[x,]
  s <- subdat$stdev_imputed_vals
  mn <- subdat$mean_imputed_vals
  grtruth <- subdat$actual_values
  
  margin <- qt(0.975,df=n-1)*s/sqrt(n)
  upper_interval <- mn + margin
  lower_interval <- mn - margin
  
  is_95 <- grtruth < upper_interval & grtruth > lower_interval
  
  final_res <- c(lower_interval,upper_interval,is_95)
})

## Now add this into our table na_vals
na_vals <- data.frame(na_vals, as.data.frame(t(CIs)))
na_vals$in_95_CI <- as.logical(na_vals$in_95_CI)
colnames(na_vals)[15:17] <- c("lower_interval","upper_interval","in_95_CI")

# What does this look like?
summary(na_vals$in_95_CI)



