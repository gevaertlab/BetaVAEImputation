## Look at confidence intervals across all NA indices
# In Multiple imputation setting
library(ggplot2)
library(dplyr)
library(reshape2)

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
colnames(na_vals)[15:17] <- c("lower_interval","upper_interval","in_95_CI")
na_vals$in_95_CI <- as.logical(na_vals$in_95_CI)

# What does this look like?
summary(na_vals$in_95_CI)

# What about the question: is the actual value between the minimum and maximum plausible value?
library(matrixStats)
na_vals$in_interval <- na_vals$actual_values > rowMins(as.matrix(na_vals[,2:11])) & na_vals$actual_values < rowMaxs(as.matrix(na_vals[,2:11]))

summary(na_vals$in_interval) # 60.57%

# Rank NA indices by true value and look at boxplot
# Do for both single (dot plot) and multiple imputation (boxplot)

na_vals_single <- read.csv("../../git_repository/BetaVAEImputation/NA_imputed_values_single_imputation.csv", 
                           stringsAsFactors = F)

na_vals_single$X <- paste("NA", na_vals_single$X + 1, sep = "_")
colnames(na_vals_single)[1] <- "NA_index"

# Order dataframe by NA index actual value
na_vals_single <- na_vals_single[order(na_vals_single$actual_values),]

## What does it look like if we plot actual values against all values including actual values and highlight them?
act_vals <- data.frame(actual_values = na_vals_single$actual_values, imputed_dataset = na_vals_single$actual_values)

pdf("actual_values_vs_single_imp_values_scatterplot.pdf", width = 10)
ggplot(data = na_vals_single, mapping = aes(x = actual_values, y = imputed_dataset)) +
  geom_point(alpha = 0.5) +
  geom_point(data = act_vals, colour = "red") +
  theme_bw() +
  xlab("Actual values") +
  ylab("Imputed values")
dev.off()

cor(na_vals_single$actual_values, na_vals_single$imputed_dataset)
# 0.923

## Now let's do the same for multiple imputation
na_vals <- na_vals[order(na_vals$actual_values),]
na_vals$actual_values <- as.factor(na_vals$actual_values)
na_vals_melt <- melt(na_vals[,c(2:12)], by = c("actual_values"))

#ggplot(data = na_vals_melt, mapping = aes(x = actual_values, y = value)) +
#  geom_boxplot()
  
## What about numeric?
na_vals_melt$actual_values <- as.numeric(paste(na_vals_melt$actual_values))

ggplot(data = na_vals_melt, mapping = aes(x = actual_values, y = value)) +
  geom_point(alpha = 0.5)

## What does it look like if we plot actual values against all values including actual values and highlight them?
act_vals <- data.frame(actual_values = as.numeric(paste(na_vals_melt$actual_values)), value = as.numeric(paste(na_vals_melt$actual_values)))

pdf("actual_values_vs_mult_imp_values_scatterplot.pdf", width = 10)
ggplot(data = na_vals_melt, mapping = aes(x = actual_values, y = value)) +
  geom_point(alpha = 0.5) +
  geom_point(data = act_vals, colour = "red") +
  theme_bw() +
  xlab("Actual values") +
  ylab("Imputed values")
dev.off()

# What do the distribution of actual and imputed values look like in a density plot?
na_vals$actual_values_numeric <- as.numeric(paste(na_vals$actual_values))

ggplot(data = na_vals, mapping = aes(x = actual_values_numeric)) +
  geom_density() +
  theme_bw()

ggplot(data = na_vals_single, mapping = aes(x = imputed_dataset)) +
  geom_density() +
  theme_bw()

# What about multiple imputation setting?
ggplot(data = na_vals_melt, mapping = aes(x = value)) +
  geom_density() +
  theme_bw()



