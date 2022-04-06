# Test 5-fold cross validation models

# Read in MI tables
setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/Qiu_et_al/analysis/5foldCV")

# Reconstruction error
re_1 <- read.csv("../../git_repository/BetaVAEImputation/output/5foldCV/CV1/NA_indices/reconstruction_error.csv", header = F)
re_2 <- read.csv("../../git_repository/BetaVAEImputation/output/5foldCV/CV2/NA_indices/reconstruction_error.csv", header = F)
re_3 <- read.csv("../../git_repository/BetaVAEImputation/output/5foldCV/CV3/NA_indices/reconstruction_error.csv", header = F)
re_4 <- read.csv("../../git_repository/BetaVAEImputation/output/5foldCV/CV4/NA_indices/reconstruction_error.csv", header = F)
re_5 <- read.csv("../../git_repository/BetaVAEImputation/output/5foldCV/CV5/NA_indices/reconstruction_error.csv", header = F)

# Rubin's rules
dfs <- list(re_1, re_2, re_3, re_4, re_5)

for (i in 1:length(dfs)) {
  reconstr_err_comb <- sum(dfs[[i]][,1]/nrow(dfs[[i]]))
  
  if (i==1) {
    rub_rul_re <- c(reconstr_err_comb)
  } else {
    rub_rul_re <- c(rub_rul_re, reconstr_err_comb)
  }
}

library(ggplot2)

forPlot <- as.data.frame(rub_rul_re)
forPlot$MI_dataset <- paste("CV_dataset",1:5, sep = "_")

ggplot(data = forPlot, mapping = aes(x = MI_dataset, y = rub_rul_re, colour = MI_dataset)) +
  geom_point() +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none") +
  ylim(0,1) +
  ylab("Reconstruction error \nvia Rubin's rules") +
  xlab("CV dataset")


