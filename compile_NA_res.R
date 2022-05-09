# Compile NA indices results into one file
setwd('/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/Metropolis-within-Gibbs/')

files <- list.files()
files <- files[grep("NA",files)]

for (i in 1:length(files)) {
    df <- read.csv(files[i], row.names = 1, stringsAsFactors = F)
    print(paste("reading in file number", i))
    if (i == 1) {
        final <- df
    } else {
        datname <- colnames(df)[2]
        final <- data.frame(final, df[,2])
        colnames(final)[i+1] <- datname
    }
}

write.csv(final, '/exports/igmm/eddie/ponting-lab/breeshey/projects/BetaVAEImputation/output/Metropolis-within-Gibbs/compiled_NA_indices.csv', row.names = F)


