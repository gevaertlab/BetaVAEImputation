#!/bin/bash

declare -a ids=($(seq 1 40))

for i in ${!ids[@]}
do
    dig=${ids[$i]}
    
    echo "#/usr/bin/bash
#$ -cwd
#$ -l h_vmem=60G
#$ -o logs/run_Metropolis_Gibbs/dat${dig}.out
#$ -e logs/run_Metropolis_Gibbs/dat${dig}.err

. /etc/profile.d/modules.sh

module load anaconda/5.3.1
source activate vae_imp_tf2

python evaluate_metropolis_gibbs_eddie.py  --dataset ${dig} " > job_submission_scripts/generate_dataset_MG_${dig}.sh

    qsub job_submission_scripts/generate_dataset_MG_${dig}.sh

done
