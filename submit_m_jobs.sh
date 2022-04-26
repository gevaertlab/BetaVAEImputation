#!/bin/bash

declare -a ids=($(seq 1 40))

for i in ${!ids[@]}
do
    dig=${ids[$i]}
    
    echo "#/usr/bin/bash
#$ -cwd
#$ -l h_vmem=60G
#$ -o logs/run_pseudo_Gibbs/dat${dig}.out
#$ -e logs/run_pseudo_Gibbs/dat${dig}.err

. /etc/profile.d/modules.sh

module load anaconda/5.3.1
source activate vae_imp_tf2

python evaluate_pseudo_Gibbs_eddie.py --dataset ${dig} " > job_submission_scripts/generate_dataset_pG_${dig}.sh

    qsub job_submission_scripts/generate_dataset_pG_${dig}.sh

done
