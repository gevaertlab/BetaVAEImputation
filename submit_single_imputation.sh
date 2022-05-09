#!/bin/bash

echo "#/usr/bin/bash
#$ -cwd
#$ -l h_vmem=60G
#$ -o logs/run_single_imputation/singimp.out
#$ -e logs/run_single_imputation/singimp.err

. /etc/profile.d/modules.sh

module load anaconda/5.3.1
source activate vae_imp_tf2

python evaluate_single_imputation_eddie.py " > job_submission_scripts/generate_dataset_singleimp.sh

qsub job_submission_scripts/generate_dataset_singleimp.sh
