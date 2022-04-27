#!/bin/bash

echo "#/usr/bin/bash
#$ -cwd
#$ -l h_vmem=60G
#$ -o logs/run_importance_sampling/impsamp.out
#$ -e logs/run_importance_sampling/impsamp.err

. /etc/profile.d/modules.sh

module load anaconda/5.3.1
source activate vae_imp_tf2

python evaluate_importance_sampling_eddie.py " > job_submission_scripts/generate_dataset_isamp.sh

qsub job_submission_scripts/generate_dataset_isamp.sh
