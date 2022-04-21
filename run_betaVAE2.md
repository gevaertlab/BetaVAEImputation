## Current steps for running BetaVAE training and multiple imputation

### Create conda environment with required packages

```
conda create --name vae_imp_tf2 -c conda-forge tensorflow tensorflow-probability matplotlib pandas scikit-learn
```

Currently (as of 21/04/22), this will install tensorflow v2.7.0

### Activate your environment and run script `betaVAEv2.py` in a screen session to train your model

```
screen
conda activate vae_imp_tf2
python betaVAEv2.py
```

`ctrl+a+d` to detach from the screen session and `screen -r` to resume

Your trained model will be saved to the `output/` directory once done training

### Imputation for Mteropolis-within-Gibbs
* Currently the script submit_m_jobs.sh is configured to submit 40 jobs in parallel that run the following command

```
python evaluate_metropolis_gibbs_eddie.py --dataset m
```

With m being the sequence from 1 to 40, and outputs from each of the 40 datasets being placed in `output/Metropolis-within-Gibbs`

