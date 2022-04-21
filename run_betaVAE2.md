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
ctrl+a+d to detach
```

Your trained model will be saved to the `output/` directory once done training

