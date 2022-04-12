import numpy as np
import os
import tensorflow as tf
from lib.helper_functions import get_scaled_data
from betaVAEv2 import VariationalAutoencoderV2, Sampling, network_architecture
import time
os.chdir('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
"""
This module evaluates the amount of time it takes to run the decoder on a large number of samples
"""

for dir in sorted(os.listdir('output/'), reverse=True):
    if not os.path.isdir('output/' + dir) or 'epoch' not in dir:
        continue
    dir = sorted(os.listdir('output/'), reverse=True)[-1]
    x = np.random.normal(loc=0, scale=1, size=(10_000, 200))
    decoder_path = '/home/jwells/Documents/BetaVAEImputation/output/20220406-10:46:54_loss1566_beta1_lr5e-05_epoch321/decoder_masked.keras'
    decoder = tf.keras.models.load_model(decoder_path, custom_objects={'Sampling': Sampling})
    start = time.time()
    results = decoder.predict(x)
    total = time.time() - start
    print(total)
    bp=True
