import os
import datetime
import pickle
import numpy as np
from tensorflow import keras

from lib.helper_functions import get_scaled_data
from betaVAEv2 import load_model_v2

epochs = 10
lr = 0.00005
beta = 1
data, data_missing = get_scaled_data()
model = load_model_v2(load_pretrained=False, beta=1)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))
for i in range(85):
    history = model.fit(x=data_missing, y=data_missing, epochs=epochs)
    final_loss = int(history.history['loss'][-1])
    output_dir = f"output/non_masked/{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_loss{final_loss}_beta{str(beta).replace('.', 'p')}_lr{str(lr).replace('.', 'p')}_epoch{epochs * (i + 1)}/"
    os.mkdir(output_dir)
    with open(output_dir + 'train_history_dict.pickle', 'wb') as file_handle:
        pickle.dump(history.history, file_handle)
    decoder_save_path = f"{output_dir}decoder.keras"
    encoder_save_path = f"{output_dir}encoder.keras"
    model.encoder.save(encoder_save_path)
    model.decoder.save(decoder_save_path)