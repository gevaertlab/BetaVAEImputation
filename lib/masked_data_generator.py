import datetime
import pickle
import numpy as np
from tensorflow import keras

from lib.helper_functions import get_scaled_data
from betaVAEv2 import load_model_v2



# def load_data():
#     x = pd.read_csv('../data/LGGGBM_missing_10perc_trial1.csv')
#     y = pd.read_csv('../data/data_complete.csv')
#     x, y = apply_scaler(y, x, return_scaler=False)
#     return x, y





class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_train, y_train, batchSize, prop_missing_patients=0.2, prop_missing_features=0.1): # you can add parameters here
        self.batchSize = batchSize
        self.x_train = x_train
        self.y_train = y_train
        self.n_cols  = self.x_train.shape[-1]
        self.n_rows = self.x_train.shape[-2]
        self.n_rows_to_add_null = int(len(self.x_train) * prop_missing_patients)
        self.n_cols_to_null = int(self.n_cols * prop_missing_features)
        self.change_mask_and_shuffle()
        print(f'x_train.shape;', self.x_train.shape)

    def __len__(self):
        return self.n_rows//self.batchSize

    def __getitem__(self, index):
        return self.x_train[index*self.batchSize:(index+1)*self.batchSize:], self.y_train[index*self.batchSize:(index+1)*self.batchSize:]

    def on_epoch_end(self):
        self.change_mask_and_shuffle() # change your data here

    def get_random_col_selection(self):
        return np.random.choice(range(self.n_cols), self.n_cols_to_null, replace=False)

    def shuffle(self):
        shuffle_index = np.random.choice(range(self.n_rows), self.n_rows, replace=False)
        self.x_train = self.x_train[shuffle_index]
        self.y_train = self.y_train[shuffle_index]
        bp = True

    def change_mask_and_shuffle(self):
        random_rows = np.random.choice(range(self.n_rows), self.n_rows_to_add_null, replace=False)
        null_row_indexes = np.array([np.repeat(i, repeats=self.n_cols_to_null) for i in random_rows]).flatten()
        null_col_indexes = np.array([self.get_random_col_selection() for _ in range(self.n_rows_to_add_null)]).flatten()
        new_masked_x = np.copy(self.y_train)
        new_masked_x[null_row_indexes, null_col_indexes] = 0
        self.x_train = new_masked_x
        self.shuffle()

if __name__ == "__main__":
    data, data_missing = get_scaled_data()
    training_generator = DataGenerator(x_train=data_missing, y_train=np.copy(data_missing), batchSize=250)
    model = load_model_v2(load_pretrained=False)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005, clipnorm=1.0))
    history = model.fit(training_generator, use_multiprocessing=True, workers=4, epochs=350)
    with open('output/masked_train_history_dict.pickle', 'wb') as file_handle:
        pickle.dump(history.history, file_handle)
    decoder_save_path = f"output/masked_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_decoder.keras"
    encoder_save_path = f"output/masked_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_encoder.keras"
    model.encoder.save(encoder_save_path)
    model.decoder.save(decoder_save_path)