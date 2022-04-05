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
        bp=True
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

    def change_mask_and_shuffle(self):
        random_rows = np.random.choice(range(self.n_rows), self.n_rows_to_add_null, replace=False)
        null_row_indexes = np.array([np.repeat(i, repeats=self.n_cols_to_null) for i in random_rows]).flatten()
        null_col_indexes = np.array([self.get_random_col_selection() for _ in range(self.n_rows_to_add_null)]).flatten()
        new_masked_x = np.copy(self.y_train)
        new_masked_x[null_row_indexes, null_col_indexes] = 0
        self.x_train = new_masked_x
        self.shuffle()

# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#
#         return X, y
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)
#
#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')
#
#             # Store class
#             y[i] = self.labels[ID]
#
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

if __name__ == "__main__":

    data, data_missing = get_scaled_data()
    # n_row = data.shape[1]
    # generator should return a sample of
    training_generator = DataGenerator(x_train=data_missing, y_train=np.copy(data_missing), batchSize=250)
    # validation_generator = DataGenerator(partition['validation'], labels, **params)
    # Train model on dataset
    model = load_model_v2()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0))
    history = model.fit(training_generator,
                        # validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4, epochs=100)
    with open('output/masked_train_history_dict.pickle', 'wb') as file_handle:
        pickle.dump(history.history, file_handle)
    decoder_save_path = f"output/masked_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_decoder.keras"
    encoder_save_path = f"output/masked_{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}_encoder.keras"
    model.encoder.save(encoder_save_path)
    model.decoder.save(decoder_save_path)