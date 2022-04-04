import numpy as np
from tensorflow import keras
import pandas as pd
from lib.helper_functions import apply_scaler
from lib.helper_functions import load_saved_model

class CustomCallback(keras.callbacks.Callback):
    # call backs probably can't be used to modify the dataset
    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

def load_data():
    x = pd.read_csv('../data/LGGGBM_missing_10perc_trial1.csv')
    y = pd.read_csv('../data/data_complete.csv')
    x, y = apply_scaler(y, x, return_scaler=False)
    pass

def load_y():
    df =
    pass



class DataGenerator(keras.utils.Sequence):
    def __init__(self, batchSize, prop_missing_patients=0.2, prop_missing_features=0.1): # you can add parameters here
        self.batchSize = batchSize
        self.x_train, self.y_train = load_data()
        self.n_cols  = self.x_train.shape[1]
        self.n_rows = self.x_train.shape[0]
        self.n_rows_to_add_null = int(len(self.x_train) * prop_missing_patients)
        self.n_cols_to_null = int(self.n_cols * prop_missing_features)
        self.x_train = self.change_mask_and_shuffle()

    def __len__(self):
        return self.xData.shape[0]//self.batchSize

    def __getitem__(self, index):
        return self.x_train[index*self.batchSize:(index+1)*self.batchSize:]

    def on_epoch_end(self):
        self.x_train = self.change_mask_and_shuffle() # change your data here

    def get_random_col_selection(self):
        return np.random.choice(range(self.n_cols), self.n_cols_to_null, replace=False)

    def shuffle(self):
        shuffle_index = np.random.shuffle(np.arange(self.n_rows))
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
    # generator should return a sample of
    training_generator = DataGenerator(batchSize=)
    # validation_generator = DataGenerator(partition['validation'], labels, **params)
    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        # validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=4)
    pass