#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

# read the dataset
data_list = np.empty([0, 10])
data_directory = "./2/csv/"
folders = os.listdir(data_directory)
for folder in folders:
    for filename in os.listdir(os.path.join(data_directory, folder)):
        data = pd.read_csv(os.path.join(data_directory, folder, filename), usecols=[0,1,2,3,4,5,6,7,8,9])
        data_numpy = np.array(data)
        data_list = np.concatenate((data_list, data_numpy))

X = data_list[:,:8]
y = data_list[:,-2:]

X = X/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
print(X.shape)

encoding_dim = 4
states_input = Input(shape=(8, ))
encoded = Dense(encoding_dim, activation='relu')(states_input)
decoded = Dense(8, activation='tanh')(encoded)

encoder = Model(states_input, encoded)
autoencoder = Model(states_input, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.fit(X, X, epochs=200, batch_size=256)

encoder.save('encoder.h5')
autoencoder.save('autoencoder.h5')
