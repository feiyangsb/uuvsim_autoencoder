#!/usr/bin/python3
#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model 
#from routines.ICAD import ICAD

#%%
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
X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_calibration.shape)


#%%
autoencoder = load_model("./autoencoder.h5")
encoder = load_model("./encoder.h5")
X_train_encoded = encoder.predict(X_train)
X_calibration_encoded = encoder.predict(X_calibration)
X_train_reconstructed = autoencoder.predict(X_train)
X_calibration_reconstructed = autoencoder.predict(X_calibration)
reconstruction_error_train = np.square(X_train-X_train_reconstructed).mean(axis=1)
reconstruction_error_calibration = np.square(X_calibration-X_calibration_reconstructed).mean(axis=1)
print(reconstruction_error_train.shape, reconstruction_error_calibration.shape)



#%%
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(X_train_encoded)
distances, indices = nbrs.kneighbors(X_calibration_encoded)
calibration_nbrs_reconstruction_error = reconstruction_error_train[indices]
calibration_NC = np.sum(calibration_nbrs_reconstruction_error, axis=1)
calibration_NC.sort()
print(calibration_nbrs_reconstruction_error.shape, calibration_NC.shape)
plt.plot(calibration_NC)

#%%
test_data_path = "./2/2.csv"
test_data = pd.read_csv(test_data_path, usecols=[0,1,2,3,4,5,6,7])
test_data = np.array(test_data)/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]

for i in range(len(test_data)):
    test_point = test_data[i].reshape(1, -1)
    test_point_encoded = encoder.predict(test_point)
    test_point_reconstruction = autoencoder.predict(test_point)
    reconstruction_error_test = np.square(test_point-test_point_reconstruction).mean(axis=1)
    print(test_point_reconstruction.shape, reconstruction_error_test.shape)
    


#%%
