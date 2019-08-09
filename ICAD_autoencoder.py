#!/usr/bin/python3
#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.integrate as integrate
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
calibration_nbrs_distances_sum = np.sum(distances, axis=1)
calibration_nbrs_reconstruction_error_sum = np.sum(calibration_nbrs_reconstruction_error, axis=1)
calibration_NC_normolized = calibration_nbrs_distances_sum #/(5.0*calibration_nbrs_reconstruction_error_sum)
calibration_NC_normolized.sort()
print(calibration_NC_normolized.shape)
plt.plot(calibration_NC_normolized)

#%%
p_list = []
def integrand(x):
    result = 1.0
    for i in range(len(p_list)):
        result *= x*(p_list[i]**(x-1.0))
    return result

#%%
import math
test_data_path = "./2/10.csv"
test_data = pd.read_csv(test_data_path, usecols=[0,1,2,3,4,5,6,7])
test_data = np.array(test_data)/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]

NCM_list = []
global p_list
p_smooth_list = []
p_neg_log_list = []
M_list = []
M_sw_list = []
SMM_list = []
M = 1.0
epsilon = 0.75
for i in range(len(test_data)):
    test_point = test_data[i].reshape(1, -1)
    test_point_encoded = encoder.predict(test_point)
    test_point_reconstruction = autoencoder.predict(test_point)
    reconstruction_error_test = np.square(test_point-test_point_reconstruction).mean(axis=1)
    distances, indices = nbrs.kneighbors(test_point_encoded)
    test_nbrs_reconstruction_error = reconstruction_error_train[indices]
    test_nbrs_distances_sum = np.sum(distances, axis=1)
    test_nbrs_reconstruction_error_sum = np.sum(test_nbrs_reconstruction_error, axis=1)
    test_NC_normalized = test_nbrs_distances_sum#/(5.0*test_nbrs_reconstruction_error_sum)
    NCM_list.append(test_NC_normalized)
    count = 0
    count_same = 0
    for j in range(len(calibration_NC_normolized)):
        if test_NC_normalized[0] < calibration_NC_normolized[j]:
            count += 1
        if test_NC_normalized[0] == calibration_NC_normolized[j]:
            count_same += 1

    tau = np.random.uniform(0,1.0)
    p = (count) / float(len(calibration_NC_normolized))
    if (p == 0.0):
        p = 0.05
    M = M * epsilon * (p ** (epsilon - 1))
    M_list.append(M)
    SMM, err = integrate.quad(integrand, 0, 1)
    SMM_list.append(SMM)
    p_neg_log = -math.log10(p)
    p_list.append(p)
    if (len(p_list)<20):
        M_sw_list.append(0.0)
    else:
        sub_list = p_list[-20:]
        M_sw = np.prod((np.power(sub_list, (epsilon-1)) * epsilon))
        M_sw_list.append(M_sw)
    p_smooth = (count+tau*count_same) / float(len(calibration_NC_normolized))
    p_smooth_list.append(p_smooth)
    p_neg_log_list.append(p_neg_log)

plt.figure(1)
plt.plot(NCM_list)

plt.figure(2)
plt.plot(p_list)

plt.figure(3)
plt.plot(p_neg_log_list)

plt.figure(4)
plt.plot(p_smooth_list)

plt.figure(5)
plt.plot(M_sw_list)

plt.figure(6)
plt.plot(M_list)

plt.figure(7)
plt.plot(SMM_list)
#%%

