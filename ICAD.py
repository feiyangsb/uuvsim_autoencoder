#%%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
print("# of training data point: {}, # of calibration point： {}".format(len(X_train), len(X_calibration))) 

#%%
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(X_train)
distances, indices = nbrs.kneighbors(X_calibration)
calibration_nbrs_distances_sum = np.sum(distances, axis=1)
calibration_NC = calibration_nbrs_distances_sum
calibration_NC.sort()
print(calibration_NC.shape)
plt.figure(1)
plt.plot(calibration_NC)
#plt.show()

#%%
test_data_path = "./2/0.csv"
test_data = pd.read_csv(test_data_path, usecols=[0,1,2,3,4,5,6,7])
test_data = np.array(test_data)/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
NCM_list = []
p_list = []
for i in range(len(test_data)):
    test_point = test_data[i].reshape(1, -1)
    distances, indices = nbrs.kneighbors(test_point)
    test_nbrs_distances_sum = np.sum(distances, axis=1)
    test_NC = test_nbrs_distances_sum
    NCM_list.append(test_NC)

    # compute p value
    count = 0
    for j in range(len(calibration_NC)):
        if test_NC[0] < calibration_NC[j]:
            count += 1
    p = (count) / float(len(calibration_NC))
    p_list.append(p)

plt.figure(2)
plt.plot(p_list)
plt.show()