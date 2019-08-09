#!/usr/bin/python3

#%%
import pandas as pd
import os
import numpy as np

data_directory = "./2/obstacle/"
folders = os.listdir(data_directory)
obstacle_list = []
for folder in folders:
    for file in os.listdir(os.path.join(data_directory, folder)):
        print(folder, file)
        data = pd.read_csv(os.path.join(data_directory, folder, file), usecols=[0,1,2,3,4,5,6,7,8,9])
        data = np.array(data)
        data_obstacle_candidates = data[(data[:,6]>0) & (data[:,7]>0)]
        data_obstacle_candidates_roll_right = np.roll(data_obstacle_candidates, 1, axis=0)
        data_obstacle_candidates_roll_right[0, 7] = 0.0
        filter_array = data_obstacle_candidates[:,7] - data_obstacle_candidates_roll_right[:,7]
        index_array = np.argwhere(filter_array>0.0)
        for i in range(len(index_array)):
            if i+1 == len(index_array):
                obstacle_list.append(data_obstacle_candidates[index_array[i][0]:])
            else:
                obstacle_list.append(data_obstacle_candidates[index_array[i][0]:index_array[i+1][0]])

#%%
import matplotlib.pyplot as plt
plt.plot(obstacle_list[12][:,6])

#%%
from sklearn.model_selection import train_test_split
X = []
Y = []
for i in range(len(obstacle_list)):
    X.append(obstacle_list[i][:,:8]/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0])
    Y.append(obstacle_list[i][:,-2:])

X_train, X_calibration, Y_train, Y_calibration = train_test_split(X,Y,test_size=0.2, random_state=42)
print("# of training traj: {}, # of calibration traj: {}".format(len(X_train), len(X_calibration)))

#%%
from scipy.spatial.distance import directed_hausdorff

v = []
k = 5
for i in range(len(X_calibration)):
    v_row = []
    for j in range(len(X_train)):
        v_row.append(directed_hausdorff(X_calibration[i], X_train[j])[0])
    v_row.sort()
    v.append(sum(v_row[0:k]))

#%%
test_obstacle_list = []
test_data_path = "./2/10.csv"
test_data = pd.read_csv(test_data_path, usecols=[0,1,2,3,4,5,6,7,8,9]) 
test_data = np.array(test_data)

traj = np.empty([0,8])
pre_point = np.zeros([8,])
p_list = []
for i in range(len(test_data)):
    current_point = test_data[i][0:8]/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
    if (current_point[7]-pre_point[7]>=0.0):
        traj = np.empty([0,8])
    pre_point = np.copy(current_point)
    if (current_point[6]>0.0 and current_point[7]>0.0):
        traj = np.concatenate((traj, current_point.reshape([1,8])))

    if (current_point[6]>0.0 and current_point[7]>0.0):
        v_row = []
        for j in range(len(X_train)):
            v_row.append(directed_hausdorff(traj, X_train[j])[0])
        v_row.sort()
        alpha = sum(v_row[0:k])
        p = sum(ele > alpha for ele in v) /float(len(v))
        p_list.append(p)
    else:
        p_list.append(None)
plt.plot(p_list)

#%%
