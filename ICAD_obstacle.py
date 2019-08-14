#%%
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

data_directory = "./2/obstacle/"
folders = os.listdir(data_directory)
obstacle_list = np.empty([0, 10])
for folder in folders:
    for file in os.listdir(os.path.join(data_directory, folder)):
        data = pd.read_csv(os.path.join(data_directory, folder, file), usecols=[0,1,2,3,4,5,6,7,8,9])
        data = np.array(data)
        data_obstacle_candidates = data[(data[:,6]>0) & (data[:,7]>0)]
        data_obstacle_candidates_roll_right = np.roll(data_obstacle_candidates, 1, axis=0)
        data_obstacle_candidates_roll_right[0, 7] = 0.0
        filter_array = data_obstacle_candidates[:,7] - data_obstacle_candidates_roll_right[:,7]
        index_array = np.argwhere(filter_array>0.0)
        for i in range(len(index_array)):
            if i+1 == len(index_array):
                obstacle_list = np.concatenate((obstacle_list,data_obstacle_candidates[index_array[i][0]:]))
            else:
                obstacle_list = np.concatenate((obstacle_list, data_obstacle_candidates[index_array[i][0]:index_array[i+1][0]]))

X = obstacle_list[:,:8]
y = obstacle_list[:,-2:]
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
plt.figure(1)
plt.plot(calibration_NC)

#%%
from routines.martingales import RPM, SMM, PIM
RPM_list = []
SMM_list = []
PIM_list = []

test_obstacle_list = []
test_data_path = "./2/10.csv"
test_data = pd.read_csv(test_data_path, usecols=[0,1,2,3,4,5,6,7,8,9]) 
test_data = np.array(test_data)

p_list = []
pre_point = np.zeros([8,])
for i in range(len(test_data)):
    current_point = test_data[i][0:8]/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
    if (current_point[7]-pre_point[7]>=0.0):
        rpm = RPM(0.75)
        smm = SMM()
        pim = PIM()

    pre_point = np.copy(current_point)
    if (current_point[6]>0.0 and current_point[7]>0.0):
        pass

    if (current_point[6]>0.0 and current_point[7]>0.0):
        distances, indices = nbrs.kneighbors(current_point.reshape(1, -1))
        test_nbrs_distances_sum = np.sum(distances, axis=1)
        test_NC = test_nbrs_distances_sum
        # compute p value
        count = 0
        for j in range(len(calibration_NC)):
            if test_NC[0] < calibration_NC[j]:
                count += 1
        p = (count) / float(len(calibration_NC))
        p_list.append(p)
        RPM_list.append(rpm(p))
        SMM_list.append(smm(p))
        PIM_list.append(pim(p))
    else:
        p_list.append(None)
        RPM_list.append(None)
        SMM_list.append(None)
        PIM_list.append(None)
"""
plt.figure(2)
plt.plot(p_list)
plt.figure(3)
plt.plot(RPM_list)
plt.figure(4)
plt.plot(SMM_list)
plt.figure(5)
plt.plot(PIM_list)
plt.show()
"""
f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
plt.suptitle("Single Point ICAD (Obstacle Avoidance)")
ax1.plot(p_list)
ax1.set_ylabel("p")
ax2.plot(RPM_list)
ax2.set_ylabel("RPM")
ax3.plot(SMM_list)
ax3.set_ylabel("SMM")
ax4.plot(PIM_list)
ax4.set_xlabel("Time(S)")
ax4.set_ylabel("PIM")
plt.show()