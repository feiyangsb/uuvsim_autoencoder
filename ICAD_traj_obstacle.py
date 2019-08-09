#!/usr/bin/python3

#%%
import pandas as pd
import os
import numpy as np

data_list = []
data_directory = "./2/obstacle/"
files = os.listdir(data_directory)
for filename in files:
    data = pd.read_csv(os.path.join(data_directory, filename), usecols=[0,1,2,3,4,5,6,7,8,9])
    data_numpy = np.array(data)
    one_obstacle_list = np.empty([0, 10])
    for i in range(len(data_numpy)):
        if data_numpy[i][6] <= 0 and len(one_obstacle_list)!=0:
            data_list.append(one_obstacle_list)
            one_obstacle_list = np.empty([0, 10])
        if data_numpy[i][6] > 0:
            one_obstacle_list.append(data_numpy[i])

print(len(data_list))


#%%
