#!/usr/bin/python3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class dataLoader():
    def __init__(self, isTrajectory=False, isObstacle=False):
        self.isTrajectory = isTrajectory
        self.isObstacle = isObstacle
        self.normalizer = [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0, 1.0, 1.0]
        if self.isObstacle:
            self.dataPath = "./data/obstacle"
        else:
            self.dataPath = "./data/tracking"
            

    def load(self):
        if self.isTrajectory:
            X, Y = self.__loadTrajectoryData()
        else:
            X, Y = self.__loadPointData()

        X_train, X_calibration, Y_train, Y_calibration = train_test_split(X,Y, test_size=0.2, random_state=42)
        if self.isTrajectory:
            print("# of training trajectories: {}, # of calibration trajectories: {}".format(len(X_train), len(X_calibration)))
        else:
            print("# of training data points: {}, # of calibration data points: {}".format(len(X_train), len(X_calibration)))
        
        return (X_train, Y_train), (X_calibration, Y_calibration)

    def __loadTrajectoryData(self):
        traj_list = []
        for folder in os.listdir(self.dataPath):
            for file in os.listdir(os.path.join(self.dataPath, folder)):
                data = pd.read_csv(os.path.join(self.dataPath, folder, file), usecols=[0,1,2,3,4,5,6,7,8,9])
                data = np.array(data)/self.normalizer

                # filter out the obstacle segments
                if self.isObstacle:
                    data_obstacle_candidates = data[(data[:,6]>0) & (data[:,7]>0)]
                    data_obstacle_candidates_roll_right = np.roll(data_obstacle_candidates, 1, axis=0)
                    data_obstacle_candidates_roll_right[0, 7] = 0.0
                    filter_array = data_obstacle_candidates[:,7] - data_obstacle_candidates_roll_right[:,7]
                    index_array = np.argwhere(filter_array>0.0)
                    for i in range(len(index_array)):
                        if i+1 == len(index_array):
                            traj_list.append(data_obstacle_candidates[index_array[i][0]:])
                        else:
                            traj_list.append(data_obstacle_candidates[index_array[i][0]:index_array[i+1][0]])
                
                # use all data as a segments
                # TODO: use different length of segments
                else:
                    slice_length = 20
                    count = 0
                    while (count+slice_length < len(data)):
                        traj_list.append(data[count:count+slice_length, :])
                        count += slice_length
                    traj_list.append(data[count:, :])

        X = []                
        Y = []
        for i in range(len(traj_list)):
            X.append(traj_list[i][:,:8])
            Y.append(traj_list[i][:,-2:])

        X = np.asarray(X)
        Y = np.asarray(Y)
        return X, Y

    def __loadPointData(self):
        data_list = np.empty([0, 10])
        for folder in os.listdir(self.dataPath):
            for file in os.listdir(os.path.join(self.dataPath, folder)):
                data = pd.read_csv(os.path.join(self.dataPath, folder, file), usecols=[0,1,2,3,4,5,6,7,8,9])
                data = np.array(data)/self.normalizer

                # filter out the non-obstalce data
                if self.isObstacle:
                    data_obstacle_candidates = data[(data[:,6]>0) & (data[:,7]>0)]
                    data_obstacle_candidates_roll_right = np.roll(data_obstacle_candidates, 1, axis=0)
                    data_obstacle_candidates_roll_right[0, 7] = 0.0
                    filter_array = data_obstacle_candidates[:,7] - data_obstacle_candidates_roll_right[:,7]
                    index_array = np.argwhere(filter_array>0.0)
                    for i in range(len(index_array)):
                        if i+1 == len(index_array):
                            data_list = np.concatenate((data_list,data_obstacle_candidates[index_array[i][0]:]))
                        else:
                            data_list = np.concatenate((data_list, data_obstacle_candidates[index_array[i][0]:index_array[i+1][0]]))
                # use all the data
                else:
                    data_list = np.concatenate((data_list,data))

        X = data_list[:,:8]
        Y = data_list[:,-2:]

        return X, Y

        
def testDataLoader(path, isObstacle):
    normalizer = [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0, 1.0, 1.0]
    data_list = np.empty([0, 10])
    data = pd.read_csv(path, usecols=[0,1,2,3,4,5,6,7,8,9])
    data = np.array(data)/normalizer
    if isObstacle:
        pre_point = np.zeros([10,])
        for i in range(len(data)):
            current_point = data[i][0:10]
            if (current_point[6]<0.0 or current_point[7]<0.0 or current_point[7]-pre_point[7]>=0):
                data_list = np.concatenate((data_list, np.array([None]*10).reshape(1,10)))
            else:
                data_list = np.concatenate((data_list,current_point.reshape(1,10)))    
            pre_point = np.copy(current_point)
    
    else:
        data_list = np.copy(data[:,0:10])
    
    return data_list
