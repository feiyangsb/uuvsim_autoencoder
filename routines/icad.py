from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
import numpy as np

class ICAD():
    def __init__(self, isTrajectory, trainingData, calibrationData, ncm=1):
        self.trainingData = trainingData
        self.calibrationData = calibrationData
        self.isTrajectory =isTrajectory
        self.ncm = ncm
        
        if self.isTrajectory:
            self.traj = np.empty([0, 8])
            v = []
            self.k = 5
            for i in range(len(self.calibrationData)):
                v_row = []
                for j in range(len(self.trainingData)):
                    v_row.append(directed_hausdorff(self.calibrationData[i], self.trainingData[j])[0])
                v_row.sort()
                v.append(sum(v_row[0:self.k]))
                self.calibration_NC = np.asarray(v)
                self.calibration_NC.sort()
        else:
            # sum of dists to k nearest neighbors
            if self.ncm == 1: 
                self.nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(self.trainingData)
                dists, indices = self.nbrs.kneighbors(self.calibrationData) 
                calibration_nbrs_distances_sum = np.sum(dists, axis=1)
                self.calibration_NC = calibration_nbrs_distances_sum
                self.calibration_NC.sort()
            # TODO: need to add the new ncm 
            else:
                raise Exception('This nonconformity measure has not been implemented.')


            
    
    def __point_evaluate(self, data_point):
        data_point = data_point.reshape(1,8)
        if data_point[0][0] == None:
            return None
        else:
            distances, _ = self.nbrs.kneighbors(data_point)
            test_nbrs_distances_sum = np.sum(distances, axis=1)
            test_NC = test_nbrs_distances_sum
            count = 0
            for i in range(len(self.calibration_NC)):
                if test_NC[0] < self.calibration_NC[i]:
                    count += 1
            p = count / float(len(self.calibration_NC))
            return p

    
    def __traj_evaluate(self, data_point):
        data_point = data_point.reshape(1,8)
        if data_point[0][0] == None:
            self.traj = np.empty([0,8])
            return None
        else:
            self.traj = np.concatenate((self.traj, data_point))
            v_row = []
            for i in range(len(self.trainingData)):
                v_row.append(directed_hausdorff(self.traj, self.trainingData[i])[0])
            v_row.sort()
            test_NC = sum(v_row[0:self.k])
            p = sum(ele > test_NC for ele in self.calibration_NC) / float(len(self.calibration_NC))
            return p
        

    
    def __call__(self, data_point):
        if self.isTrajectory:
            p = self.__traj_evaluate(data_point)
        else:
            p = self.__point_evaluate(data_point)
        return p