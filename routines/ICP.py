from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

class ICP(object):
    def __init__(self, X_train, y_train, X_calibration, y_calibration, n_neighbors):
        self.n_neighbors = n_neighbors
        self.y_train = y_train
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X_train[:,:])
        distances, indices = self.nbrs.kneighbors(X_calibration[:, 0:8])
        calibration_nbrs_outputs = y_train[indices]
        std_output = np.std(calibration_nbrs_outputs,axis=1)
        calibration_nbrs_outputs_mean = np.mean(calibration_nbrs_outputs, axis=1)
        calibration_NC = np.abs(calibration_nbrs_outputs_mean-y_calibration)
        """
        g7 = plt.figure(7)
        plt.plot(calibration_NC_sort[:,0])
        plt.title("NC for calibration dataset without normalizer(Heading Change)")
        g8 = plt.figure(8)
        plt.plot(calibration_NC_sort[:,1])
        plt.title("NC for calibration dataset without normalizer(Speed)")
        """
        sum_distances = np.repeat(np.sum(distances, axis=1).reshape(-1,1), 2, axis=1)
        self.rho = np.array([1.0,1.0])#np.array([[1.0],[1.0]]).repeat(len(X_calibration), axis=1)
        self.ksi = np.array([5.0,5.0])#np.array([[1.0],[1.0]]).repeat(len(X_calibration), axis=1)
        #normalizer = sum_distances*self.rho + std_distances*self.ksi
        #normalizer = np.exp(sum_distances*self.rho) + np.exp(std_output*self.ksi)
        #normalizer = np.exp(sum_distances/sum_distances_median*self.rho) + np.exp(std_output/std_output_median*self.ksi)
        #normalizer = sum_distances/float(n_neighbors)*self.rho + std_output*self.ksi
        #normalizer = np.abs(self.rho + sum_distances + std_output)
        normalizer = np.exp(sum_distances/float(self.n_neighbors)*self.rho) + np.exp(std_output*self.ksi)
        calibration_NC_normalized = calibration_NC / normalizer
        self.calibration_nonconformities = np.sort(calibration_NC_normalized, axis=0)
        """
        g1 = plt.figure(5)
        plt.plot(self.calibration_nonconformities[:,0])
        plt.title("NC for calibration dataset(Heading Change)")
        g2 = plt.figure(6)
        plt.plot(self.calibration_nonconformities[:,1])
        plt.title("NC for calibration dataset(Speed)")
        """
    
    def evaluate(self, data_test):
        X_test = data_test[0:8].reshape(1, -1)
        y_test = data_test[-2:].reshape(1,-1)
        neighbors_dists, neighbors_indices = self.nbrs.kneighbors(X_test[0:8])
        test_nbrs_outputs = self.y_train[neighbors_indices]
        test_nbrs_outputs_mean = np.mean(test_nbrs_outputs, axis=1)
        test_NC = np.abs(test_nbrs_outputs_mean-y_test)
        sum_distances = np.repeat(np.sum(neighbors_dists, axis=1).reshape(-1,1), 2, axis=1)
        sum_distances_median = np.median(sum_distances, axis=0)
        std_distances = np.repeat(np.std(neighbors_dists, axis=1).reshape(-1,1), 2, axis=1)
        std_output = np.std(test_nbrs_outputs,axis=1)
        std_output_median = np.median(std_output)
        #normalizer = np.exp(sum_distances*self.rho) + np.exp(std_output*self.ksi)
        #normalizer = np.abs(self.rho + sum_distances + std_output)
        #normalizer = np.exp(sum_distances/sum_distances_median*self.rho) + np.exp(std_output/std_output_median*self.ksi)
        normalizer = np.exp(sum_distances/float(self.n_neighbors)*self.rho) + np.exp(std_output*self.ksi)
        test_NC_normalized = test_NC / normalizer
        #print(test_NC_normalized)
        #print(self.calibration_nonconformities[int(len(self.calibration_nonconformities)*0.9)], normalizer)
        action_normalized90 = (self.calibration_nonconformities[int(len(self.calibration_nonconformities)*0.9)]*normalizer.reshape(2,))
        action_normalized95 = self.calibration_nonconformities[int(len(self.calibration_nonconformities)*0.95)]*normalizer.reshape(2,)
        action_normalized99 = self.calibration_nonconformities[int(len(self.calibration_nonconformities)*0.99)]*normalizer.reshape(2,)

        return action_normalized90, action_normalized95, action_normalized99, test_NC_normalized            