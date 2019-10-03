from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
import numpy as np
from keras.models import model_from_json
from routines.deep_svdd import deepSVDD
from routines.vae import VAE
from scipy import stats

class ICAD():
    def __init__(self, isTrajectory, isObstacle, trainingData, calibrationData, ncm=1):
        self.trainingData = trainingData
        self.calibrationData = calibrationData
        self.isTrajectory =isTrajectory
        self.isObstacle = isObstacle
        self.ncm = ncm
        self.sliding_window_length = 10
        
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
            
            # VAE
            elif self.ncm == 2:
                try:
                    print("Load the pretrained VAE model")
                    if self.isObstacle:
                        with open('./nnmodel/uuvsim/vae_architecture_obstacle.json','r') as f:
                            self.vae_model = model_from_json(f.read())
                        self.vae_model.load_weights('./nnmodel/uuvsim/vae_weights_obstacle.h5')
                    else:
                        with open('./nnmodel/uuvsim/vae_architecture.json','r') as f:
                            self.vae_model = model_from_json(f.read())
                        self.vae_model.load_weights('./nnmodel/uuvsim/vae_weights.h5')
                        
                except:
                    print("Cannot find the pretrained model, training it from start")
                    vae = VAE(self.trainingData)
                    self.vae_model = vae.fit()
                    # save the model and center
                    if self.isObstacle:
                        self.vae_model.save_weights('./nnmodel/uuvsim/vae_weights_obstacle.h5')
                        with open('./nnmodel/uuvsim/vae_architecture_obstacle.json', 'w') as f:
                            f.write(self.vae_model.to_json())
                    else:
                        self.vae_model.save_weights('./nnmodel/uuvsim/vae_weights.h5')
                        with open('./nnmodel/uuvsim/vae_architecture.json', 'w') as f:
                            f.write(self.vae_model.to_json())
                reconstructed_inputs = self.vae_model.predict(self.calibrationData)
                self.calibration_NC = np.square(reconstructed_inputs - self.calibrationData).mean(axis=1)
                self.calibration_NC.sort()

            # SVDD
            elif self.ncm == 4:
                try:
                    print("Load the pretrained svdd model")
                    if self.isObstacle:
                        with open('./nnmodel/uuvsim/svdd_architecture_obstacle.json','r') as f:
                            self.svdd_model = model_from_json(f.read())
                        self.svdd_model.load_weights('./nnmodel/uuvsim/svdd_weights_obstacle.h5')
                        self.center = np.load('./nnmodel/uuvsim/svdd_center_obstacle.npy')
                    else:
                        with open('./nnmodel/uuvsim/svdd_architecture.json','r') as f:
                            self.svdd_model = model_from_json(f.read())
                        self.svdd_model.load_weights('./nnmodel/uuvsim/svdd_weights.h5')
                        self.center = np.load('./nnmodel/uuvsim/svdd_center.npy')
                        
                except:
                    print("Cannot find the pretrained model, training it from start")
                    svdd = deepSVDD(self.trainingData)
                    self.svdd_model, self.center, radius = svdd.fit()
                    # save the model and center
                    if self.isObstacle:
                        self.svdd_model.save_weights('./nnmodel/uuvsim/svdd_weights_obstacle.h5')
                        with open('./nnmodel/uuvsim/svdd_architecture_obstacle.json', 'w') as f:
                            f.write(self.svdd_model.to_json())
                        np.save('./nnmodel/uuvsim/svdd_center_obstacle', self.center)
                    else:
                        self.svdd_model.save_weights('./nnmodel/uuvsim/svdd_weights.h5')
                        with open('./nnmodel/uuvsim/svdd_architecture.json', 'w') as f:
                            f.write(self.svdd_model.to_json())
                        np.save('./nnmodel/uuvsim/svdd_center', self.center)
                
                reps = self.svdd_model.predict(self.calibrationData)
                dists = np.sum((reps-self.center) ** 2, axis=1)
                self.calibration_NC = dists
                self.calibration_NC.sort()


            # TODO: need to add the new ncm 
            else:
                raise Exception('This nonconformity measure has not been implemented.')


            
    
    def __point_evaluate(self, data_point):
        data_point = data_point.reshape(1,8)
        if self.ncm == 1:
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
        if self.ncm == 2:
            if data_point[0][0] == None:
                return None
            else:
                reconstructed_input = self.vae_model.predict(data_point)
                reconstruction_error = np.square(reconstructed_input - data_point).mean(axis=1)
                p = (100 - stats.percentileofscore(self.calibration_NC, reconstruction_error))/float(100)
                return p

        if self.ncm == 4:
            if data_point[0][0] == None:
                return None
            else:
                rep = self.svdd_model.predict(data_point)
                dist = np.sum((rep-self.center)**2, axis=1)
                p = (100 - stats.percentileofscore(self.calibration_NC, dist))/float(100)
                return p

    
    def __traj_evaluate(self, data_point):
        data_point = data_point.reshape(1,8)
        if data_point[0][0] == None:
            self.traj = np.empty([0,8])
            return None
        else:
            if self.isObstacle:
                self.traj = np.concatenate((self.traj, data_point))
                v_row = []
                for i in range(len(self.trainingData)):
                    v_row.append(directed_hausdorff(self.traj, self.trainingData[i])[0])
                v_row.sort()
                test_NC = sum(v_row[0:self.k])
                p = sum(ele > test_NC for ele in self.calibration_NC) / float(len(self.calibration_NC))
                return p
            else:
                if len(self.traj)<self.sliding_window_length:
                    self.traj = np.concatenate((self.traj, data_point))
                else:
                    self.traj[:-1] = self.traj[1:]
                    self.traj[-1] = data_point

                v_row = []
                for i in range(len(self.trainingData)):
                    v_row.append(directed_hausdorff(self.traj, self.trainingData[i])[0])
                v_row.sort()
                test_NC = sum(v_row[0:self.k])
                p = sum(ele > test_NC for ele in self.calibration_NC) / float(len(self.calibration_NC))
                return p


    def __call__(self, data_point):
        data_point = data_point[:8]
        if self.isTrajectory:
            p = self.__traj_evaluate(data_point)
        else:
            p = self.__point_evaluate(data_point)
        return p