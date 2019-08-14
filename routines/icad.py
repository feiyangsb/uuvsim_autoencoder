class ICAD():
    def __init__(self, isTrajectory, trainingData, calibrationData):
        self.trainingData = trainingData
        self.calibrationData = calibrationData
        self.isTrajectory =isTrajectory
    
    def __point_evaluate(self):
        print("checkpoint 1")
    
    def __traj_evaluate(self):
        print("checkpoint 2")
    
    def __call__(self):
        if self.isTrajectory:
            self.__traj_evaluate()
        else:
            self.__point_evaluate()
    