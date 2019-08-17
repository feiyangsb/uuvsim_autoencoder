#!/usr/bin/python3

from routines.data_loader_uuv import dataLoader
from routines.deep_svdd import deepSVDD
from keras.models import model_from_json

data_loader = dataLoader()
(X_train, Y_train), (X_calibration, Y_calibration) = data_loader.load()

"""
class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        self.radius = radius
        self.model = model
        self.cvar = cvar
    
    def on_epoch_end(self, batch, log={}):
        reps = self.model.predict(X_train)
        dist = np.sum((reps-self.cvar) ** 2, axis=1)
        val = np.sort(dist)
        R_new = np.percentile(val, nu*100)
        self.radius = R_new
"""
svdd = deepSVDD(X_train)    
model, center, radius = svdd.fit()

model.save_weights('svdd_weights.h5')
with open('svdd_architecture.json', 'w') as f:
    f.write(model.to_json())
model.save("svdd.h5")

print(center[0])
f = open("svdd_para.txt", "w+")
f.write(center)
for i in len(center):
    f.write(center[i])
f.close()