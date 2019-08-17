#!/usr/bin/python3
from keras.models import model_from_json
from routines.data_loader_uuv import testDataLoader, dataLoader
import numpy as np
from scipy import stats

with open('svdd_architecture.json', 'r') as f:
    model = model_from_json(f.read())


center = np.array([0.1, 0.29649734, 0.14629067, 0.22795035]).reshape(1,4)
model.load_weights('svdd_weights.h5')

data_loader = dataLoader()
(_, _), (X_calibration, _) = data_loader.load()

reps = model.predict(X_calibration)
dists = np.sum((reps-center) ** 2, axis=1)
calibration_NC = np.sort(dists)

test_list = testDataLoader("./2/10.csv", isObstacle=False)
for i in range(len(test_list)):
    rep = model.predict(test_list[i].reshape(1,8))
    dist = np.sum((rep - center) ** 2, axis=1)
    p = (100 - stats.percentileofscore(calibration_NC, dist))/float(100)
    print(dist,p)
