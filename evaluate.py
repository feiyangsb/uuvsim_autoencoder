#!/usr/bin/python3

from keras.models import load_model
import pandas as pd
import numpy as np

autoencoder = load_model('./autoencoder.h5')
data_path = './2/3.csv'
data = pd.read_csv(data_path, usecols=[0,1,2,3,4,5,6,7,8,9])
data = np.array(data) 
X = data[:,:8]
y = data[:,-2:]
X = X/[100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
X_predicted = autoencoder.predict(X)
X_predicted = X_predicted * [100.0, 100.0, 120.0, 10.0, 120.0, 120.0, 60.0, 60.0]
print(data[90])
print(X_predicted[90])