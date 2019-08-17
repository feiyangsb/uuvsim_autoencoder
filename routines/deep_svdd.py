from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
from keras.callbacks import Callback, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K

nu = 0.2

class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.radius = radius
        self.model = model
        self.inputs = X_train
        self.cvar = cvar
        self.y_reps = np.zeros((len(X_train), 4))

    def on_epoch_end(self, batch, logs={}):

        reps = self.model.predict(self.inputs)
        self.y_reps = reps
        center = self.cvar
        dist = np.sum((reps - self.cvar) ** 2, axis=1)
        scores = dist
        val = np.sort(scores)
        R_new = np.percentile(val, nu * 100)  # qth quantile of the radius.
        R_updated = R_new
        # print("[INFO:] Center (c)  Used.", center)
        # print("[INFO:] Updated Radius (R) .", R_updated)
        self.radius = R_new
        print("[INFO:] \n Updated Radius Value...", R_new)
        # print("[INFO:] \n Updated Rreps value..", self.y_reps)
        return self.radius


class deepSVDD():
    def __init__(self, X_train):
        print("Initialize the SVDD")
        self.inputs = X_train
        self.Rvar = 0.0
        self.cvar = 0.0
    
    def create_model(self):
        state_input = Input(shape=(8,))
        hidden_layer = Dense(6, activation='relu', use_bias=False)(state_input)
        output = Dense(4,activation='relu', use_bias=False)(hidden_layer)
        svdd = Model(state_input, output)
        svdd.summary()
        return svdd

    def initialize_c_with_mean(self, inputs, model):
        reps = model.predict(inputs)
        
        eps = 0.1
        c = np.mean(reps, axis=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c >= 0)] = eps

        self.cvar = c

        dist = np.sum((reps - c) ** 2, axis=1)
        val = np.sort(dist)
        self.Rvar = np.percentile(val, nu * 100)

        print("Radius initialized.", self.Rvar)

        # Custom loss SVDD_loss ball interpretation
    def custom_ocnn_hypershere_loss(self):

        center = self.cvar

        # val = np.ones(Cfg.mnist_rep_dim) * 0.5
        # center = K.variable(value=val)


        # define custom_obj_ball
        def custom_obj_ball(y_true, y_pred):
            # compute the distance from center of the circle to the

            dist = (K.sum(K.square(y_pred - center), axis=1))
            avg_distance_to_c = K.mean(dist)

            return (avg_distance_to_c)

        return custom_obj_ball

    def fit(self):
        self.model_svdd = self.create_model()
        self.initialize_c_with_mean(self.inputs, self.model_svdd)
        out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, self.inputs)

        def lr_scheduler(epoch):
            lr = 1e-4
            if epoch > 50:
                lr = 1e-5
                if(epoch== 51):
                    print('lr: rate adjusted for fine tuning %f' % lr)

            # print('lr: %f' % lr)
            return lr

        scheduler = LearningRateScheduler(lr_scheduler)
        opt = Adam(lr=1e-4)
        callbacks = [out_batch, scheduler]

        self.model_svdd.compile(loss=self.custom_ocnn_hypershere_loss(), optimizer=opt)
        y_reps = out_batch.y_reps

        self.model_svdd.fit(self.inputs, y_reps, shuffle=True, batch_size=64, epochs=150, validation_split=0.01, verbose=0, callbacks=callbacks)
        self.Rvar = out_batch.radius
        self.cvar = out_batch.cvar
        print(self.cvar, self.Rvar)
        return self.model_svdd, self.cvar, self.Rvar
    

