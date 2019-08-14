#!/usr/bin/python3
import scipy.integrate as integrate
from scipy import stats
import numpy as np

class RPM(object):
    def __init__(self, epsilon):
        self.M = 1.0
        self.epsilon = epsilon
    
    def __call__(self, p):
        if (p == 0.0):
            p = 0.05
        betting = self.epsilon *  (p ** (self.epsilon-1.0))
        self.M *= betting
        return self.M


class SMM(object):
    def __init__(self):
        self.p_list = []
    
    def __integrand(self, x):
        result = 1.0
        for i in range(len(self.p_list)):
            result *= x*(self.p_list[i]**(x-1.0))
        return result
    
    def __call__(self, p):
        if (p==0.0):
            p = 0.05
        self.p_list.append(p)
        M, _ = integrate.quad(self.__integrand, 0.0, 1.0)
        return M

class PIM(object):
    def __init__(self):
        self.extended_p_list = [0.5, -0.5, 1.5]#np.random.uniform(-1.0,2.0, 300).tolist()
        self.M = 1.0
    
    def __call__(self, p):
        array = np.array(self.extended_p_list).reshape(1, -1)
        kernel = stats.gaussian_kde(array, bw_method='silverman') 
        normalizer = kernel.integrate_box_1d(0.0, 1.0)
        betting = kernel(p)[0] / normalizer
        self.M *= betting
        print(self.M, betting)
        self.extended_p_list.append(p)
        self.extended_p_list.append(-p)
        self.extended_p_list.append(2.0-p)
        return self.M
