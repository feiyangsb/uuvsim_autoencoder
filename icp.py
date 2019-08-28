#!/usr/bin/python3
import argparse
from argparse import RawTextHelpFormatter
from routines.data_loader_uuv import dataLoader, testDataLoader
from routines.icad import ICAD
import os
from routines.martingales import RPM, SMM, PIM
import matplotlib.pyplot as plt
from routines.ICP import ICP 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose different modes for inductive anomaly detection.', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-t","--trajectory",help="anomaly detection based on the time trajectories", action="store_true")
    parser.add_argument("-o","--obstacle",help="choose the obstacle avoidance dataset", action="store_true")
    parser.add_argument("-p","--path", help="test data path")
                                                
    args = parser.parse_args()
    data_loader = dataLoader(args.trajectory, args.obstacle)
    (X_train, Y_train), (X_calibration, Y_calibration) = data_loader.load()
    icp = ICP(X_train, Y_train, X_calibration, Y_calibration, 20)
    data = testDataLoader(os.path.abspath(args.path), args.obstacle)
    cw_heading_list = []
    cw_speed_list = []
    for i in range(len(data)):
        if data[i][0] == None:
            cw_heading_list.append(None)
            cw_speed_list.append(None)
        else:
            action_normalized90, _, _, _ = icp.evaluate(data[i])
            cw_heading_list.append(1.0/action_normalized90[0])
            cw_speed_list.append(1.0/action_normalized90[1])
             
    g3 = plt.figure(3)
    plt.plot(cw_heading_list,'g')
    plt.ylim([0,30])
    plt.xlabel("Time(S)")
    plt.ylabel("1/Confidence Width for Heading Change (Obstacle)")
    g4 = plt.figure(4)
    plt.plot(cw_speed_list, 'g')
    plt.ylim([0,110])
    plt.xlabel("Time(S)")
    plt.ylabel("1/Confidence Width for Speed (Obstacle)")
    plt.show()