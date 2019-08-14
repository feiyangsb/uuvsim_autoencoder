#!/usr/bin/python3
import argparse
from argparse import RawTextHelpFormatter
from routines.data_loader_uuv import dataLoader, testDataLoader
from routines.icad import ICAD
import os
from routines.martingales import RPM, SMM, PIM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose different modes for inductive anomaly detection.', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-t","--trajectory",help="anomaly detection based on the time trajectories", action="store_true")
    parser.add_argument("-o","--obstacle",help="choose the obstacle avoidance dataset", action="store_true")
    parser.add_argument("-n","--nonconformity", type=int, help="choose norconformity measure (only for point anomaly detection):\n"
                                                        "1. Sum of the distances to k-nearest neighbors \n"
                                                        "2. Reconstruction error \n"
                                                        "3. Sum of the reconstruction errors of k-nearest neighbors in latent space \n"
                                                        "4. SVDD")
    parser.add_argument("-p","--path", help="test data path")
                                                
    args = parser.parse_args()
    try:
        if (args.trajectory and args.nonconformity):
            raise Exception('The nonconformity measure just for single point anomaly detection. For trajectory, Hausdorff distance is the only option of the nonconformity measure.')
        if args.nonconformity == None:
            args.nonconformity = 1
        data_loader = dataLoader(args.trajectory, args.obstacle)
        (X_train, _), (X_calibration, _) = data_loader.load()
        icad = ICAD(args.trajectory, X_train, X_calibration, args.nonconformity)
        data = testDataLoader(os.path.abspath(args.path), args.obstacle)

        p_list = []
        if not args.trajectory:
            RPM_list = []
            SMM_list = []
            PIM_list = []
            epsilon = 0.75
            rpm = RPM(epsilon)
            smm = SMM()
            pim = PIM()

        for i in range(len(data)):
            p = icad(data[i])
            p_list.append(p)
            if not args.trajectory:
                if data[i][0] == None:
                    rpm = RPM(0.75)
                    smm = SMM()
                    pim = PIM()
                    RPM_list.append(None)
                    SMM_list.append(None)
                    PIM_list.append(None)
                else:
                    RPM_list.append(rpm(p))
                    SMM_list.append(smm(p))
                    PIM_list.append(pim(p))

    except Exception as error:
        print('Caught this error: ' + repr(error))