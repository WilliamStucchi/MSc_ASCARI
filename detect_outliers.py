import math

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------

def z_score_sd(elements):
    outliers = dict()

    for j in range(len(elements) - 4):
        mean = (elements[j] + elements[j+1] + elements[j+2] + elements[j+3]) / 4
        sd = math.sqrt((math.pow(elements[j] - mean, 2) + math.pow(elements[j+1] - mean, 2) +
                        math.pow(elements[j+2] - mean, 2) + math.pow(elements[j+3] - mean, 2)) / 4)

        score = abs((elements[j+4] - mean) / sd)

        if score >= 8:
            outliers.update({str(j+4): str(score)})

    if not outliers:
        return None
    else:
        return outliers


# ----------------------------------------------------------------------------------------------------------------------


# Acquire training data
directories = ['train_road_grip/grip_1/']

print('ACQUIRING TRAINING DATA')
for k, dir_ in enumerate(directories):
    print('ACQUIRING TRAINING DATA from ' + dir_)
    for i in tqdm(range(0, 17, 1)):
        # Load set
        data = pd.read_csv('../CRT_data/' + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        print('END ACQUISITION TRAINING DATA from ' + dir_ + 'test_' + str(i))

        print('START OUTLIER DETECTION')
        idxs = z_score_sd(yaw_rate)
        if idxs is not None:
            print('OUTLIERS FOUND IN yaw_rate: ', idxs)

        idxs = z_score_sd(uy)
        if idxs is not None:
            print('OUTLIERS FOUND IN lateral_velocity: ', idxs)

        idxs = z_score_sd(ux)
        if idxs is not None:
            print('OUTLIERS FOUND IN longitudinal_velocity: ', idxs)

        idxs = z_score_sd(steer)
        if idxs is not None:
            print('OUTLIERS FOUND IN steering_angle: ', idxs)

        idxs = z_score_sd(fx)
        if idxs is not None:
            print('OUTLIERS FOUND IN longitudinal_force: ', idxs)
