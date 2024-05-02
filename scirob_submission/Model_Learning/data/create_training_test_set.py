import pandas as pd
import numpy as np
from tqdm import tqdm


def create_training_set(path_to_data, path_to_output, number_of_sets, step):
    data = [None] * number_of_sets
    result = [None] * number_of_sets
    yaw_rate = np.zeros(4)
    uy = np.zeros(4)
    ux = np.zeros(4)
    steer = np.zeros(4)
    fx = np.zeros(4)

    for i in tqdm(range(0, number_of_sets)):
        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data[i] = pd.read_csv(path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data[i] = data[i].drop(0, axis='rows')  # remove the row containing the measure units
        data[i].reset_index(drop=True, inplace=True)

        # print(len(data[i]))
        # input('Wait')

        result[i] = np.zeros((len(data[i]), 5*4+3))  # 5 inputs, 4 steps in the past, 3 objectives

        count = 0
        for idx in range(0, len(data[i]) - 5 + 1, step):
            # print(idx)
            for j in range(0, 4):
                yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data[i], idx + j)

            yaw_acc = float(data[i]['Vehicle_States.yaw_angular_acc_wrt_road'][idx + 4])
            ay = float(data[i]['Vehicle_States.lateral_acc_wrt_road'][idx + 4])
            ax = float(data[i]['Vehicle_States.longitudinal_acc_wrt_road'][idx + 4])

            result[i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                           yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                           yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                           yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                           yaw_acc, ay, ax]
            count += 1

    # Put together all data extracted
    result_flat = flatten_extend(result)

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat==0, axis=1)]

    # Shuffle input
    np.random.seed(1)
    np.random.shuffle(result_filtered)

    np.savetxt('CRT/train_data_step'+str(step)+'.csv', result_filtered, delimiter=',')



def flatten_extend(matrix):
     flat_list = []
     for elem in matrix:
         flat_list.extend(elem)
     return flat_list


def pick_data_idx(data, idx):
    yaw_rate = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
    uy = float(data['Vehicle_States.lateral_vel_wrt_road'][idx])
    ux = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
    steer = float(data['driver_demands.steering'][idx])
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    fr = (frl + frr) / 2
    ff = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    fx = fr + ff

    return yaw_rate, uy, ux, steer, fx


def create_test_set(path_to_data, number_of_sets):
    for i in range(0, number_of_sets):
        data = pd.read_csv(path_to_data + '/test/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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
        ff = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        fx = fr + ff

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv('CRT/test_set_' + str(i) + '.csv', index=False, header=False)


path_to_data = '../../../../CRT_data'

"""
path_to_output = 'CRT/train_data_step_1'
create_training_set(path_to_data, path_to_output, 18, 1)
path_to_output = 'CRT/train_data_step_2'
create_training_set(path_to_data, path_to_output, 18, 2)
path_to_output = 'CRT/train_data_step_3'
create_training_set(path_to_data, path_to_output, 18, 3)
path_to_output = 'CRT/train_data_step_4'
create_training_set(path_to_data, path_to_output, 18, 4)
"""
create_test_set(path_to_data, 2)
