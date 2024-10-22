import numpy as np
import pandas as pd
from tqdm import tqdm


def create_test_set_latest(path_to_test_set, path_to_input_set):
    dir = ['perf_75', 'perf_50', 'perf_25']

    print('CREATING TEST DATA')

    for i, dir_ in tqdm(enumerate(dir)):
        # Load test set
        data = pd.read_csv(path_to_test_set + dir_ + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_mu_1_' + dir_ + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


def create_test_set(path_to_test_set, path_to_input_set, number_of_sets):
    print('CREATING TEST DATA')

    for i in tqdm(range(0, number_of_sets)):
        # Load test set
        data = pd.read_csv(path_to_test_set + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_' + str(i) + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')

# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_mu_latest(path_to_test_set, path_to_input_set, number_of_tests):
    print('CREATING TEST DATA')
    mu_list = ['grip_1_perf_100', 'grip_06_perf_100', 'grip_06_perf_75', 'grip_06_perf_50', 'grip_08_perf_100',
              'grip_08_perf_75', 'grip_08_perf_50']

    for i in tqdm(range(0, number_of_tests, 1)):
        # Load test set
        data = pd.read_csv(path_to_test_set + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_mu_' + mu_list[i] + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')

def create_test_set_mu(path_to_test_set, path_to_input_set, number_of_tests):
    print('CREATING TEST DATA')
    mu_set = ['grip_1_perf_100', 'grip_06_perf_100', 'grip_06_perf_75', 'grip_06_perf_50', 'grip_08_perf_100',
              'grip_08_perf_75', 'grip_08_perf_50']

    for i, mu_set_ in tqdm(enumerate(mu_set)):
        # Load test set
        data = pd.read_csv(path_to_test_set + mu_set_ + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_mu_' + mu_set_.replace('grip_', '') + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_mass(path_to_test_set, path_to_input_set, number_of_tests):
    print('CREATING TEST DATA')

    for i in tqdm(range(0, number_of_tests, 1)):
        # Load test set
        data = pd.read_csv(path_to_test_set + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_mass_' + str(i) + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------


def create_training_set(path_to_training_set, path_to_input_set, number_of_sets):
    sum_ = 0
    directories = ['mid_low_perf/', 'mid_perf/', 'mid_high_perf/', 'high_perf/']

    print('CREATING TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATING TRAINING DATA from ' + dir_)
        for i in tqdm(range(0, number_of_sets, 1)):
            # Load set
            data = pd.read_csv(path_to_training_set + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            # print(data.head())

            # Columns of interest
            # Torques
            trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
            trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
            tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
            tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

            tl = trl + tfl
            tr = trr + tfr

            # Brake pressures
            bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
            bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
            bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

            bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
            bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
            bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

            # Steering
            steer = data['driver_demands.steering'].to_numpy().astype(float)

            # Accelerations
            ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
            ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
            yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

            # Velocities
            ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
            uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

            # Create test_set
            set_ = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
            # print(set_.shape)

            # Save set
            dataframe = pd.DataFrame(set_)
            dataframe.to_csv(path_to_input_set + 'data_to_train_' + str(k * number_of_sets + i) + '.csv', index=False,
                             header=False)

            sum_ += len(set_)
        print('END CREATIon TRAINING DATA from ' + dir_)

    print('Total length: ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_param_study(path_to_data, path_to_output_):
    directories = ['/grip_1_perf_100/', '/grip_1_perf_75/', '/grip_1_perf_50/',
                   '/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/',
                   '/grip_08_perf_100/', '/grip_08_perf_75/', '/grip_08_perf_50/']

    print('CREATING TEST DATA')
    for i, dir_ in tqdm(enumerate(directories)):
        # Load test set
        data = pd.read_csv(path_to_test_set + dir_ + 'DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_input_set + '/test_set_paramstudy_' + dir_.replace('/' , '') + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_piste_training_complete(path_to_data, path_to_out, number_of_sets):
    for i in tqdm(range(1, number_of_sets + 1, 1)):
        # Load test set
        data = pd.read_csv(path_to_data + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy().astype(float)
        trr = data['differential_torques.output_torque_right_rear'].to_numpy().astype(float)
        tfl = data['differential_torques.output_torque_left_front'].to_numpy().astype(float)
        tfr = data['differential_torques.output_torque_right_front'].to_numpy().astype(float)

        tl = trl + tfl
        tr = trr + tfr

        # Brake pressures
        bp_fl = data['Brake.Chamber_Pressure.L1'].to_numpy().astype(float)
        bp_fr = data['Brake.Chamber_Pressure.R1'].to_numpy().astype(float)
        bp_f = ((bp_fl + bp_fr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        bp_rl = data['Brake.Chamber_Pressure.L2'].to_numpy().astype(float)
        bp_rr = data['Brake.Chamber_Pressure.R2'].to_numpy().astype(float)
        bp_r = ((bp_rl + bp_rr) / 2) * 10  # average of the two lines and conversion from Mpa to bar

        # Steering
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Create test_set
        test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_out + '/test_set_' + str(i) + '.csv', index=False, header=False)


# ----------------------------------------------------------------------------------------------------------------------

"""path_to_input_set = 'inputs/trainingdata/piste_training_complete/'
create_piste_training_complete(path_to_test_set, path_to_input_set, 32)"""

# Create test set
# path_to_test_set = '../../CRT_data/parameters_study/'
path_to_test_set = '../../CRT_data/parameters_study'
path_to_input_set = 'inputs/trainingdata/new/'
# create_test_set(path_to_test_set, path_to_input_set, 5)
# create_test_set_mass(path_to_test_set, path_to_input_set, 6)
create_test_set_param_study(path_to_test_set, path_to_input_set)

# Create training datasets
"""path_to_training_set = '../../CRT_data/'
path_to_input_set = path_to_input_set
create_training_set(path_to_training_set, path_to_input_set, 17)"""

# mu test
"""path_to_test_set = '../../CRT_data/'
path_to_input_set = 'inputs/trainingdata//'
create_test_set_mu(path_to_test_set, path_to_input_set, 7)"""

path_to_input_set = 'inputs/trainingdata/latest/'

"""path_to_test_set = '../../new_CRT/test_car_perf/'
create_test_set_latest(path_to_test_set, path_to_input_set)"""

"""path_to_test_set = '../../new_CRT/test_road_grip/'
create_test_set_mu(path_to_test_set, path_to_input_set, 7)"""

