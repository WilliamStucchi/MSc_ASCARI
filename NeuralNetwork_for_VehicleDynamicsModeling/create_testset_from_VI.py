import numpy as np
import pandas as pd


def create_test_set(path_to_test_set):
    # Load test set
    data = pd.read_csv(path_to_test_set)

    # Columns of interest
    # Torques
    trl = data['differential_torques_output_torque_left_rear'].to_numpy()
    trr = data['differential_torques_output_torque_right_rear'].to_numpy()

    # Brake pressures
    bp_fl = data['Brake_Chamber_Pressure_L1'].to_numpy()
    bp_fr = data['Brake_Chamber_Pressure_R1'].to_numpy()
    bp_f = (bp_fl + bp_fr) / 2 * 10  # average of the two lines and conversion from Mpa to bar

    bp_rl = data['Brake_Chamber_Pressure_L2'].to_numpy()
    bp_rr = data['Brake_Chamber_Pressure_R2'].to_numpy()
    bp_r = (bp_rl + bp_rr) / 2 * 10  # average of the two lines and conversion from Mpa to bar

    # Steering
    steer = data['driver_demands_steering'].to_numpy()

    # Accelerations
    ax = data['Vehicle_States_longitudinal_acc_wrt_road'].to_numpy()
    ay = data['Vehicle_States_lateral_acc_wrt_road'].to_numpy()
    yaw_rate = data['Vehicle_States_yaw_angular_vel_wrt_road'].to_numpy()

    # Velocities
    ux = data['Vehicle_States_longitudinal_vel_wrt_road'].to_numpy()
    uy = data['Vehicle_States_lateral_vel_wrt_road'].to_numpy()

    # Create test_set
    test_set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, trl, trr, bp_f, bp_r]))
    print(test_set.shape)

    # Save test_set
    dataframe = pd.DataFrame(test_set)
    dataframe.to_csv('inputs/trainingdata/testset.csv', index=False, header=False)


def create_training_set(path_to_training_set, path_to_input_set, number_of_sets):
    for i in range(1,number_of_sets+1, 1):
        # Load set
        data = pd.read_csv(path_to_training_set + 'test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data = data.drop(0, axis='rows')  # remove the row containing the measure units

        print(data.head())

        # Columns of interest
        # Torques
        trl = data['differential_torques.output_torque_left_rear'].to_numpy()
        trr = data['differential_torques.output_torque_right_rear'].to_numpy()
        tfl = data['differential_torques.output_torque_left_front'].to_numpy()
        tfr = data['differential_torques.output_torque_right_front'].to_numpy()

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
        steer = data['driver_demands.steering'].to_numpy()

        # Accelerations
        ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy()
        ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy()
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy()

        # Velocities
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy()
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy()

        # Create test_set
        set = np.transpose(np.array([ux, uy, yaw_rate, ax, ay, steer, tl, tr, bp_f, bp_r]))
        print(set.shape)

        # Save set
        dataframe = pd.DataFrame(set)
        dataframe.to_csv(path_to_input_set+'data_to_train_'+str(i-1)+'.csv', index=False, header=False)


# Create test set
path_to_test_set = '../CRT_data/test/DemoSportsCar_data.csv'
create_test_set(path_to_test_set)

# Create training datasets
path_to_training_set = '../CRT_data/flat/'
path_to_input_set = 'inputs/trainingdata/CRT/'
create_training_set(path_to_training_set, path_to_input_set, 9)
