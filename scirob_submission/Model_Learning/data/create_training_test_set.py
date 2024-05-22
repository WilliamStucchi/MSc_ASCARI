import pandas as pd
import numpy as np
from tqdm import tqdm


def create_training_set(path_to_data, path_to_output, number_of_sets, step):
    sum_ = 0
    directories = ['mid_low_perf/', 'mid_perf/', 'mid_high_perf/', 'high_perf/']
    result = []
    for i in range(number_of_sets*len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(4)
        uy = np.zeros(4)
        ux = np.zeros(4)
        steer = np.zeros(4)
        fx = np.zeros(4)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            result[k * number_of_sets + i] = np.zeros((len(data), 5 * 4 + 3))  # 5 inputs, 4 steps in the past, 3 objectives

            count = 0
            for idx in range(0, len(data) - 5 + 1, step):
                # print(idx)
                for j in range(0, 4):
                    yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + 4])
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + 4])
                ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + 4])

                result[k * number_of_sets + i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                                    yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                                    yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                                    yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                                    yaw_acc, ay, ax]
                count += 1

                if idx < (len(data) - 5):
                    ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                else:
                    for j in range(0, 4):
                        ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    np.savetxt(path_to_output + 'train_ax_' + str(step) + '.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '.csv', ay_to_plot, delimiter=',')

    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle' + str(step) + '.csv', result_filtered, delimiter=',')

    # Shuffle input
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step' + str(step) + '.csv', result_filtered, delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set_v1(path_to_data, path_to_output, number_of_sets, step):
    data = [None] * number_of_sets
    result = [None] * number_of_sets
    yaw_rate = np.zeros(4)
    uy = np.zeros(4)
    ux = np.zeros(4)
    steer = np.zeros(4)
    fx = np.zeros(4)

    for i in tqdm(range(0, number_of_sets)):
        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data[i] = pd.read_csv(path_to_data + '/flat/test_' + str(i + 1) + '/DemoSportsCar_mxp.csv', dtype=object)
        data[i] = data[i].drop(0, axis='rows')  # remove the row containing the measure units
        data[i].reset_index(drop=True, inplace=True)

        # print(len(data[i]))
        # input('Wait')

        result[i] = np.zeros((len(data[i]), 5 * 4 + 3))  # 5 inputs, 4 steps in the past, 3 objectives

        count = 0
        for idx in range(0, len(data[i]) - 5 + 1, step):
            # print(idx)
            for j in range(0, 4):
                yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data[i], idx + j)

            yaw_t = float(data[i]['Vehicle_States.yaw_angular_vel_wrt_road'][idx + 4])
            uy_t = float(data[i]['Vehicle_States.lateral_vel_wrt_road'][idx + 4])
            ux_t = float(data[i]['Vehicle_States.longitudinal_vel_wrt_road'][idx + 4])

            result[i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                                yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                                yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                                yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                                yaw_t, uy_t, ux_t]
            count += 1

    # Put together all data extracted
    result_flat = flatten_extend(result)

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]

    # Shuffle and Divide into datasets
    # train, dev, test = shuffle_and_divide(data, Param)
    train, dev, test = shuffle_and_divide(result_filtered, 0.8, 0.15)

    # Write out the first generated dataset , dev=dev, test=test
    np.savez(path_to_output + 'train_data_step' + str(step),
             train_f=train[0],
             train_t=train[1],
             dev_f=dev[0],
             dev_t=dev[1],
             test_f=test[0],
             test_t=test[1])


# ----------------------------------------------------------------------------------------------------------------------

def shuffle_and_divide(gen_data, train_perc, dev_perc):
    # first shuffle the data generated
    # This is necessary for the two friction data.
    np.random.seed(1)
    np.random.shuffle(gen_data)

    # split into features and targets
    features = gen_data[:, 0:-3]
    targets = gen_data[:, -3:]

    # split into train, dev, and test sets

    # calculate train index
    train_ind = int(train_perc * features.shape[0])
    dev_ind = int((train_perc + dev_perc) * features.shape[0])

    # do the splitting
    train = (features[0:train_ind, :], targets[0:train_ind, :])

    dev = (features[train_ind:dev_ind, :], targets[train_ind:dev_ind, :])

    test = (features[dev_ind:, :], targets[dev_ind:, :])

    return train, dev, test

# ----------------------------------------------------------------------------------------------------------------------

def flatten_extend(matrix):
    flat_list = []
    for elem in matrix:
        flat_list.extend(elem)
    return flat_list


# ----------------------------------------------------------------------------------------------------------------------

def pick_data_idx(data, idx):
    yaw_rate = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
    uy = float(data['Vehicle_States.lateral_vel_wrt_road'][idx])
    ux = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
    steer = float(data['driver_demands.steering'][idx])
    """# old version
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    fr = (frl + frr) / 2
    ff = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    fx = fr + ff"""

    # new version
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    fr = (frl + frr) / 2
    ffl = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    ffr = float(data['Tire.Ground_Surface_Force_X.R1'][idx])
    ff = (ffl + ffr) / 2
    fx = (ff + fr) / 2

    return yaw_rate, uy, ux, steer, fx


# ----------------------------------------------------------------------------------------------------------------------

def pick_acc_idx(data, idx):
    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx])
    ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx])

    return ax, ay

# ----------------------------------------------------------------------------------------------------------------------

def create_test_set(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_' + str(i+1) + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_piste_training_complete(data_, output_, number_of_sets):
    print('CREATION DATASET FROM TRAINING DATA')
    for i in tqdm(range(1, number_of_sets + 1)):
        data = pd.read_csv(data_ + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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
        # TODO: CONTROLLARE COSA SUCCEDE QUANDO FACCIO QUESTA OPERAZIONE DI NORMALIZZAZIONE
        # TODO: FA LA MEDIA CORRETTAMENTE? E SE IO VOLESSI PRENDERE LA FORZA ANTERIORE IN FRENATA
        #       E LA FORZA POSTERIORE IN ACCELERAZIONE?
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(output_ + 'test_set_' + str(i+1) + '.csv', index=False, header=False)
    print('END CREATION DATASET FROM TRAINING DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set_accel(path_to_data, path_to_output, number_of_sets, step):
    print('CREATION TRAINING DATA')
    data = [None] * number_of_sets
    result = [None] * number_of_sets
    yaw_rate = np.zeros(4)
    uy = np.zeros(4)
    ux = np.zeros(4)
    steer = np.zeros(4)
    fx = np.zeros(4)

    for i in tqdm(range(0, number_of_sets)):
        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data[i] = pd.read_csv(path_to_data + '/flat/test_' + str(i + 1) + '/DemoSportsCar_mxp.csv', dtype=object)
        data[i] = data[i].drop(0, axis='rows')  # remove the row containing the measure units
        data[i].reset_index(drop=True, inplace=True)

        ay_to_plot = np.zeros(len(data[i]))
        ax_to_plot = np.zeros(len(data[i]))

        # print(len(data[i]))
        # input('Wait')

        result[i] = np.zeros((len(data[i]), 5 * 4 + 1))  # 5 inputs, 4 steps in the past, 1 objectives

        count = 0
        for idx in range(0, len(data[i]) - 5 + 1, step):
            # print(idx)
            for j in range(0, 4):
                yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data[i], idx + j)

            ay = float(data[i]['Vehicle_States.lateral_acc_wrt_road'][idx + 4])

            result[i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                                yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                                yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                                yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                                ay]
            count += 1

            if idx < (len(data[i]) - 5):
                ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data[i], idx)
            else:
                for j in range(0, 4):
                    ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data[i], idx + j)

    np.savetxt(path_to_output + 'train_ax_' + str(step) + '.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '.csv', ay_to_plot, delimiter=',')

    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle' + str(step) + '.csv', result_filtered, delimiter=',')

    # Shuffle input
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step' + str(step) + '.csv', result_filtered, delimiter=',')

    print('END CREATION TRAINING DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_accel(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(1, number_of_sets + 1)):
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
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_' + str(i) + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------
