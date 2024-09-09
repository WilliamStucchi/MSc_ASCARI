import pandas as pd
import numpy as np
from tqdm import tqdm


def create_training_set_car_perf(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                 vy_estimation=False, include_grip=False):
    
    input_shape = 5
    output_shape = 3

    sum_ = 0
    directories = ['/train_car_perf/mid_perf/',
                   '/train_car_perf/mid_high_perf/', '/train_car_perf/high_perf/',
                   '/train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/']
    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here

    result = []
    for i in range(number_of_sets_ * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            if not include_grip:
                result[k * number_of_sets_ + i] = np.zeros(
                        (len(data),
                         input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives
            else:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape + 1))  # 5 inputs, 4 steps in the past, 4 objectives

            vy_est_prev = 0.0  # in case of estimation of vy
            vy_est = 0.0
            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                # print(idx)
                for j in range(0, timesteps_back):
                    yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])
                ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                if vy_estimation:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]

                        # vy_dot = ay - yaw_rate * vx
                        ay_obj = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + obj])
                        vy_dot = ay_obj - yaw_rate[obj] * ux[obj]

                        if obj == 0:
                            vy_est = vy_dot * 0.01 + vy_est_prev
                            vy_est_prev = vy_est
                        else:
                            vy_est = vy_dot * 0.01 + vy_est

                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = vy_est
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                else:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                    if include_grip:
                        if 'grip_06' in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 0.6
                        elif 'grip_08' in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 0.8
                        elif 'grip_' not in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 1.0

                count += 1

                if idx < (len(data) - (timesteps_back + 1)):
                    ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                else:
                    for j in range(0, timesteps_back):
                        ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_gripest.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_gripest.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_gripest.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_gripest.csv', result_filtered, delimiter=',')

    # Shuffle input
    print('SHUFFLING')
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_gripest.csv', result_filtered, delimiter=',')
    print('Saving training at: ' + path_to_output + 'train_data_step_' + str(step) + '_gripest.csv')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_scheduled_sampling(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4):
    input_shape = 5
    output_shape = 3

    sum_ = 0
    directories = ['/train_car_perf/mid_low_perf/', '/train_car_perf/mid_perf/', '/train_car_perf/mid_high_perf/',
                   '/train_car_perf/high_perf/', '/train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/', '/train_road_grip/grip_08/',
                   '/train_road_grip/grip_08_perf_04/', '/train_road_grip/grip_08_perf_06/']
    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here

    result = []
    for i in range(number_of_sets_ * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # ay threshold for selection of samples
            data['Vehicle_States.lateral_acc_wrt_road'] = pd.to_numeric(data['Vehicle_States.lateral_acc_wrt_road'], errors='coerce')
            th_ay = (data['Vehicle_States.lateral_acc_wrt_road']).abs().max() * 0.85

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            result[k * number_of_sets_ + i] = np.zeros(
                (len(data), input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives

            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])

                if abs(ay) >= th_ay:
                    yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                    for j in range(0, timesteps_back):
                        yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                    count += 1

                    if idx < (len(data) - (timesteps_back + 1)):
                        ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                    else:
                        for j in range(0, timesteps_back):
                            ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_sched_cplt.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_sched_cplt.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_sched_cplt.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_sched_cplt.csv', result_filtered, delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set_road_grip(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                  no_vy=False, vy_estimation=False):
    balancing_len = 0  # max len of data based on shortest directory
    balanced_len = 0  # len taken from each directory that must match balanced_len

    if no_vy:
        input_shape = 5 + 2
    else:
        input_shape = 5
    output_shape = 3

    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here

    sum_ = 0
    directories = ['train_car_perf/high_perf/', 'train_car_perf/mid_high_perf/', 'train_car_perf/mid_perf/',
                   'train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/', '/train_road_grip/grip_06_perf_03/',
                   'train_road_grip/grip_08/', 'train_road_grip/grip_08_perf_06/', 'train_road_grip/grip_08_perf_04/']

    result = []
    for i in range(number_of_sets * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        balanced_len = 0
        if k == 1:
            print('BALANCING ', balancing_len)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            # computation of len for balancing the sets
            if k == 0:
                balancing_len += len(data)
            elif balanced_len + len(data) <= balancing_len:
                balanced_len += len(data)
            elif balanced_len + len(data) > balancing_len and balanced_len < balancing_len:
                print('BEFORE ', len(data))
                data.drop(data.tail(balancing_len - balanced_len).index, inplace=True)
                print('AFTER ', len(data))
                balanced_len += len(data)
            else:
                print('BALANCED ', balanced_len)
                break

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            if no_vy:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives
            else:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives

            vy_est_prev = 0.0
            vy_est = 0.0
            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                # print(idx)
                for j in range(0, timesteps_back):
                    yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])
                ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                if vy_estimation:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]

                        # vy_dot = ay - yaw_rate * vx
                        ay_obj = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + obj])
                        vy_dot = ay_obj - yaw_rate[obj] * ux[obj]

                        if obj == 0:
                            vy_est = vy_dot * 0.01 + vy_est_prev
                            vy_est_prev = vy_est
                        else:
                            vy_est = vy_dot * 0.01 + vy_est

                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = vy_est
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                else:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                count += 1

                if idx < (len(data) - (timesteps_back + 1)):
                    ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                else:
                    for j in range(0, timesteps_back):
                        ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_cplt.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay' + str(step) + '_cplt.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_cplt.csv', result_flat,
               delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_cplt.csv',
               result_filtered,
               delimiter=',')

    print('SHUFFLING')
    # Shuffle input
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_cplt.csv', result_filtered,
               delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------


def create_training_set_mass(path_to_data, path_to_output, number_of_sets, step):
    sum_ = 0
    directories = ['test_mass/']
    result = []
    for i in range(number_of_sets * len(directories)):
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

        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data = pd.read_csv(path_to_data + dir_ + 'test_2/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        ay_to_plot = np.zeros(len(data))
        ax_to_plot = np.zeros(len(data))

        # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
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

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_mass.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay' + str(step) + '_mass.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_mass.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step' + str(step) + '_mass.csv', result_filtered, delimiter=',')

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


def pick_data_idx_rl(data, idx):
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
    ffl = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    ffr = float(data['Tire.Ground_Surface_Force_X.R1'][idx])
    fl = (ffl + frl) / 2
    fr = (ffr + frr) / 2

    return yaw_rate, uy, ux, steer, ffr, ffl, frr, frl


# ----------------------------------------------------------------------------------------------------------------------

def pick_acc_idx(data, idx):
    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx])
    ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx])

    return ax, ay


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_car_perf(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_car_perf/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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

        """fr = (ffr + frr) / 2
        fl = (ffl + frl) / 2"""

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        test_set_rl = np.transpose(np.array([yaw_rate, ux, steer, ffr, ffl, frr, frl, uy]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_' + str(i) + '.csv', index=False, header=False)
        dataframe_rl = pd.DataFrame(test_set_rl)
        dataframe_rl.to_csv(path_to_output_ + 'test_set_' + str(i) + '_rrff.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_road_grip(path_to_data, path_to_output_, number_of_sets):
    mu_set = ['1', '08', '06', '0806', '0804', '06045', '0603']

    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_road_grip/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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

        # Save test set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_mu_' + mu_set[i] + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_static_equilibrium(path_to_output_, seconds_of_test):
    test_length = seconds_of_test * 100

    # Creating steering tests
    print('CREATING STEERING EQUILIBRA TEST')
    yaw_rate = np.full(test_length, 0.0)
    uy = np.full(test_length, 0.0)
    ux = np.full(test_length, 0.0)
    steer = np.full(test_length, 0.0)
    fx = np.full(test_length, 10)

    # Save test set
    test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

    # Save test set
    dataframe = pd.DataFrame(test_set)
    dataframe.to_csv(path_to_output_ + 'test_set_steer_equilibria.csv', index=False, header=False)
    print('END CREATION STEERING EQUILIBRA TEST')



# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_mass(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_mass/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
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
        dataframe.to_csv(path_to_output_ + 'test_set_mass_' + str(i) + '.csv', index=False, header=False)

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
        dataframe.to_csv(output_ + 'test_set_' + str(i + 1) + '.csv', index=False, header=False)
    print('END CREATION DATASET FROM TRAINING DATA')

# ----------------------------------------------------------------------------------------------------------------------
