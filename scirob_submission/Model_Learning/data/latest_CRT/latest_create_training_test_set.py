import pandas as pd
import numpy as np
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------

def pick_data_idx(data, idx):
    yaw_rate = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
    uy = float(data['Vehicle_States.lateral_vel_wrt_road'][idx])
    ux = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
    steer = float(data['driver_demands.steering'][idx])

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

def flatten_extend(matrix):
    flat_list = []
    for elem in matrix:
        flat_list.extend(elem)
    return flat_list


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                 vy_estimation=False, include_grip=False):
    len_mu1_perf100 = 0
    len_others = 0
    enter = True

    input_shape = 5
    output_shape = 3
    sum_ = 0

    name_file = 'mu106'
    directories = ['/train_car_perf/perf_100/', '/train_car_perf/perf_75/', '/train_car_perf/perf_50/',
                   '/train_road_grip/grip_06_perf_100/', '/train_road_grip/grip_06_perf_75/',
                   '/train_road_grip/grip_06_perf_50/']

    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here
    filename = 'DemoSportsCar_mxp.csv'

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

        if 'const_speed' in dir_:
            number_of_sets = 9
            filename = 'DemoSportsCar_mnt.csv'

        len_others = 0

        for i in tqdm(range(0, number_of_sets)):
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            """if '/train_car_perf/perf_100/' in dir_:
                len_mu1_perf100 += len(data)
                enter = True
            else:
                if len_others >= len_mu1_perf100:
                    enter = False
                else:
                    enter = True
                    len_others += len(data)"""

            if enter:
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
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_' + name_file + '.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_' + name_file + '.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_' + name_file + '.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    """# Load bicycle data
    bicycle_data = np.loadtxt('../data/new/bicycle_model_2grip.csv', delimiter=',')
    result_flat = np.concatenate((result_flat, bicycle_data), axis=0)"""

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_' + name_file + '.csv', result_filtered,
               delimiter=',')

    # Shuffle input
    print('SHUFFLING')
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_' + name_file + '.csv', result_filtered, delimiter=',')
    print('Saving training at: ' + path_to_output + 'train_data_step_' + str(step) + '_' + name_file + '.csv')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_car_perf(path_to_data, path_to_output_):
    perf = ['perf_75', 'perf_50', 'perf_25']
    print('CREATION TEST DATA')
    for i, perf_ in tqdm(enumerate(perf)):
        data = pd.read_csv(path_to_data + 'test_car_perf/' + perf_ + '/DemoSportsCar_mxp.csv', dtype=object)
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
        dataframe.to_csv(path_to_output_ + 'test_set_mu_1_' + perf_ + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------


def create_test_set_road_grip(path_to_data, path_to_output_):
    mu_set = ['grip_1_perf_100', 'grip_06_perf_100', 'grip_06_perf_75', 'grip_06_perf_50', 'grip_08_perf_100',
              'grip_08_perf_75', 'grip_08_perf_50']

    print('CREATION TEST DATA')
    for i, set_ in tqdm(enumerate(mu_set)):
        data = pd.read_csv(path_to_data + 'test_road_grip/' + set_ + '/DemoSportsCar_mxp.csv', dtype=object)
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
        dataframe.to_csv(path_to_output_ + 'test_set_mu_' + set_.replace('grip_', '') + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')