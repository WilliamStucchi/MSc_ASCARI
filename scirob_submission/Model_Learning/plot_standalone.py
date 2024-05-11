import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def plot_run(filepath2results: str,
             filepath2testdata: str,
             filepath2plots: str,
             counter,
             start):
    """Plots test results of comparison between neural network and provided vehicle data.

    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param counter: [description]
    :type counter: [type]
    :param start: [description]
    :type start: [type]
    """

    # load results
    with open(filepath2results, 'r') as fh:
        results = np.loadtxt(fh)

    # load label data
    with open(filepath2testdata, 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    vx_result = results[:, 0][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    yaw_result = results[:, 2][:, np.newaxis]

    vx_label = labels[start:1500 + start, 0][:, np.newaxis]
    vy_label = labels[start:1500 + start, 1][:, np.newaxis]
    yaw_label = labels[start:1500 + start, 2][:, np.newaxis]

    yaw_diff = yaw_label - yaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result

    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))

    scaler_temp_result = np.concatenate((vx_result, vy_result, yaw_result), axis=1)
    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    vx_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    yaw_result_scaled = scaler_temp_result[:, 2]

    vx_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    yaw_label_scaled = scaler_temp_label[:, 2]

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: ' + 'Test No. ' + str(counter))

    data = np.asarray([mean_squared_error(yaw_label, yaw_result),
                       mean_squared_error(vx_label, vx_result),
                       mean_squared_error(vy_label, vy_result),
                       mean_absolute_error(yaw_label, yaw_result),
                       mean_absolute_error(vx_label, vx_result),
                       mean_absolute_error(vy_label, vy_result)]).reshape(2, 5).round(round_digits)

    column_header = ['yaw rate', 'long. vel. vx', 'lat. vel. vy']
    row_header = ['MSE', 'MAE']

    row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('MSE AND MAE OF SCALED VALUES: ' + 'Test No. ' + str(counter))

    data = np.asarray([mean_squared_error(yaw_label_scaled, yaw_result_scaled),
                       mean_squared_error(vx_label_scaled, vx_result_scaled),
                       mean_squared_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(yaw_label_scaled, yaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled)]).reshape(2, 5).round(round_digits)

    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(yaw_result, yaw_label, yaw_diff, 'Yaw rate in rad/s',
                  filepath2plots + 'yaw' + str(counter) + '.png')
    plot_and_save(vy_result, vy_label, vy_diff, 'Lat. vel. vy in m/s',
                  filepath2plots + 'vy' + str(counter) + '.png')
    plot_and_save(vx_result, vx_label, vx_diff, 'Long. vel. vx in m/s',
                  filepath2plots + 'vx' + str(counter) + '.png')


# ----------------------------------------------------------------------------------------------------------------------


def plot_run_test_CRT(filepath2results: str,
                      filepath2testdata: str,
                      filepath2plots: str,
                      counter: int):
    """Plots test results of comparison between neural network and provided vehicle data.

    :param counter:
    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    """

    # load results
    with open(filepath2results, 'r') as fh:
        results = np.loadtxt(fh)

    # load label data
    with open(filepath2testdata, 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    vx_result = results[:, 2][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    yaw_result = results[:, 0][:, np.newaxis]

    vx_label = labels[:, 0][:, np.newaxis]
    vy_label = labels[:, 1][:, np.newaxis]
    yaw_label = labels[:, 2][:, np.newaxis]

    yaw_diff = yaw_label - yaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result

    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))

    scaler_temp_result = np.concatenate((vx_result, vy_result, yaw_result), axis=1)
    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    vx_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    yaw_result_scaled = scaler_temp_result[:, 2]

    vx_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    yaw_label_scaled = scaler_temp_label[:, 2]

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: Test CRT')

    data = np.asarray([mean_squared_error(yaw_label, yaw_result),
                       mean_squared_error(vx_label, vx_result),
                       mean_squared_error(vy_label, vy_result),
                       mean_absolute_error(yaw_label, yaw_result),
                       mean_absolute_error(vx_label, vx_result),
                       mean_absolute_error(vy_label, vy_result)]).reshape(2, 3).round(round_digits)

    column_header = ['yaw rate', 'long. vel. vx', 'lat. vel. vy']
    row_header = ['MSE', 'MAE']

    row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    save_to_csv(data, 'MSE AND MAE OF UNSCALED VALUES Test CRT', filepath2plots, counter)

    print('MSE AND MAE OF SCALED VALUES: Test CRT')

    data = np.asarray([mean_squared_error(yaw_label_scaled, yaw_result_scaled),
                       mean_squared_error(vx_label_scaled, vx_result_scaled),
                       mean_squared_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(yaw_label_scaled, yaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled)]).reshape(2, 3).round(round_digits)

    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    save_to_csv(data, 'MSE AND MAE OF SCALED VALUES Test CRT', filepath2plots, counter)

    print('\n')

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(yaw_result, yaw_label, None, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(vy_result, vy_label, None, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.png')
    plot_and_save(vx_result, vx_label, None, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.png')

    # Difference
    plot_and_save(None, None, yaw_diff, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_diff_' + str(counter) + '.png')
    plot_and_save(None, None, vy_diff, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_diff_' + str(counter) + '.png')
    plot_and_save(None, None, vx_diff, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_diff_' + str(counter) + '.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(inp_1, inp_2, inp_3, value, savename):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    if inp_1 is not None:
        plt.plot(inp_1, label='Result', color='tab:orange')

    if inp_2 is not None:
        plt.plot(inp_2, label='Label', color='tab:blue')

    if inp_3 is not None:
        plt.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.0)

    plt.ylabel(value)
    plt.xlabel('Time steps (8 ms)')
    plt.legend()
    plt.grid()

    plt.savefig(savename, format='png')
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def save_to_csv(data, title, path_, counter):
    np.savetxt(path_ + title + ' ' + str(counter) + '.csv', data,
               header='long. vel. vx, lat. vel. vy, yaw rate, long. acc. ax, lat. acc. ay', delimiter=',')

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# TEST CRT
# ----------------------------------------------------------------------------------------------------------------------

path_to_results = 'results/step_4/callbacks/2024_05_10/09_39_23/results_test_'
path_to_data = '../../NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/test_set_'
path_to_plots = 'results/step_4/callbacks/2024_05_10/09_39_23/images/'

for num_tests in range(1, 3):
    index_last_underscore = path_to_results.rfind('_')
    path_to_results = path_to_results[:index_last_underscore + 1]
    path_to_results += str(num_tests) + '.csv'

    index_last_underscore = path_to_data.rfind('_')
    path_to_data = path_to_data[:index_last_underscore + 1]
    path_to_data += str(num_tests) + '.csv'

    plot_run_test_CRT(filepath2results=path_to_results,
                      filepath2testdata=path_to_data,
                      filepath2plots=path_to_plots,
                      counter=num_tests)

# ----------------------------------------------------------------------------------------------------------------------
# TEST TUM DATA
# ----------------------------------------------------------------------------------------------------------------------
"""
path_to_results = '../outputs/2024_05_06/09_52_48/matfiles/prediction_result_feedforward'
path_to_data = '../inputs/trainingdata/data_to_run.csv'
path_to_plots = '../outputs/2024_05_06/09_52_48/figures/'

for i_count in range(0, 10):

    idx_start = 0 + i_count * 1250

    # save and plot results (if activated in parameter file)
    plot_run(filepath2results=path_to_results+str(i_count)+'.csv',
             filepath2testdata=path_to_data,
             filepath2plots=path_to_plots,
             counter=i_count,
             start=idx_start)
"""