import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt

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

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')


# ----------------------------------------------------------------------------------------------------------------------

def smooth(noisy_data):
    return moving_average(noisy_data, window_size=2000)

# ----------------------------------------------------------------------------------------------------------------------

def plot_run_test_CRT(filepath2results: str,
                      filepath2testdata: str,
                      filepath2resax: str,
                      filepath2resay: str,
                      # filepath2resgrip: str,
                      filepath2labelsaccel: str,
                      filepath2plots: str,
                      counter: str):
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

    with open(filepath2resax, 'r') as fh:
        res_ax = np.loadtxt(fh)

    with open(filepath2resay, 'r') as fh:
        res_ay = np.loadtxt(fh)

    """with open(filepath2resgrip, 'r') as fh:
        res_grip = np.loadtxt(fh)"""

    with open(filepath2labelsaccel, 'r') as fh:
        label_accel = np.loadtxt(fh, delimiter=',')

    # load label data
    with open(filepath2testdata, 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    vx_result = results[:, 2][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    yaw_result = results[:, 0][:, np.newaxis]
    ax_result = res_ax[:, np.newaxis]
    ay_result = res_ay[:, np.newaxis]

    """grip_result = res_grip[:, np.newaxis]
    grip_result_smooth = smooth(res_grip)
    grip_result_smooth = grip_result_smooth[:, np.newaxis]"""

    vx_label = labels[:, 2][:, np.newaxis]
    vy_label = labels[:, 1][:, np.newaxis]
    yaw_label = labels[:, 0][:, np.newaxis]
    ax_label = label_accel[:, 3][:, np.newaxis]
    ay_label = label_accel[:, 4][:, np.newaxis]

    if '06' in filepath2results and '08' not in filepath2results:
        grip_label = np.full((ay_label.shape[0]), 0.6, dtype=float)
    elif '08' in filepath2results:
        grip_label = np.full((ay_label.shape[0]), 0.8, dtype=float)
    else:
        grip_label = np.full((ay_label.shape[0]), 1.0, dtype=float)

    yaw_diff = yaw_label - yaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result
    ax_diff = ax_label - ax_result
    ay_diff = ay_label - ay_result

    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))

    scaler_temp_result = np.concatenate((vx_result, vy_result, yaw_result, ax_result, ay_result), axis=1)
    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label, ax_label, ay_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    vx_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    yaw_result_scaled = scaler_temp_result[:, 2]
    ax_result_scaled = scaler_temp_result[:, 3]
    ay_result_scaled = scaler_temp_result[:, 4]

    vx_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    yaw_label_scaled = scaler_temp_label[:, 2]
    ax_label_scaled = scaler_temp_label[:, 3]
    ay_label_scaled = scaler_temp_label[:, 4]

    print('Saving diff files')
    np.savetxt(filepath2results[:filepath2results.rfind('/')+1] + 'diff_vx_' + str(counter) + '.csv', vx_diff)
    np.savetxt(filepath2results[:filepath2results.rfind('/')+1] + 'diff_vy_' + str(counter) + '.csv', vy_diff)
    np.savetxt(filepath2results[:filepath2results.rfind('/')+1] + 'diff_yaw_' + str(counter) + '.csv', yaw_diff)
    np.savetxt(filepath2results[:filepath2results.rfind('/')+1] + 'diff_ax_' + str(counter) + '.csv', ax_diff)
    np.savetxt(filepath2results[:filepath2results.rfind('/')+1] + 'diff_ay_' + str(counter) + '.csv', ay_diff)

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: Test CRT ' + str(counter))

    data = np.asarray([mean_squared_error(yaw_label, yaw_result),
                       mean_squared_error(vx_label, vx_result),
                       mean_squared_error(vy_label, vy_result),
                       mean_squared_error(ax_label, ax_result),
                       mean_squared_error(ay_label, ay_result),
                       mean_absolute_error(yaw_label, yaw_result),
                       mean_absolute_error(vx_label, vx_result),
                       mean_absolute_error(vy_label, vy_result),
                       mean_absolute_error(ax_label, ax_result),
                       mean_absolute_error(ay_label, ay_result)]).reshape(2, 5).round(round_digits)

    column_header = ['yaw rate', 'long. vel. vx', 'lat. vel. vy', 'long. acc. ax', 'lat. acc. ay']
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
                       mean_squared_error(ax_label_scaled, ax_result_scaled),
                       mean_squared_error(ay_label_scaled, ay_result_scaled),
                       mean_absolute_error(yaw_label_scaled, yaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(ax_label_scaled, ax_result_scaled),
                       mean_absolute_error(ay_label_scaled, ay_result_scaled)]).reshape(2, 5).round(round_digits)

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
    plot_and_save(ax_result, ax_label, None, 'Long. accel. ax [m/s^2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.png')
    plot_and_save(ay_result, ay_label, None, 'Lat. accel. ay [m/s^2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.png')
    """plot_and_save(grip_result, grip_label, None, 'Grip level',
                  filepath2plots + 'grip_test_' + str(counter) + '.png')
    plot_and_save(grip_result_smooth, grip_label, None, 'Grip level smooth',
                  filepath2plots + 'grip_test_smooth_' + str(counter) + '.png')"""

    # Difference
    plot_and_save(None, None, yaw_diff, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_diff_' + str(counter) + '.png')
    plot_and_save(None, None, vy_diff, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_diff_' + str(counter) + '.png')
    plot_and_save(None, None, vx_diff, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_diff_' + str(counter) + '.png')
    plot_and_save(None, None, ax_diff, 'Long. accel. ax [m/s^2]',
                  filepath2plots + 'ax_diff_' + str(counter) + '.png')
    plot_and_save(None, None, ay_diff, 'Lat. accel. ay [m/s^2]',
                  filepath2plots + 'ay_diff_' + str(counter) + '.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(inp_1, inp_2, inp_3, value, savename):
    plt.figure(figsize=(25, 10))
    plt.rc('font', size=14)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=16)  # Titolo degli assi
    plt.rc('axes', labelsize=14)  # Etichette degli assi
    plt.rc('xtick', labelsize=12)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=12)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=14)  # Legenda
    ax = plt.gca()

    if 'yaw' in savename:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
    elif 'grip' in savename:
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))

    if inp_1 is not None:
        plt.plot(inp_1, label='Result', color='red', linewidth=1.5)

    if inp_2 is not None:
        plt.plot(inp_2, label='Label', color='blue', linewidth=1.5)

    if inp_3 is not None:
        plt.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.5)

    plt.ylabel(value)
    plt.xlabel('Time steps (10 ms)')
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
"""
path_to_results = 'results/step_4/callbacks/2024_05_10/09_39_23/results_test_'
path_to_data = '../../NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/test_set_'
path_to_plots = 'results/step_4/callbacks/2024_05_10/09_39_23/images/'
"""
"""path_to_results = 'results/step_1/callbacks/2024_09_02/16_42_37/results_test_mu_'
path_to_data = 'data/new/test_set_mu_'
path_to_plots = 'results/step_1/callbacks/2024_09_02/16_42_37/images/'
path_to_labels_for_accel = '../../NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_mu_'"""
path_to_results = 'results/step_1/callbacks/2024_10_02/16_30_34/results_test_mu_'
path_to_data = 'data/new/test_set_mu_'
path_to_plots = 'results/step_1/callbacks/2024_10_02/16_30_34/images/'
path_to_labels_for_accel = '../../NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_mu_'

for num_test in ['1', '08', '06', '0806', '0804', '06045', '0603']:

    path_to_results = path_to_results[:path_to_results.rfind('_') + 1] + num_test + '.csv'
    path_to_res_ax = path_to_results[:path_to_results.rfind('_') + 1] + num_test + '_ax.csv'
    path_to_res_ay = path_to_results[:path_to_results.rfind('_') + 1] + num_test + '_ay.csv'
    # path_to_res_grip = path_to_results[:path_to_results.rfind('_') + 1] + num_test + '_grip.csv'
    path_to_data = path_to_data[:path_to_data.rfind('_') + 1] + num_test + '.csv'
    path_to_labels_for_accel = (path_to_labels_for_accel[:path_to_labels_for_accel.rfind('_') + 1]
                                + num_test + '.csv')

    plot_run_test_CRT(filepath2results=path_to_results,
                      filepath2testdata=path_to_data,
                      filepath2resax=path_to_res_ax,
                      filepath2resay=path_to_res_ay,
                      #filepath2resgrip=path_to_res_grip,
                      filepath2labelsaccel=path_to_labels_for_accel,
                      filepath2plots=path_to_plots,
                      counter='mu_' + num_test)

"""for num_test in range(1, 4):

    path_to_results = path_to_results[:path_to_results.rfind('_') + 1] + str(num_test) + '.csv'
    path_to_res_ax = path_to_results[:path_to_results.rfind('_') + 1] + str(num_test) + '_ax.csv'
    path_to_res_ay = path_to_results[:path_to_results.rfind('_') + 1] + str(num_test) + '_ay.csv'
    path_to_res_grip = path_to_results[:path_to_results.rfind('_') + 1] + str(num_test) + '_grip.csv'
    path_to_data = path_to_data[:path_to_data.rfind('_') + 1] + str(num_test) + '.csv'
    path_to_labels_for_accel = (path_to_labels_for_accel[:path_to_labels_for_accel.rfind('_') + 1]
                                + str(num_test) + '.csv')

    plot_run_test_CRT(filepath2results=path_to_results,
                      filepath2testdata=path_to_data,
                      filepath2resax=path_to_res_ax,
                      filepath2resay=path_to_res_ay,
                      #filepath2resgrip=path_to_res_grip,
                      filepath2labelsaccel=path_to_labels_for_accel,
                      filepath2plots=path_to_plots,
                      counter='perf_' + str(num_test))"""

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
