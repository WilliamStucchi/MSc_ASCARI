import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def plot(data: list,
         titles: list,
         label: str,
         filepath2plots: str,
         counter: str):

    data_ = []
    ax_ = []
    ay_ = []
    for el in data:
        temp = []
        temp1 = []
        temp2 = []
        data_.append(temp)
        ax_.append(temp1)
        ay_.append(temp2)

    # load results
    for i, el in enumerate(data):
        with open(el + str(counter) + '.csv', 'r') as fh:
            data_[i] = np.loadtxt(fh)
        with open(el + str(counter) + '_ax.csv', 'r') as fh:
            ax_[i] = np.loadtxt(fh)
        with open(el + str(counter) + '_ay.csv', 'r') as fh:
            ay_[i] = np.loadtxt(fh)

    # load label data
    with open(label + '.csv', 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    # Extract results for each feature
    res_vx = []
    res_vy = []
    res_yaw = []
    res_ax = []
    res_ay = []
    for el in data:
        temp = []
        res_vx.append(temp)
        res_vy.append(temp)
        res_yaw.append(temp)
        res_ax.append(temp)
        res_ay.append(temp)

    for i, el in enumerate(data_):
        # Yaw rate
        res_yaw[i] = el[:, 0]
        # Vy
        res_vy[i] = el[:, 1]
        # Vx
        res_vx[i] = el[:, 2]

    # Ax
    for i, el in enumerate(ax_):
        res_ax[i] = el[:]

    # Ay
    for i, el in enumerate(ay_):
        res_ay[i] = el[:]

    # Labels
    labels_vx = labels[:, 0]
    labels_vy = labels[:, 1]
    labels_yaw = labels[:, 2]
    labels_ax = labels[:, 3]
    labels_ay = labels[:, 4]

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(res_vx, labels_vx, titles, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.png')
    plot_and_save(res_vy, labels_vy, titles, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.png')
    plot_and_save(res_yaw, labels_yaw, titles, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(res_ax, labels_ax, titles, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.png')
    plot_and_save(res_ay, labels_ay, titles, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.png')

    """
    # Differences
    diff_data1 = labels_all - res_data1
    diff_data2 = labels_all - res_data2
    diff_data3 = labels_all - res_data3
   
    # Scaled results
    results_scaled_data1, labels_scaled_data1 = scale_results(res_data1, labels_all)

    results_scaled_data2, labels_scaled_data2 = scale_results(res_data2, labels_all)

    results_scaled_data3, labels_scaled_data3 = scale_results(res_data3, labels_all)

    # Compute metrics
    row_header = ['MSE', 'RMSE', 'MAE', 'R2']
    column_header = ['long. vel. vx', 'lat. vel. vy', 'yaw rate', 'long. acc. ax', 'lat. acc. ay']

    my_dict = {'MSE': mean_squared_error,
               'RMSE': mean_squared_error,
               'MAE': mean_absolute_error,
               'R2': r2_score
               }

    print('\n')
    print('Test CRT TUM FeedForward')
    compute_metrics_matrix(row_header, column_header, my_dict, results_TUM_ff, labels_all,
                           'Test CRT TUM FeedForward', filepath2plots, counter)

    print('Test CRT TUM FeedForward scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_TUM_ff, labels_scaled_TUM_ff,
                                  'Test CRT TUM FeedForward scaled', filepath2plots, counter)

    print('Test CRT TUM Recurrent')
    compute_metrics_matrix(row_header, column_header, my_dict, results_TUM_rr, labels_all,
                           'Test CRT TUM Recurrent', filepath2plots, counter)

    print('Test CRT TUM Recurrent scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_TUM_rr, labels_scaled_TUM_rr,
                                  'Test CRT TUM Recurrent scaled', filepath2plots, counter)

    print('Test CRT Stanford')
    compute_metrics_matrix(row_header, column_header, my_dict, results_STAN, labels_all,
                           'Test CRT Stanford', filepath2plots, counter)

    print('Test CRT Stanford scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_STAN, labels_scaled_STAN,
                                  'Test CRT Stanford scaled', filepath2plots, counter)"""


    """# Plot and save differences
    plot_and_save(diff_data1[0], diff_data2[0], diff_data3[0], None, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[1], diff_data2[1], diff_data2[1], None, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[2], diff_data2[2], diff_data2[2], None, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[3], diff_data2[3], diff_data2[3], None, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[4], diff_data2[4], diff_data2[4], None, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_diff_' + str(counter) + '.png')"""

# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(res, label, titles, value, savename):
    colors = ['xkcd:red', 'xkcd:green', 'xkcd:purple', 'xkcd:orange', 'xkcd:black']
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    if label is not None:
            plt.plot(label, label='Ground Truth', color='xkcd:blue', alpha=0.5)

    for i, el in enumerate(res):
        plt.plot(res[i], label=titles[i], color=colors[i], alpha=0.75)

    plt.ylabel(value)
    plt.xlabel('Time steps (10 ms)')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
    plt.ion()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def complete_path(path: str, num: int) -> str:
    index_last_underscore = path.rfind('_')
    path = path[:index_last_underscore + 1]
    path += str(num) + '.csv'
    return path


# ----------------------------------------------------------------------------------------------------------------------

def scale_results(results, labels):
    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))

    if len(results) == len(labels) and len(results) == 5:
        scaler_temp_result = np.concatenate((results[0], results[1], results[2], results[3], results[4]), axis=1)
        scaler_temp_label = np.concatenate((labels[0], labels[1], labels[2], labels[3], labels[4]), axis=1)
        scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

        scaler_results = scaler_results.fit(scaler_temp)
        scaler_temp_result = scaler_results.transform(scaler_temp_result)
        scaler_temp_label = scaler_results.transform(scaler_temp_label)

        return scaler_temp_result, scaler_temp_label

    else:
        scaler_temp_result = np.concatenate((results[0], results[1], results[2]), axis=1)
        scaler_temp_label = np.concatenate((labels[0], labels[1], labels[2]), axis=1)
        scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

        scaler_results = scaler_results.fit(scaler_temp)
        scaler_temp_result = scaler_results.transform(scaler_temp_result)
        scaler_temp_label = scaler_results.transform(scaler_temp_label)

        return scaler_temp_result, scaler_temp_label


# ----------------------------------------------------------------------------------------------------------------------

def compute_metric(function, label, prediction):
    return function(label, prediction)


# ----------------------------------------------------------------------------------------------------------------------

def compute_metrics_matrix(rows_head, column_head, dictionary, predictions_head, labels_head, title,
                           path_, counter, round_digits=5):
    data = np.zeros(predictions_head.shape[0] * len(rows_head))

    for i, key in enumerate(dictionary.keys()):
        for idx in range(predictions_head.shape[0]):
            if key == 'RMSE':
                data[i * predictions_head.shape[0] + idx] = np.sqrt(
                    compute_metric(dictionary[key], labels_head[idx], predictions_head[idx]))
            else:
                data[i * predictions_head.shape[0] + idx] = compute_metric(dictionary[key], labels_head[idx],
                                                                           predictions_head[idx])

    data = data.reshape(4, len(column_head)).round(round_digits)

    row_format = "{:>15}" * (len(column_head) + 1)
    print(row_format.format("", *column_head))
    for row_head, row_data in zip(rows_head, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    save_to_csv(data, title, path_, counter)


# ----------------------------------------------------------------------------------------------------------------------


def compute_metrics_matrix_scaled(rows_head, column_head, dictionary, predictions_head, labels_head, title,
                                  path_, counter, round_digits=5):
    data = np.zeros(predictions_head.shape[1] * len(rows_head))

    for i, key in enumerate(dictionary.keys()):
        for idx in range(predictions_head.shape[1]):
            if key == 'RMSE':
                data[i * predictions_head.shape[1] + idx] = np.sqrt(
                    compute_metric(dictionary[key], labels_head[:, idx], predictions_head[:, idx]))
            else:
                data[i * predictions_head.shape[1] + idx] = compute_metric(dictionary[key], labels_head[:, idx],
                                                                           predictions_head[:, idx])

    data = data.reshape(4, len(column_head)).round(round_digits)

    row_format = "{:>15}" * (len(column_head) + 1)
    print(row_format.format("", *column_head))
    for row_head, row_data in zip(rows_head, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    save_to_csv(data, title, path_, counter)


# ----------------------------------------------------------------------------------------------------------------------

def save_to_csv(data, title, path_, counter):
    np.savetxt(path_ + title + ' ' + str(counter) + '.csv', data,
               header='long. vel. vx, lat. vel. vy, yaw rate, long. acc. ax, lat. acc. ay', delimiter=',')


# ----------------------------------------------------------------------------------------------------------------------


result_1 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_07_22/13_34_22/results_test_'
title_1 = 'base'
result_2 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_08_30/12_10_48/results_test_'
title_2 = 'corners+'
result_3 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_07_22/13_34_22/eps_1/results_test_'
title_3 = 'esp1'
result_4 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_07_22/13_34_22/eps_2/results_test_'
title_4 = 'esp2'

path_to_results = [result_1, result_2]
titles = [title_1, title_2]

path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_mu_'

path_to_plots = '../stan combined/'

tests = ['1', '08', '06', '0806', '0804', '06045', '0603']
for test in tests:
    print('Printing results for test ' + test)
    path_to_labels = path_to_labels[:path_to_labels.rfind('_') + 1] + test

    plot(data=path_to_results,
         titles=titles,
         label=path_to_labels,
         filepath2plots=path_to_plots,
         counter='mu_' + test)

"""tot_num_tests = 4
for num_tests in range(1, tot_num_tests):
    print('Printing results for test ' + num_tests)
    path_to_labels = path_to_labels[:path_to_labels.rfind('_') + 1] + str(num_tests)
    plot(data=path_to_results,
         titles=titles,
         label=path_to_labels,
         filepath2plots=path_to_plots,
         counter='perf_' + str(num_tests))"""


# ----------------------------------------------------------------------------------------------------------------------
# Best
# ----------------------------------------------------------------------------------------------------------------------

"""
TUM_path_to_results_ff = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_06/09_52_48/matfiles/prediction_result_feedforward_CRT_'
TUM_path_to_results_rr = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_10/14_26_01/matfiles/prediction_result_recurrent_CRT_'
STAN_path_to_results = 'scirob_submission/Model_Learning/results/step_4/callbacks/2024_05_10/09_39_23/results_test_'
TUM_path_to_data = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/test_set_'
path_to_plots = '../results_combined_4/'
"""