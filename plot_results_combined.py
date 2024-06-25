import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def plot(data1: str,
         data2: str,
         data3: str,
         test_data: str,
         filepath2plots: str,
         counter: int):
    # load results
    with open(data1 + str(counter) + '.csv', 'r') as fh:
        data1_ = np.loadtxt(fh)
    with open(data1 + str(counter) + '_ax.csv', 'r') as fh:
        data1_ax = np.loadtxt(fh)
    with open(data1 + str(counter) + '_ay.csv', 'r') as fh:
        data1_ay = np.loadtxt(fh)

    with open(data2 + str(counter) + '.csv', 'r') as fh:
        data2_ = np.loadtxt(fh)
    with open(data2 + str(counter) + '_ax.csv', 'r') as fh:
        data2_ax = np.loadtxt(fh)
    with open(data2 + str(counter) + '_ay.csv', 'r') as fh:
        data2_ay = np.loadtxt(fh)

    with open(data3 + str(counter) + '.csv', 'r') as fh:
        data3_ = np.loadtxt(fh)
    with open(data3 + str(counter) + '_ax.csv', 'r') as fh:
        data3_ax = np.loadtxt(fh)
    with open(data3 + str(counter) + '_ay.csv', 'r') as fh:
        data3_ay = np.loadtxt(fh)

    # load label data
    with open(test_data + str(counter) + '.csv', 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    res_data1 = np.zeros((5, len(data1_), 1))
    res_data2 = np.zeros((5, len(data2_), 1))
    res_data3 = np.zeros((5, len(data3_), 1))
    labels_all = np.zeros((5, len(labels), 1))

    # Vx
    res_data1[0] = data1_[:, 0][:, np.newaxis]
    res_data2[0] = data2_[:, 0][:, np.newaxis]
    res_data3[0] = data3_[:, 2][:, np.newaxis]

    # Vy
    res_data1[1] = data1_[:, 1][:, np.newaxis]
    res_data2[1] = data2_[:, 1][:, np.newaxis]
    res_data3[1] = data3_[:, 1][:, np.newaxis]

    # Yaw rate
    res_data1[2] = data1_[:, 2][:, np.newaxis]
    res_data2[2] = data2_[:, 2][:, np.newaxis]
    res_data3[2] = data3_[:, 0][:, np.newaxis]

    # Ax
    res_data1[3] = data1_ax[:][:, np.newaxis]
    res_data2[3] = data2_ax[:][:, np.newaxis]
    res_data3[3] = data3_ax[:][:, np.newaxis]

    # Ay
    res_data1[4] = data1_ay[:][:, np.newaxis]
    res_data2[4] = data2_ay[:][:, np.newaxis]
    res_data3[4] = data3_ay[:][:, np.newaxis]

    # Labels
    labels_all[0] = labels[:, 0][:, np.newaxis]
    labels_all[1] = labels[:, 1][:, np.newaxis]
    labels_all[2] = labels[:, 2][:, np.newaxis]
    labels_all[3] = labels[:, 3][:, np.newaxis]
    labels_all[4] = labels[:, 4][:, np.newaxis]

    # Differences
    diff_data1 = labels_all - res_data1
    diff_data2 = labels_all - res_data2
    diff_data3 = labels_all - res_data3
    """
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

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(res_data1[0], res_data2[0], res_data3[0], labels_all[0], 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.png')
    plot_and_save(res_data1[1], res_data2[1], res_data3[1], labels_all[1], 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.png')
    plot_and_save(res_data1[2], res_data2[2], res_data3[2], labels_all[2], 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(res_data1[3], res_data1[3], res_data3[3], labels_all[3], 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.png')
    plot_and_save(res_data1[4], res_data2[4], res_data3[4], labels_all[4], 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.png')

    # Plot and save differences
    plot_and_save(diff_data1[0], diff_data2[0], diff_data3[0], None, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[1], diff_data2[1], diff_data2[1], None, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[2], diff_data2[2], diff_data2[2], None, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[3], diff_data2[3], diff_data2[3], None, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_diff_' + str(counter) + '.png')
    plot_and_save(diff_data1[4], diff_data2[4], diff_data2[4], None, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_diff_' + str(counter) + '.png')

# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(data1, data2, data3, label, value, savename):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    if data1 is not None:
        plt.plot(data1, label='TUM feedforward', color='orange')

    if data2 is not None:
        plt.plot(data2, label='TUM recurrent', color='red')

    if data3 is not None:
        plt.plot(data3, label='Stanford', color='green')

    if label is not None:
        plt.plot(label, label='Ground Truth', color='blue')

    plt.ylabel(value)
    plt.xlabel('Time steps (10 ms)')
    plt.legend()
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


training_cplt = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_05_17/12_29_37/results_test_'
training_no_low = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_06_03/10_49_13/results_test_'
training_no_low_no_mid = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_06_04/15_40_10/results_test_'
TUM_path_to_data = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_'
path_to_plots = '../test/'

for num_tests in range(0, 4):
    plot(data1=training_cplt,
         data2=training_no_low,
         data3=training_no_low_no_mid,
         test_data=TUM_path_to_data,
         filepath2plots=path_to_plots,
         counter=num_tests)


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