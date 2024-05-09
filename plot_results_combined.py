import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def plot(TUM_filepath2results_ff: str,
         TUM_filepath2results_rr: str,
         STAN_filepath2results: str,
         filepath2testdata: str,
         filepath2plots: str,
         counter: int):
    # load results
    with open(TUM_filepath2results_ff, 'r') as fh:
        TUM_data_ff = np.loadtxt(fh)

    with open(TUM_filepath2results_rr, 'r') as fh:
        TUM_data_rr = np.loadtxt(fh)

    with open(STAN_filepath2results, 'r') as fh:
        STAN_data = np.loadtxt(fh)

    # load label data
    with open(filepath2testdata, 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    results_TUM_ff = np.zeros((5, len(TUM_data_ff), 1))
    results_TUM_rr = np.zeros((5, len(TUM_data_rr), 1))
    results_STAN = np.zeros((3, len(STAN_data), 1))
    labels_all = np.zeros((5, len(labels), 1))

    # Vx
    results_TUM_ff[0] = TUM_data_ff[:, 0][:, np.newaxis]
    results_TUM_rr[0] = TUM_data_rr[:, 0][:, np.newaxis]
    results_STAN[0] = STAN_data[:, 2][:, np.newaxis]

    # Vy
    results_TUM_ff[1] = TUM_data_ff[:, 1][:, np.newaxis]
    results_TUM_rr[1] = TUM_data_rr[:, 1][:, np.newaxis]
    results_STAN[1] = STAN_data[:, 1][:, np.newaxis]

    # Yaw rate
    results_TUM_ff[2] = TUM_data_ff[:, 2][:, np.newaxis]
    results_TUM_rr[2] = TUM_data_rr[:, 2][:, np.newaxis]
    results_STAN[2] = STAN_data[:, 0][:, np.newaxis]

    # Ax
    results_TUM_ff[3] = TUM_data_ff[:, 3][:, np.newaxis]
    results_TUM_rr[3] = TUM_data_rr[:, 3][:, np.newaxis]

    # Ay
    results_TUM_ff[4] = TUM_data_ff[:, 4][:, np.newaxis]
    results_TUM_rr[4] = TUM_data_rr[:, 4][:, np.newaxis]

    labels_all[0] = labels[:, 0][:, np.newaxis]
    labels_all[1] = labels[:, 1][:, np.newaxis]
    labels_all[2] = labels[:, 2][:, np.newaxis]
    labels_all[3] = labels[:, 3][:, np.newaxis]
    labels_all[4] = labels[:, 4][:, np.newaxis]

    results_scaled_TUM_ff, labels_scaled_TUM_ff = scale_results(results_TUM_ff, labels_all)

    results_scaled_TUM_rr, labels_scaled_TUM_rr = scale_results(results_TUM_rr, labels_all)

    results_scaled_STAN, labels_scaled_STAN = scale_results(results_STAN, labels_all[:3])

    # print deviation from label

    round_digits = 5
    row_header = ['MSE', 'RMSE', 'MAE', 'R2']
    column_header = ['long. vel. vx', 'lat. vel. vy', 'yaw rate', 'long. acc. ax', 'lat. acc. ay']

    my_dict = {'MSE': mean_squared_error,
               'RMSE': mean_squared_error,
               'MAE': mean_absolute_error,
               'R2': r2_score
               }

    print('\n')
    print('Test CRT TUM FeedForward')
    compute_metrics_matrix(row_header, column_header, my_dict, results_TUM_ff, labels_all, 'Test CRT TUM FeedForward')

    print('Test CRT TUM FeedForward scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_TUM_ff, labels_scaled_TUM_ff,
                                  'Test CRT TUM FeedForward scaled')

    print('Test CRT TUM Recurrent')
    compute_metrics_matrix(row_header, column_header, my_dict, results_TUM_rr, labels_all, 'Test CRT TUM Recurrent')

    print('Test CRT TUM Recurrent scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_TUM_rr, labels_scaled_TUM_rr,
                                  'Test CRT TUM Recurrent scaled')

    column_header = ['long. vel. vx', 'lat. vel. vy', 'yaw rate']
    print('Test CRT Stanford')
    compute_metrics_matrix(row_header, column_header, my_dict, results_STAN, labels_all, 'Test CRT Stanford')

    print('Test CRT Stanford scaled')
    compute_metrics_matrix_scaled(row_header, column_header, my_dict, results_scaled_STAN, labels_scaled_STAN,
                                  'Test CRT Stanford scaled')

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(results_TUM_ff[0], results_TUM_rr[0], results_STAN[0], labels_all[0], 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '_.png')
    plot_and_save(results_TUM_ff[1], results_TUM_rr[1], results_STAN[1], labels_all[1], 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '_.png')
    plot_and_save(results_TUM_ff[2], results_TUM_rr[2], results_STAN[2], labels_all[2], 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(results_TUM_ff[3], results_TUM_rr[3], [], labels_all[3], 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '_.png')
    plot_and_save(results_TUM_ff[4], results_TUM_rr[4], [], labels_all[4], 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '_.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(TUM_ff, TUM_rr, STAN, label, value, savename):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    plt.plot(TUM_ff, label='TUM feedforward', color='tab:orange')
    plt.plot(TUM_rr, label='TUM recurrent', color='tab:red')
    if len(STAN) != 0:
        plt.plot(STAN, label='Stanford', color='tab:green')

    plt.plot(label, label='Ground Truth', color='tab:blue')

    plt.ylabel(value)
    plt.xlabel('Time steps (8 ms)')
    plt.legend()
    plt.grid()

    index_last_underscore = savename.rfind('_')
    savename = savename[:index_last_underscore + 1]
    plt.savefig(savename + 'full.png', format='png', dpi=300)
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

def compute_metrics_matrix(rows_head, column_head, dictionary, predictions_head, labels_head, title, round_digits=5):
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

    save_to_csv(data, title, column_head)


# ----------------------------------------------------------------------------------------------------------------------


def compute_metrics_matrix_scaled(rows_head, column_head, dictionary, predictions_head, labels_head, title,
                                  round_digits=5):
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

    save_to_csv(data, title, column_head)


# ----------------------------------------------------------------------------------------------------------------------

def save_to_csv(data, title, columns_head):
    np.savetxt('../results_combined/ ' + title + '.csv', data,
               header='long. vel. vx, lat. vel. vy, yaw rate, long. acc. ax, lat. acc. ay', delimiter=',')


# ----------------------------------------------------------------------------------------------------------------------


TUM_path_to_results_ff = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_06/09_52_48/matfiles/prediction_result_feedforward_CRT_'
TUM_path_to_results_rr = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_09/08_49_58/matfiles/prediction_result_recurrent_CRT_'
STAN_path_to_results = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_05_02/09_59_22/results_test_'
TUM_path_to_data = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/test_set_'
path_to_plots = '../results_combined/'

for num_tests in range(1, 3):
    TUM_path_to_results_ff = complete_path(TUM_path_to_results_ff, num_tests)
    TUM_path_to_results_rr = complete_path(TUM_path_to_results_rr, num_tests)
    STAN_path_to_results = complete_path(STAN_path_to_results, num_tests)

    TUM_path_to_data = complete_path(TUM_path_to_data, num_tests)

    plot(TUM_filepath2results_ff=TUM_path_to_results_ff,
         TUM_filepath2results_rr=TUM_path_to_results_rr,
         STAN_filepath2results=STAN_path_to_results,
         filepath2testdata=TUM_path_to_data,
         filepath2plots=path_to_plots,
         counter=num_tests)
