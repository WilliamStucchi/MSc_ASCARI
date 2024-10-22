import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    data_ = [arr[:-4] for arr in data_]
    ax_ = [arr[:-4] for arr in ax_]
    ay_ = [arr[:-4] for arr in ay_]

    # load label data
    with open(label + '.csv', 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')
        labels = labels[:-4]

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

    metrics_array = []
    column_header = ', '
    for i in range(len(res_yaw)):
        metrics_array.append(mean_squared_error(labels_yaw, res_yaw[i]))
        metrics_array.append(mean_squared_error(labels_vx, res_vx[i]))
        metrics_array.append(mean_squared_error(labels_vy, res_vy[i]))
        metrics_array.append(mean_squared_error(labels_ax, res_ax[i]))
        metrics_array.append(mean_squared_error(labels_ay, res_ay[i]))
        column_header = column_header + titles[i] + ' yaw_rate, '
        column_header = column_header + titles[i] + ' long. vel. vx, '
        column_header = column_header + titles[i] + ' lat. vel. vy, '
        column_header = column_header + titles[i] + ' long. acc. ax, '
        column_header = column_header + titles[i] + ' lat. acc. ay, '

    for i in range(len(res_yaw)):
        metrics_array.append(mean_absolute_error(labels_yaw, res_yaw[i]))
        metrics_array.append(mean_absolute_error(labels_vx, res_vx[i]))
        metrics_array.append(mean_absolute_error(labels_vy, res_vy[i]))
        metrics_array.append(mean_absolute_error(labels_ax, res_ax[i]))
        metrics_array.append(mean_absolute_error(labels_ay, res_ay[i]))
        
    data = np.asarray([metrics_array]).reshape(2, 5*len(res_yaw)).round(5)

    save_to_csv(data, 'MSE AND MAE OF UNSCALED VALUES Test CRT', filepath2plots, counter, column_header)

    save_histogram(res_yaw, labels_yaw, titles, 'Yaw rate', filepath2plots + 'metrics/yaw_metrics_' + str(counter) + '.png')
    save_histogram(res_ay, labels_ay, titles, 'Lat. acc. ay', filepath2plots + 'metrics/ay_metrics_' + str(counter) + '.png')
    save_histogram(res_ax, labels_ax, titles, 'Long. acc. ax', filepath2plots + 'metrics/ax_metrics_' + str(counter) + '.png')
    save_histogram(res_vx, labels_vx, titles, 'Long. vel. vx', filepath2plots + 'metrics/vx_metrics_' + str(counter) + '.png')
    save_histogram(res_vy, labels_vy, titles, 'Lat. vel. vy', filepath2plots + 'metrics/vy_metrics_' + str(counter) + '.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(res, label, titles, value, savename):
    colors = ['r', 'green', 'orange', 'violet', 'xkcd:black']
    plt.figure(figsize=(25, 10))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    time_values = np.linspace(0, len(label) / 100, len(label))
    for i, el in enumerate(res):
        plt.plot(time_values, res[i], label=titles[i], color=colors[i], alpha=1.0, linewidth=2.0)

    if label is not None:
            plt.plot(time_values, label, label='Ground Truth', color='b', alpha=1.0, linewidth=2.0)

    plt.ylabel(value)
    plt.xlabel('Time [s]')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
    plt.ion()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def save_histogram(nn_res, labels, titles, value, savename):
    metrics_labels = ['RMSE', 'MAE']

    rmse = []
    mae = []
    for i in range(len(nn_res)):
        rmse.append(np.sqrt(mean_squared_error(labels, nn_res[i])))
        mae.append(mean_absolute_error(labels, nn_res[i]))

    metrics_values = []
    for i in range(len(nn_res)):
        metrics_values.append(list({'RMSE': rmse[i], 'MAE': mae[i]}.values()))

    # Plotting
    plt.figure(figsize=(18, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=20)  # Titolo degli assi
    plt.rc('axes', labelsize=20)  # Etichette degli assi
    plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=12)  # Legenda

    colors = ['xkcd:red', 'xkcd:green', 'xkcd:purple', 'xkcd:orange', 'xkcd:black']
    x = np.arange(len(metrics_labels))  # X locations for features
    width = [-0.125-0.0625, -0.0625, 0.0625, 0.0625+0.125]

    for i in range(len(nn_res)):
        plt.bar(x + width[i], metrics_values[i], 0.125, label=titles[i], color=colors[i])

    """plt.bar(x, metrics_bicycle_values, width, label='Bicycle model', color='green')
    plt.bar(x + width, metrics_bicycle_vx_comp_values, width, label='Bicycle model with Fx as input',
                     color='orange')"""

    # Aggiunta delle etichette
    plt.ylabel('Values')
    plt.title('Comparison of the metrics for the ' + value)
    plt.xticks([0, 1], metrics_labels)
    plt.legend(loc='best')

    # Mostrare il grafico
    plt.tight_layout()
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
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

def save_to_csv(data, title, path_, counter, column_header):
    new_column = np.array([['MSE'], ['MAE']])
    data = np.hstack((new_column, data))
    np.savetxt(path_ + title + ' ' + str(counter) + '.csv', data, header=column_header, delimiter=',',
               fmt='%s', comments='')


# ----------------------------------------------------------------------------------------------------------------------


result_1 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_14/19_30_02/results_latest_'
title_1 = 'CRT (μ=1)'
result_2 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_17/16_05_26/results_latest_'
title_2 = 'Bike (μ=1) + CRT (μ=1)'
result_3 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_14/20_28_05/results_latest_'
title_3 = 'Bike (μ=1, μ=0.6) + CRT (μ=1)'
result_4 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_13/16_22_36/results_latest_'
title_4 = 'bike + mu1'

path_to_results = [result_1, result_2, result_3]
titles = [title_1, title_2, title_3]

path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/latest/test_set_'

path_to_plots = '../test/test_post_20241020/'

for num_test in ['mu_1_perf_50', 'mu_06_perf_100', 'mu_08_perf_100', 'mu_1_perf_100', 'mu_1_perf_75', 'mu_1_perf_25', 'mu_06_perf_75', 'mu_06_perf_50',
                 'mu_08_perf_75', 'mu_08_perf_50']:

    print('Printing results for test ' + str(num_test))
    path_to_labels_ = path_to_labels[:path_to_labels.rfind('_') + 1] + str(num_test)
    plot(data=path_to_results,
         titles=titles,
         label=path_to_labels_,
         filepath2plots=path_to_plots,
         counter=str(num_test))


"""tests = ['1', '08', '06', '0806', '0804', '06045', '0603']
for test in tests:
    print('Printing results for test mu' + test)
    path_to_labels = path_to_labels[:path_to_labels.rfind('_') + 1] + test

    plot(data=path_to_results,
         titles=titles,
         label=path_to_labels,
         filepath2plots=path_to_plots,
         counter='mu_' + test)


path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_'

tot_num_tests = 4
for num_tests in range(1, tot_num_tests):
    print('Printing results for test perf' + str(num_tests))
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