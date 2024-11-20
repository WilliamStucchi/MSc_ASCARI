import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def plot(data: list,
         titles: list,
         label: str,
         filepath2plots: str,
         counter: str,
         save_format: str):

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

    """# plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(res_vx, labels_vx, titles, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.' + save_format, save_format)
    plot_and_save(res_vy, labels_vy, titles, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.' + save_format, save_format)
    plot_and_save(res_yaw, labels_yaw, titles, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.' + save_format, save_format)
    plot_and_save(res_ax, labels_ax, titles, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.' + save_format, save_format)
    plot_and_save(res_ay, labels_ay, titles, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.' + save_format, save_format)"""

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

    """save_to_csv(data, 'MSE AND MAE OF UNSCALED VALUES Test CRT', filepath2plots, counter, column_header)

    save_histogram(res_yaw, labels_yaw, titles, 'Yaw rate',
                   filepath2plots + 'metrics/yaw_metrics_' + str(counter) + '.' + save_format, save_format)
    save_histogram(res_ay, labels_ay, titles, 'Lat. acc. ay',
                   filepath2plots + 'metrics/ay_metrics_' + str(counter) + '.' + save_format, save_format)
    save_histogram(res_ax, labels_ax, titles, 'Long. acc. ax',
                   filepath2plots + 'metrics/ax_metrics_' + str(counter) + '.' + save_format, save_format)
    save_histogram(res_vx, labels_vx, titles, 'Long. vel. vx',
                   filepath2plots + 'metrics/vx_metrics_' + str(counter) + '.' + save_format, save_format)
    save_histogram(res_vy, labels_vy, titles, 'Lat. vel. vy',
                   filepath2plots + 'metrics/vy_metrics_' + str(counter) + '.' + save_format, save_format)
    """
    return data


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(res, label, titles, value, savename, save_format):
    colors = ['r', 'green', 'orange', 'violet', 'xkcd:black']
    plt.figure(figsize=(33, 9))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
    plt.rc('axes', labelsize=35)  # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=25)  # Legenda
    ax = plt.gca()

    if 'yaw' not in savename:
        if 'vx' not in savename:
            ax.yaxis.set_major_locator(MultipleLocator(2.5))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    time_values = np.linspace(0, len(label) / 100, len(label))
    for i, el in enumerate(res):
        plt.plot(time_values, res[i], label=titles[i], color=colors[i], alpha=1.0, linewidth=2.5)

    if label is not None:
            plt.plot(time_values, label, label='Ground Truth', color='b', alpha=1.0, linewidth=2.5)

    plt.ylabel(value)
    plt.xlabel('Time [s]', labelpad=15)
    if 'ax' in savename:
        plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    plt.savefig(savename, format=save_format, dpi=300)
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def save_histogram(nn_res, labels, titles, value, savename, save_format):
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

    plt.savefig(savename, format=save_format, dpi=300)
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

# NN_1 metrics
rmse_NN_1_ax = []
rmse_NN_1_ay = []
rmse_NN_1_vx = []
rmse_NN_1_vy = []
rmse_NN_1_yaw = []

mae_NN_1_ax = []
mae_NN_1_ay = []
mae_NN_1_vx = []
mae_NN_1_vy = []
mae_NN_1_yaw = []

# NN_2_2 metrics
rmse_NN_2_ax = []
rmse_NN_2_ay = []
rmse_NN_2_vx = []
rmse_NN_2_vy = []
rmse_NN_2_yaw = []

mae_NN_2_ax = []
mae_NN_2_ay = []
mae_NN_2_vx = []
mae_NN_2_vy = []
mae_NN_2_yaw = []

color_dict = {'Red_pastel': '#960018',
              'Blue_pastel': '#2A52BE',
              'Green_pastel': '#3CB371',
              'Saffron_pastel': '#FF9933',
              'Purple_pastel': '#9F00FF'}

save_format = 'eps'
result_1 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_20/16_09_55/results_test_'
title_1 = 'FF trained on μ=1.0'
result_2 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_08_30/12_10_48/results_test_'
title_2 = 'FF trained on μ=1.0 and μ=0.6'
result_3 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_14/20_28_05/results_test_'
title_3 = 'Bike (μ=1, μ=0.6) + CRT (μ=1)'
result_4 = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_10_13/16_22_36/results_test_'
title_4 = 'bike + mu1'

path_to_results = [result_1, result_2]
titles = [title_1, title_2]

path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_paramstudy_'

path_to_plots = '../test/test_post_20241020/comparison_between_training_sets/'

for num_test in ['grip_1_perf_50', 'grip_06_perf_100', 'grip_08_perf_100', 'grip_1_perf_100', 'grip_1_perf_75',
                 'grip_06_perf_75', 'grip_06_perf_50', 'grip_08_perf_75', 'grip_08_perf_50']:

    print('Printing results for test ' + str(num_test))
    path_to_labels_ = path_to_labels[:path_to_labels.rfind('_') + 1] + str(num_test)
    metrics = plot(data=path_to_results,
         titles=titles,
         label=path_to_labels_,
         filepath2plots=path_to_plots,
         counter=str(num_test),
         save_format=save_format)

    # NN_1 metrics
    rmse_NN_1_ax.append(metrics[0, 3])
    rmse_NN_1_ay.append(metrics[0, 4])
    rmse_NN_1_vx.append(metrics[0, 1])
    rmse_NN_1_vy.append(metrics[0, 2])
    rmse_NN_1_yaw.append(metrics[0, 0])

    mae_NN_1_ax.append(metrics[1, 3])
    mae_NN_1_ay.append(metrics[1, 4])
    mae_NN_1_vx.append(metrics[1, 1])
    mae_NN_1_vy.append(metrics[1, 2])
    mae_NN_1_yaw.append(metrics[1, 0])

    # NN_2 metrics
    rmse_NN_2_ax.append(metrics[0, 8])
    rmse_NN_2_ay.append(metrics[0, 9])
    rmse_NN_2_vx.append(metrics[0, 6])
    rmse_NN_2_vy.append(metrics[0, 7])
    rmse_NN_2_yaw.append(metrics[0, 5])

    mae_NN_2_ax.append(metrics[1, 8])
    mae_NN_2_ay.append(metrics[1, 9])
    mae_NN_2_vx.append(metrics[1, 6])
    mae_NN_2_vy.append(metrics[1, 7])
    mae_NN_2_yaw.append(metrics[1, 5])
    
# median and iqr
# NN_1
median_rmse_NN_1_ax = np.median(rmse_NN_1_ax)
median_rmse_NN_1_ay = np.median(rmse_NN_1_ay)
median_rmse_NN_1_vx = np.median(rmse_NN_1_vx)
median_rmse_NN_1_vy = np.median(rmse_NN_1_vy)
median_rmse_NN_1_yaw = np.median(rmse_NN_1_yaw)

median_mae_NN_1_ax = np.median(mae_NN_1_ax)
median_mae_NN_1_ay = np.median(mae_NN_1_ay)
median_mae_NN_1_vx = np.median(mae_NN_1_vx)
median_mae_NN_1_vy = np.median(mae_NN_1_vy)
median_mae_NN_1_yaw = np.median(mae_NN_1_yaw)
# NN_2
median_rmse_NN_2_ax = np.median(rmse_NN_2_ax)
median_rmse_NN_2_ay = np.median(rmse_NN_2_ay)
median_rmse_NN_2_vx = np.median(rmse_NN_2_vx)
median_rmse_NN_2_vy = np.median(rmse_NN_2_vy)
median_rmse_NN_2_yaw = np.median(rmse_NN_2_yaw)

median_mae_NN_2_ax = np.median(mae_NN_2_ax)
median_mae_NN_2_ay = np.median(mae_NN_2_ay)
median_mae_NN_2_vx = np.median(mae_NN_2_vx)
median_mae_NN_2_vy = np.median(mae_NN_2_vy)
median_mae_NN_2_yaw = np.median(mae_NN_2_yaw)

# struct NN_1
metrics_NN_1_ax = list({'Median RMSE': median_rmse_NN_1_ax, 'Median MAE': median_mae_NN_1_ax}.values())
metrics_NN_1_ay = list({'Median RMSE': median_rmse_NN_1_ay, 'Median MAE': median_mae_NN_1_ay}.values())
metrics_NN_1_vx = list({'Median RMSE': median_rmse_NN_1_vx, 'Median MAE': median_mae_NN_1_vx}.values())
metrics_NN_1_vy = list({'Median RMSE': median_rmse_NN_1_vy, 'Median MAE': median_mae_NN_1_vy}.values())
metrics_NN_1_yaw = list({'Median RMSE': median_rmse_NN_1_yaw, 'Median MAE': median_mae_NN_1_yaw}.values())

# struct NN_2
metrics_NN_2_ax = list({'Median RMSE': median_rmse_NN_2_ax, 'Median MAE': median_mae_NN_2_ax}.values())
metrics_NN_2_ay = list({'Median RMSE': median_rmse_NN_2_ay, 'Median MAE': median_mae_NN_2_ay}.values())
metrics_NN_2_vx = list({'Median RMSE': median_rmse_NN_2_vx, 'Median MAE': median_mae_NN_2_vx}.values())
metrics_NN_2_vy = list({'Median RMSE': median_rmse_NN_2_vy, 'Median MAE': median_mae_NN_2_vy}.values())
metrics_NN_2_yaw = list({'Median RMSE': median_rmse_NN_2_yaw, 'Median MAE': median_mae_NN_2_yaw}.values())

# PLOT
# ----------------------------------------------------------------------------------------------------------------------
metrics_labels = ['Median RMSE', 'Median MAE']
x = np.arange(len(metrics_labels))  # la posizione delle metriche sull'asse x
width = 0.125  # larghezza delle barre

# AX
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=25)  # Legenda
plt.bar(x - width/2, metrics_NN_1_ax, width, label=titles[0], color=color_dict['Red_pastel'])
plt.bar(x + width/2, metrics_NN_2_ax, width, label=titles[1], color=color_dict['Green_pastel'])

# Aggiunta delle etichette
plt.ylabel('Values')
plt.title('Comparison of the metrics for Long. Acc. $a_x$', pad=15)
plt.xticks([0, 1], metrics_labels)
plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/median'
                           '/median_ax.' + save_format, format=save_format, dpi=300)
plt.close()

# AY
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
plt.bar(x - width/2, metrics_NN_1_ay, width, label=titles[0], color=color_dict['Red_pastel'])
plt.bar(x + width/2, metrics_NN_2_ay, width, label=titles[1], color=color_dict['Green_pastel'])

# Aggiunta delle etichette
plt.ylabel('Values')
plt.title('Comparison of the metrics for Lateral Acc. $a_y$', pad=15)
plt.xticks([0, 1], metrics_labels)
# plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/median'
                            '/median_ay.' + save_format, format=save_format, dpi=300)
plt.close()

# VX
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
plt.bar(x - width/2, metrics_NN_1_vx, width, label=titles[0], color=color_dict['Red_pastel'])
plt.bar(x + width/2, metrics_NN_2_vx, width, label=titles[1], color=color_dict['Green_pastel'])

# Aggiunta delle etichette
plt.ylabel('Values')
plt.title('Comparison of the metrics for Long. Vel. $v_x$', pad=15)
plt.xticks([0, 1], metrics_labels)
# plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/median'
                            '/median_vx.' + save_format, format=save_format, dpi=300)
plt.close()

# VY
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
plt.bar(x - width/2, metrics_NN_1_vy, width, label=titles[0], color=color_dict['Red_pastel'])
plt.bar(x + width/2, metrics_NN_2_vy, width, label=titles[1], color=color_dict['Green_pastel'])

# Aggiunta delle etichette
plt.ylabel('Values')
plt.title('Comparison of the metrics for Lateral Vel. $v_y$', pad=15)
plt.xticks([0, 1], metrics_labels)
# plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/median'
                            '/median_vy.' + save_format, format=save_format, dpi=300)
plt.close()

# YAW
# ----------------------------------------------------------------------------------------------------------------------
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
plt.bar(x - width/2, metrics_NN_1_yaw, width, label=titles[0], color=color_dict['Red_pastel'])
plt.bar(x + width/2, metrics_NN_2_yaw, width, label=titles[1], color=color_dict['Green_pastel'])

# Aggiunta delle etichette
plt.ylabel('Values')
plt.title('Comparison of the metrics for Yaw Rate $r$', pad=15)
plt.xticks([0, 1], metrics_labels)
# plt.legend(loc='best')
plt.grid()
plt.tight_layout()

# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/median'
                            '/median_yaw.' + save_format, format=save_format, dpi=300)
plt.close()

# WHISKERS
# AX
# ----------------------------------------------------------------------------------------------------------------------
rmse_data_ax = [rmse_NN_1_ax, rmse_NN_2_ax]
mae_data_ax = [mae_NN_1_ax, mae_NN_2_ax]

# Group data by metric
data = [rmse_data_ax, mae_data_ax]

# Positioning for each model within RMSE and MAE
positions = [[1, 2], [4, 5]]  # Positions for RMSE and MAE for each model

# Colors for each model
colors = [color_dict['Red_pastel'], color_dict['Green_pastel']]

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=25)  # Legenda
# Create the box plots
for i, metric_data in enumerate(data):
    for j, model_data in enumerate(metric_data):
        plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

# Set the x-axis labels and legend
plt.xticks([1.5, 4.5], ['RMSE', 'MAE'])
plt.title("RMSE and MAE Distribution for Long. Acc. $a_x$", pad=15)

# Add a legend for the models
# Create custom legend handles
legend_handles = [
    mpatches.Patch(color=colors[0], label=titles[0]),
    mpatches.Patch(color=colors[1], label=titles[1])
]
plt.legend(handles=legend_handles, loc="best")
# plt.show()
plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/whiskers'
                            '/whiskers_ax.' + save_format, format=save_format, dpi=300)
plt.close()

# AY
# ----------------------------------------------------------------------------------------------------------------------
rmse_data_ay = [rmse_NN_1_ay, rmse_NN_2_ay]
mae_data_ay = [mae_NN_1_ay, mae_NN_2_ay]

# Group data by metric
data = [rmse_data_ay, mae_data_ay]

# Positioning for each model within RMSE and MAE
positions = [[1, 2], [4, 5]]  # Positions for RMSE and MAE for each model

# Colors for each model
colors = [color_dict['Red_pastel'], color_dict['Green_pastel']]

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
# Create the box plots
for i, metric_data in enumerate(data):
    for j, model_data in enumerate(metric_data):
        plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

# Set the x-axis labels and legend
plt.xticks([1.5, 4.5], ['RMSE', 'MAE'])
plt.title("RMSE and MAE Distribution for Lateral Acc. $a_y$", pad=15)

# Add a legend for the models
# Create custom legend handles
"""legend_handles = [
    mpatches.Patch(color=colors[0], label=titles[0]),
    mpatches.Patch(color=colors[1], label=titles[1]),
    mpatches.Patch(color=colors[2], label=titles[2])
]
plt.legend(handles=legend_handles, loc="upper center")"""
# plt.show()
plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/whiskers'
                            '/whiskers_ay.' + save_format, format=save_format, dpi=300)
plt.close()

# VX
# ----------------------------------------------------------------------------------------------------------------------
rmse_data_vx = [rmse_NN_1_vx, rmse_NN_2_vx]
mae_data_vx = [mae_NN_1_vx, mae_NN_2_vx]

# Group data by metric
data = [rmse_data_vx, mae_data_vx]

# Positioning for each model within RMSE and MAE
positions = [[1, 2], [4, 5]]  # Positions for RMSE and MAE for each model

# Colors for each model
colors = [color_dict['Red_pastel'], color_dict['Green_pastel']]

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
# Create the box plots
for i, metric_data in enumerate(data):
    for j, model_data in enumerate(metric_data):
        plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

# Set the x-axis labels and legend
plt.xticks([1.5, 4.5], ['RMSE', 'MAE'])
plt.title("RMSE and MAE Distribution for Long. Vel. $v_x$", pad=15)

# Add a legend for the models
# Create custom legend handles
"""legend_handles = [
    mpatches.Patch(color=colors[0], label=titles[0]),
    mpatches.Patch(color=colors[1], label=titles[2])
]
plt.legend(handles=legend_handles, loc="upper center")"""
# plt.show()
plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/whiskers'
                            '/whiskers_vx.' + save_format, format=save_format, dpi=300)
plt.close()

# VY
# ----------------------------------------------------------------------------------------------------------------------
rmse_data_vy = [rmse_NN_1_vy, rmse_NN_2_vy]
mae_data_vy = [mae_NN_1_vy, mae_NN_2_vy]

# Group data by metric
data = [rmse_data_vy, mae_data_vy]

# Positioning for each model within RMSE and MAE
positions = [[1, 2], [4, 5]]  # Positions for RMSE and MAE for each model

# Colors for each model
colors = [color_dict['Red_pastel'], color_dict['Green_pastel']]

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
# Create the box plots
for i, metric_data in enumerate(data):
    for j, model_data in enumerate(metric_data):
        plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

# Set the x-axis labels and legend
plt.xticks([1.5, 4.5], ['RMSE', 'MAE'])
plt.title("RMSE and MAE Distribution for Lateral Vel. $v_y$", pad=15)

# Add a legend for the models
# Create custom legend handles
"""legend_handles = [
    mpatches.Patch(color=colors[0], label=titles[0]),
    mpatches.Patch(color=colors[1], label=titles[1]),
    mpatches.Patch(color=colors[2], label=titles[2])
]
plt.legend(handles=legend_handles, loc="upper center")"""
# plt.show()
plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/whiskers'
                            '/whiskers_vy.' + save_format, format=save_format, dpi=300)
plt.close()

# YAW
# ----------------------------------------------------------------------------------------------------------------------
rmse_data_yaw = [rmse_NN_1_yaw, rmse_NN_2_yaw]
mae_data_yaw = [mae_NN_1_yaw, mae_NN_2_yaw]

# Group data by metric
data = [rmse_data_yaw, mae_data_yaw]

# Positioning for each model within RMSE and MAE
positions = [[1, 2], [4, 5]]  # Positions for RMSE and MAE for each model

# Colors for each model
colors = [color_dict['Red_pastel'], color_dict['Green_pastel']]

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda
# Create the box plots
for i, metric_data in enumerate(data):
    for j, model_data in enumerate(metric_data):
        plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

# Set the x-axis labels and legend
plt.xticks([1.5, 4.5], ['RMSE', 'MAE'])
plt.title("RMSE and MAE Distribution for Yaw Rate $r$", pad=15)

# Add a legend for the models
# Create custom legend handles
"""legend_handles = [
    mpatches.Patch(color=colors[0], label=titles[0]),
    mpatches.Patch(color=colors[1], label=titles[1]),
    mpatches.Patch(color=colors[2], label=titles[2])
]
plt.legend(handles=legend_handles, loc="upper center")"""
# plt.show()
plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/whiskers'
                            '/whiskers_yaw.' + save_format, format=save_format, dpi=300)
plt.close()


# Violin plot
# ----------------------------------------------------------------------------------------------------------------------

rmse_data_yaw = [rmse_NN_1_yaw, rmse_NN_2_yaw]
mae_data_yaw = [mae_NN_1_yaw, mae_NN_2_yaw]

# Prepare lists to store the data for the DataFrame
models = []
metrics = []
values = []

## Add RMSE data to the lists
for i, rmse_values in enumerate(rmse_data_yaw):
    model_name = titles[i]
    models.extend([f"{model_name}_RMSE"] * len(rmse_values))
    metrics.extend(['RMSE'] * len(rmse_values))
    values.extend(rmse_values)

# Add MAE data to the lists
for i, mae_values in enumerate(mae_data_yaw):
    model_name = titles[i]
    models.extend([f"{model_name}_MAE"] * len(mae_values))
    metrics.extend(['MAE'] * len(mae_values))
    values.extend(mae_values)

# Create the DataFrame
df = pd.DataFrame({
    'Model_Metric': models,
    'Metric': metrics,
    'Value': values
})

print(df)

# Define a custom color palette for each model, with consistent colors for both RMSE and MAE
palette = {
    titles[0] + '_RMSE': color_dict['Red_pastel'],
    titles[1] + '_RMSE': color_dict['Saffron_pastel'],
    titles[0] + '_MAE': color_dict['Red_pastel'],
    titles[1] + '_MAE': color_dict['Saffron_pastel']
}

plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=30)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda

# Create a violin plot with the custom palette and separated categories for each model and metric
sns.violinplot(x='Model_Metric', y='Value', data=df, hue='Model_Metric', palette=palette)

# Add title and labels
plt.title('RMSE and MAE Distributions for Models')
plt.xlabel('Metric')
plt.ylabel('Value')

plt.tight_layout()
# Mostrare il grafico
plt.savefig(path_to_plots + '/metrics/violin'
                            '/violin_yaw.' + save_format, format=save_format, dpi=300)
plt.close()


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