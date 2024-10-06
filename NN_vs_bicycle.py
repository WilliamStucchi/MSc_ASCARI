import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------

def plot(NN_res: str,
         NN_res_ay: str,
         NN_res_ax: str,
         bicycle_res: str,
         bicycle_vx_computed_res: str,
         titles: list,
         labels: str,
         filepath2plots: str,
         counter: str):
    # load results
    with open(NN_res, 'r') as fh:
        nn_res_ = np.loadtxt(fh)
        nn_res_ = nn_res_[:-4]
    with open(NN_res_ay, 'r') as fh:
        nn_res_ay_ = np.loadtxt(fh)
        nn_res_ay_ = nn_res_ay_[:-4]
    with open(NN_res_ax, 'r') as fh:
        nn_res_ax_ = np.loadtxt(fh)
        nn_res_ax_ = nn_res_ax_[:-4]
    with open(bicycle_res, 'r') as fh:
        data = pd.read_csv(bicycle_res, dtype=object)

        bicycle_res_ = np.array(data, dtype=float)
        bicycle_res_[np.isnan(bicycle_res_)] = 0
        bicycle_res_ = bicycle_res_[:-4]
    with open(bicycle_vx_computed_res, 'r') as fh:
        data = pd.read_csv(bicycle_vx_computed_res, dtype=object)

        bicycle_vx_res_ = np.array(data, dtype=float)
        bicycle_vx_res_[np.isnan(bicycle_vx_res_)] = 0
        bicycle_vx_res_ = bicycle_vx_res_[:-4]

    # load label data
    with open(labels, 'r') as fh:
        labels_ = np.loadtxt(fh, delimiter=',')
        labels_ = labels_[:-4]

    # Extract results for each feature
    # NN model
    nn_res_yaw = nn_res_[:, 0][:, np.newaxis]
    nn_res_vy = nn_res_[:, 1][:, np.newaxis]
    nn_res_vx = nn_res_[:, 2][:, np.newaxis]
    nn_res_ay = nn_res_ay_[:, np.newaxis]
    nn_res_ax = nn_res_ax_[:, np.newaxis]

    # Bicycle model
    bicycle_res_ax = bicycle_res_[:, 1][:, np.newaxis]
    bicycle_res_ay = bicycle_res_[:, 2][:, np.newaxis]
    bicycle_res_vy = bicycle_res_[:, 3][:, np.newaxis]
    bicycle_res_yaw = bicycle_res_[:, 4][:, np.newaxis]
    bicycle_res_vx = bicycle_res_[:, 5][:, np.newaxis]

    # Bicycle model with vx computed from Fx
    bicycle_vx_res_ax = bicycle_vx_res_[:, 1][:, np.newaxis]
    bicycle_vx_res_ay = bicycle_vx_res_[:, 2][:, np.newaxis]
    bicycle_vx_res_vy = bicycle_vx_res_[:, 3][:, np.newaxis]
    bicycle_vx_res_yaw = bicycle_vx_res_[:, 4][:, np.newaxis]
    bicycle_vx_res_vx = bicycle_vx_res_[:, 5][:, np.newaxis]

    # Labels
    labels_vx = labels_[:, 0]
    labels_vy = labels_[:, 1]
    labels_yaw = labels_[:, 2]
    labels_ax = labels_[:, 3]
    labels_ay = labels_[:, 4]

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(nn_res_yaw, bicycle_res_yaw, bicycle_vx_res_yaw, labels_yaw, titles, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(nn_res_ay, bicycle_res_ay, bicycle_vx_res_ay, labels_ay, titles, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.png')
    plot_and_save(nn_res_ax, bicycle_res_ax, bicycle_vx_res_ax, labels_ax, titles, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.png')
    plot_and_save(nn_res_vx, bicycle_res_vx, bicycle_vx_res_vx, labels_vx, titles, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.png')
    plot_and_save(nn_res_vy, bicycle_res_vy, bicycle_vx_res_vy, labels_vy, titles, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.png')

    # print('\n MSE AND MAE OF UNSCALED VALUES: Test CRT Neural Network ' + str(counter))

    data = np.asarray([mean_squared_error(labels_yaw, nn_res_yaw),
                       mean_squared_error(labels_vx, nn_res_vx),
                       mean_squared_error(labels_vy, nn_res_vy),
                       mean_squared_error(labels_ax, nn_res_ax),
                       mean_squared_error(labels_ay, nn_res_ay),

                       mean_squared_error(labels_yaw, bicycle_res_yaw),
                       mean_squared_error(labels_vx, bicycle_res_vx),
                       mean_squared_error(labels_vy, bicycle_res_vy),
                       mean_squared_error(labels_ax, bicycle_res_ax),
                       mean_squared_error(labels_ay, bicycle_res_ay),

                       mean_squared_error(labels_yaw, bicycle_vx_res_yaw),
                       mean_squared_error(labels_vx, bicycle_vx_res_vx),
                       mean_squared_error(labels_vy, bicycle_vx_res_vy),
                       mean_squared_error(labels_ax, bicycle_vx_res_ax),
                       mean_squared_error(labels_ay, bicycle_vx_res_ay),

                       mean_absolute_error(labels_yaw, nn_res_yaw),
                       mean_absolute_error(labels_vx, nn_res_vx),
                       mean_absolute_error(labels_vy, nn_res_vy),
                       mean_absolute_error(labels_ax, nn_res_ax),
                       mean_absolute_error(labels_ay, nn_res_ay),

                       mean_absolute_error(labels_yaw, bicycle_res_yaw),
                       mean_absolute_error(labels_vx, bicycle_res_vx),
                       mean_absolute_error(labels_vy, bicycle_res_vy),
                       mean_absolute_error(labels_ax, bicycle_res_ax),
                       mean_absolute_error(labels_ay, bicycle_res_ay),

                       mean_absolute_error(labels_yaw, bicycle_vx_res_yaw),
                       mean_absolute_error(labels_vx, bicycle_vx_res_vx),
                       mean_absolute_error(labels_vy, bicycle_vx_res_vy),
                       mean_absolute_error(labels_ax, bicycle_vx_res_ax),
                       mean_absolute_error(labels_ay, bicycle_vx_res_ay)

                       ]).reshape(2, 15).round(5)

    column_header = ['NN yaw rate', 'NN long. vel. vx', 'NN lat. vel. vy', 'NN long. acc. ax', 'NN lat. acc. ay',
                     'Bi yaw rate', 'Bi long. vel. vx', 'Bi lat. vel. vy', 'Bi long. acc. ax', 'Bi lat. acc. ay',
                     'BiFx yaw rate', 'BiFx long. vel. vx', 'BiFx lat. vel. vy', 'BiFx long. acc. ax', 'BiFx lat. acc. ay']
    row_header = ['MSE', 'MAE']

    """row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))"""

    save_to_csv(data, 'MSE AND MAE OF UNSCALED VALUES Test CRT', filepath2plots, counter)

    """print('MSE AND MAE OF UNSCALED VALUES: Test CRT Bicyle Model ' + str(counter))

    data = np.asarray([mean_squared_error(labels_yaw, bicycle_res_yaw),
                       mean_squared_error(labels_vx, bicycle_res_vx),
                       mean_squared_error(labels_vy, bicycle_res_vy),
                       mean_squared_error(labels_ax, bicycle_res_ax),
                       mean_squared_error(labels_ay, bicycle_res_ay),
                       mean_absolute_error(labels_yaw, bicycle_res_yaw),
                       mean_absolute_error(labels_vx, bicycle_res_vx),
                       mean_absolute_error(labels_vy, bicycle_res_vy),
                       mean_absolute_error(labels_ax, bicycle_res_ax),
                       mean_absolute_error(labels_ay, bicycle_res_ay)]).reshape(2, 5).round(5)

    column_header = ['yaw rate', 'long. vel. vx', 'lat. vel. vy', 'long. acc. ax', 'lat. acc. ay']
    row_header = ['MSE', 'MAE']

    row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    save_to_csv(data, 'MSE AND MAE OF UNSCALED VALUES Test CRT Bicycle Model', filepath2plots, counter)"""


# ----------------------------------------------------------------------------------------------------------------------

def save_to_csv(data, title, path_, counter):
    np.savetxt(path_ + title + ' ' + str(counter) + '.csv', data,
               header='NN yaw rate, NN long. vel. vx, NN lat. vel. vy, NN long. acc. ax, NN lat. acc. ay, '
                      'Bi yaw rate, Bi long. vel. vx, Bi lat. vel. vy, Bi long. acc. ax, Bi lat. acc. ay, '
                      'BiFx yaw rate, BiFx long. vel. vx, BiFx lat. vel. vy, BiFx long. acc. ax, BiFx lat. acc. ay', delimiter=',')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(nn_res, bicycle_res, bicycle_vx_res, label, titles, value, savename):
    colors = ['xkcd:red', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:black']
    plt.figure(figsize=(25, 10))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=25)   # Titolo degli assi
    plt.rc('axes', labelsize=25)   # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    time_values = np.linspace(0, len(nn_res) / 100, len(nn_res))
    plt.plot(time_values, nn_res, label=titles[0], color=colors[0], alpha=1.0, linewidth=1.5)
    plt.plot(time_values, bicycle_res, label=titles[1], color=colors[1], alpha=1.0, linewidth=1.5)
    plt.plot(time_values, bicycle_vx_res, label=titles[2], color=colors[2], alpha=1.0, linewidth=1.5)

    if label is not None:
        plt.plot(time_values, label, label='Ground Truth', color='xkcd:blue', alpha=1.0, linewidth=1.5)

    plt.ylabel(value)
    plt.xlabel('Time steps [s]')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
    plt.ion()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

directories = ['/grip_1_perf_100/', '/grip_1_perf_75/', '/grip_1_perf_50/',
               '/grip_08_perf_100/', '/grip_08_perf_75/', '/grip_08_perf_50/',
               '/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/']

path_to_NN_res = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_08_30/12_10_48/results_test_'
path_to_bicycle_res = '../matlab/paths/sim_test_'
path_to_bicycle_vx_computed_res = '../matlab/paths_with_vx_computed/sim_test_'
path_to_plots = '../test/NN_vs_bicycle/paths_combined/'
path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_paramstudy_'
titles = ['Neural Network model', 'Bicycle model', 'Bicycle model with Fx as input']

for num_test, dir_ in tqdm(enumerate(directories)):
    path_to_NN_res = path_to_NN_res[:path_to_NN_res.rfind('results_test_') + 13] + dir_.replace('/', '') + '.csv'
    path_to_NN_res_ay = path_to_NN_res[:path_to_NN_res.rfind('.')] + '_ay.csv'
    path_to_NN_res_ax = path_to_NN_res[:path_to_NN_res.rfind('.')] + '_ax.csv'
    path_to_labels = (path_to_labels[:path_to_labels.rfind('paramstudy_') + 11] + dir_.replace('/', '') + '.csv')

    path_to_bicycle_res = path_to_bicycle_res[:path_to_bicycle_res.rfind('sim_test_') + 9] + dir_.replace('/',
                                                                                                          '') + '.csv'
    path_to_bicycle_vx_computed_res = path_to_bicycle_vx_computed_res[:path_to_bicycle_vx_computed_res.rfind('sim_test_') + 9] + dir_.replace('/',
                                                                                                          '') + '.csv'

    plot(NN_res=path_to_NN_res,
         NN_res_ay=path_to_NN_res_ay,
         NN_res_ax=path_to_NN_res_ax,
         bicycle_res=path_to_bicycle_res,
         bicycle_vx_computed_res=path_to_bicycle_vx_computed_res,
         titles=titles,
         filepath2plots=path_to_plots,
         labels=path_to_labels,
         counter=dir_.replace('/', ''))
