import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ----------------------------------------------------------------------------------------------------------------------

def plot(bicycle_res: str,
         bicycle_vx_computed_res: str,
         titles: list,
         labels: str,
         filepath2plots: str,
         counter: str):
    # load results
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
    plot_and_save(bicycle_res_yaw, bicycle_vx_res_yaw, labels_yaw, titles, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.' + save_format)
    plot_and_save(bicycle_res_ay, bicycle_vx_res_ay, labels_ay, titles, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '.' + save_format)
    plot_and_save(bicycle_res_ax, bicycle_vx_res_ax, labels_ax, titles, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '.' + save_format)
    plot_and_save(bicycle_res_vx, bicycle_vx_res_vx, labels_vx, titles, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.' + save_format)
    plot_and_save(bicycle_res_vy, bicycle_vx_res_vy, labels_vy, titles, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.' + save_format)

    print('\n MSE AND MAE OF UNSCALED VALUES: Test CRT Bicycle model ' + str(counter))

    data = np.asarray([root_mean_squared_error(labels_yaw, bicycle_res_yaw),
                       root_mean_squared_error(labels_vx, bicycle_res_vx),
                       root_mean_squared_error(labels_vy, bicycle_res_vy),
                       root_mean_squared_error(labels_ax, bicycle_res_ax),
                       root_mean_squared_error(labels_ay, bicycle_res_ay),

                       root_mean_squared_error(labels_yaw, bicycle_vx_res_yaw),
                       root_mean_squared_error(labels_vx, bicycle_vx_res_vx),
                       root_mean_squared_error(labels_vy, bicycle_vx_res_vy),
                       root_mean_squared_error(labels_ax, bicycle_vx_res_ax),
                       root_mean_squared_error(labels_ay, bicycle_vx_res_ay),

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

                       ]).reshape(2, 10).round(5)

    save_to_csv(data, 'RMSE AND MAE OF UNSCALED VALUES Test CRT', filepath2plots, counter)

    save_histogram(root_mean_squared_error(labels_yaw, bicycle_res_yaw), root_mean_squared_error(labels_yaw, bicycle_vx_res_yaw),
                   mean_absolute_error(labels_yaw, bicycle_res_yaw), mean_absolute_error(labels_yaw, bicycle_vx_res_yaw),
                   titles, 'Yaw rate', filepath2plots + 'metrics/yaw_metrics_' + str(counter) + '.' + save_format)
    save_histogram(root_mean_squared_error(labels_ay, bicycle_res_ay), root_mean_squared_error(labels_ay, bicycle_vx_res_ay),
                   mean_absolute_error(labels_ay, bicycle_res_ay), mean_absolute_error(labels_ay, bicycle_vx_res_ay),
                   titles, 'Lat. acc. ay', filepath2plots + 'metrics/ay_metrics_' + str(counter) + '.' + save_format)
    save_histogram(root_mean_squared_error(labels_ax, bicycle_res_ax), root_mean_squared_error(labels_ax, bicycle_vx_res_ax),
                   mean_absolute_error(labels_ax, bicycle_res_ax), mean_absolute_error(labels_ax, bicycle_vx_res_ax),
                   titles, 'Long. acc. ax', filepath2plots + 'metrics/ax_metrics_' + str(counter) + '.' + save_format)
    save_histogram(root_mean_squared_error(labels_vx, bicycle_res_vx), root_mean_squared_error(labels_vx, bicycle_vx_res_vx),
                   mean_absolute_error(labels_vx, bicycle_res_vx), mean_absolute_error(labels_vx, bicycle_vx_res_vx),
                   titles, 'Long. vel. vx', filepath2plots + 'metrics/vx_metrics_' + str(counter) + '.' + save_format)
    save_histogram(root_mean_squared_error(labels_vy, bicycle_res_vy), root_mean_squared_error(labels_vy, bicycle_vx_res_vy),
                   mean_absolute_error(labels_vy, bicycle_res_vy), mean_absolute_error(labels_vy, bicycle_vx_res_vy),
                   titles, 'Lat. vel. vy', filepath2plots + 'metrics/vy_metrics_' + str(counter) + '.' + save_format)


# ----------------------------------------------------------------------------------------------------------------------

def save_histogram(bike_rmse, bikeFx_rmse, bike_mae, bikeFx_mae, titles, value, savename):
    metrics_labels = ['RMSE', 'MAE']

    # Extract MSE and MAE data for each feature
    metrics_bicycle_values = list({'RMSE': bike_rmse, 'MAE': bike_mae}.values())
    metrics_bicycle_vx_comp_values = list({'RMSE': bikeFx_rmse, 'MAE': bikeFx_mae}.values())

    # Plotting
    x = np.arange(len(metrics_labels))  # X locations for features
    width = 0.25  # Width of the bars

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda

    plt.bar(x - width / 2, metrics_bicycle_values, width, label=titles[0], color='green')
    plt.bar(x + width / 2, metrics_bicycle_vx_comp_values, width, label=titles[1],
                     color='orange')

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

def save_to_csv(data, title, path_, counter):
    new_column = np.array([['RMSE'], ['MAE']])
    data = np.hstack((new_column, data))
    np.savetxt(path_ + title + ' ' + str(counter) + '.csv', data,
               header=', Bike yaw rate, Bike Vx, Bike Vy, Bike Ax, Bike Ay, '
                      'Bike w/Fx yaw rate, Bike w/Fx Vx, Bike w/Fx Vy, Bike w/Fx Ax, Bike w/Fx Ay',
               delimiter=',', fmt='%s', comments='')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(bicycle_res, bicycle_vx_res, label, titles, value, savename):
    colors = ['xkcd:red', 'xkcd:green', 'xkcd:orange', 'xkcd:purple', 'xkcd:black']
    plt.figure(figsize=(33, 9))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', labelsize=30)   # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda
    ax = plt.gca()

    if 'yaw' not in savename:
        if 'vx' not in savename:
            ax.yaxis.set_major_locator(MultipleLocator(2.5))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(5.0))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    time_values = np.linspace(0, len(label) / 100, len(label))
    plt.plot(time_values, bicycle_res, label=titles[0], color=colors[1], alpha=1.0, linewidth=2.5)
    plt.plot(time_values, bicycle_vx_res, label=titles[1], color=colors[2], alpha=1.0, linewidth=2.5)

    if label is not None:
        plt.plot(time_values, label, label='Ground Truth', color='xkcd:blue', alpha=1.0, linewidth=2.5)

    plt.ylabel(value, labelpad=12)
    plt.xlabel('Time [s]', labelpad=6)
    if 'ax' in savename:
        plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    plt.savefig(savename, format=save_format, dpi=300)
    plt.ion()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

save_format = 'eps'
directories = ['/grip_1_perf_100/', '/grip_1_perf_75/', '/grip_1_perf_50/',
               '/grip_08_perf_100/', '/grip_08_perf_75/', '/grip_08_perf_50/',
               '/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/']

path_to_bicycle_res = '../matlab/paths_with_pacejka_1/bike/sim_test_'

path_to_bicycle_vx_computed_res = '../matlab/paths_with_pacejka_1/bike_with_vx_computed/sim_test_'
# path_to_bicycle_vx_computed_res = '../matlab/paths_with_pacejka_06/sim_test_'

path_to_plots = '../matlab/plots/bike_combined/pacejka_mu1/'

path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_paramstudy_'

# titles = ['Bicycle model with μy(0.6)=0.6*μy(1.0)', 'Bicycle model', 'Bicycle model with Fx as input']
titles = ['Bicycle model', 'Bicycle model with Fx as input']

for num_test, dir_ in tqdm(enumerate(directories)):
    path_to_labels = path_to_labels[:path_to_labels.rfind('paramstudy_') + 11] + dir_.replace('/', '') + '.csv'

    path_to_bicycle_res = (
            path_to_bicycle_res[:path_to_bicycle_res.rfind('sim_test_') + 9] + dir_.replace('/', '') + '.csv')

    path_to_bicycle_vx_computed_res = (
            path_to_bicycle_vx_computed_res[:path_to_bicycle_vx_computed_res.rfind('sim_test_') + 9] +
            dir_.replace('/', '') + '.csv')

    plot(bicycle_res=path_to_bicycle_res,
         bicycle_vx_computed_res=path_to_bicycle_vx_computed_res,
         titles=titles,
         filepath2plots=path_to_plots,
         labels=path_to_labels,
         counter=dir_.replace('/', ''))
