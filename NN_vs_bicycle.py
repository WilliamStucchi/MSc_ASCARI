import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------------------------------------------------------------------

def plot(NN_res: str,
         NN_res_ay: str,
         bicycle_res: str,
         titles: list,
         labels: str,
         filepath2plots: str,
         counter: str):

    # load results
    with open(NN_res, 'r') as fh:
        nn_res_ = np.loadtxt(fh)
    with open(NN_res_ay, 'r') as fh:
        nn_res_ay_ = np.loadtxt(fh)
    with open(bicycle_res, 'r') as fh:
        data = pd.read_csv(bicycle_res, dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        bicycle_res_ = np.array(data, dtype=float)
        bicycle_res_[np.isnan(bicycle_res_)] = 0

    # load label data
    with open(labels, 'r') as fh:
        labels_ = np.loadtxt(fh, delimiter=',')

    # Extract results for each feature
    nn_res_yaw = nn_res_[:, 0][:, np.newaxis]
    nn_res_ay = nn_res_ay_[:, np.newaxis]

    bicycle_res_yaw = bicycle_res_[:, 1][:, np.newaxis]
    bicycle_res_ay = bicycle_res_[:, 2][:, np.newaxis]

    # Labels
    labels_yaw = labels_[:, 2]
    labels_ay = labels_[:, 4]

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(nn_res_yaw, bicycle_res_yaw, labels_yaw, titles, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(nn_res_ay, bicycle_res_ay, labels_ay, titles, 'Lat. acc. ay [m/s2]',
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

def plot_and_save(nn_res, bicycle_res, label, titles, value, savename):
    colors = ['xkcd:red', 'xkcd:green', 'xkcd:purple', 'xkcd:orange', 'xkcd:black']
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    if label is not None:
            plt.plot(label, label='Ground Truth', color='xkcd:blue', alpha=0.5)

    plt.plot(nn_res, label=titles[0], color=colors[0], alpha=0.75)
    plt.plot(bicycle_res, label=titles[1], color=colors[1], alpha=0.75)

    plt.ylabel(value)
    plt.xlabel('Time steps (10 ms)')
    plt.legend(loc='best')
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
    plt.ion()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------


path_to_NN_res = 'scirob_submission/Model_Learning/results/step_1/callbacks/2024_08_30/12_10_48/results_test_perf_'
path_to_bicycle_res = '../matlab/test_perf_'
path_to_plots = '../test/NN_vs_bicycle/'
path_to_labels = 'NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/new/test_set_'
titles = ['Neural Network model', 'Bicycle model']

for num_test in range(1, 4):

    path_to_NN_res = path_to_NN_res[:path_to_NN_res.rfind('_') + 1] + str(num_test) + '.csv'
    path_to_NN_res_ay = path_to_NN_res[:path_to_NN_res.rfind('_') + 1] + str(num_test) + '_ay.csv'
    path_to_labels = (path_to_labels[:path_to_labels.rfind('_') + 1] + str(num_test) + '.csv')

    path_to_bicycle_res = path_to_bicycle_res[:path_to_bicycle_res.rfind('_') + 1] + str(num_test) + '.csv'

    plot(NN_res=path_to_NN_res,
         NN_res_ay=path_to_NN_res_ay,
         bicycle_res=path_to_bicycle_res,
         titles=titles,
         filepath2plots=path_to_plots,
         labels=path_to_labels,
         counter=str(num_test))

