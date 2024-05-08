import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def plot(TUM_filepath2results_ff: str,
         TUM_filepath2results_rr: str,
         STAN_filepath2results: str,
         filepath2testdata: str,
         filepath2plots: str,
         counter: int):

    # load results
    with open(TUM_filepath2results_ff, 'r') as fh:
        TUM_results_ff = np.loadtxt(fh)

    with open(TUM_filepath2results_rr, 'r') as fh:
        TUM_results_rr = np.loadtxt(fh)

    with open(STAN_filepath2results, 'r') as fh:
        STAN_results = np.loadtxt(fh)

    # load label data
    with open(filepath2testdata, 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    vx_TUM_result_ff = TUM_results_ff[:, 0][:, np.newaxis]
    vx_TUM_result_rr = TUM_results_rr[:, 0][:, np.newaxis]
    vx_STAN_result = STAN_results[:, 2][:, np.newaxis]

    vy_TUM_result_ff = TUM_results_ff[:, 1][:, np.newaxis]
    vy_TUM_result_rr = TUM_results_rr[:, 1][:, np.newaxis]
    vy_STAN_result = STAN_results[:, 1][:, np.newaxis]

    yaw_TUM_result_ff = TUM_results_ff[:, 2][:, np.newaxis]
    yaw_TUM_result_rr = TUM_results_rr[:, 2][:, np.newaxis]
    yaw_STAN_result = STAN_results[:, 0][:, np.newaxis]

    ax_TUM_result_ff = TUM_results_ff[:, 3][:, np.newaxis]
    ax_TUM_result_rr = TUM_results_rr[:, 3][:, np.newaxis]

    ay_TUM_result_ff = TUM_results_ff[:, 4][:, np.newaxis]
    ay_TUM_result_rr = TUM_results_rr[:, 4][:, np.newaxis]

    vx_label = labels[:, 0][:, np.newaxis]
    vy_label = labels[:, 1][:, np.newaxis]
    yaw_label = labels[:, 2][:, np.newaxis]
    ax_label = labels[:, 3][:, np.newaxis]
    ay_label = labels[:, 4][:, np.newaxis]

    """
    # calculate scaled results
    scaler_results = MinMaxScaler(feature_range=(0, 1))
    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label, ax_label, ay_label), axis=1)
    
    scaler_temp_result_TUM_ff = np.concatenate(
        (vx_TUM_result_ff, vy_TUM_result_ff, yaw_TUM_result_ff, ax_TUM_result_ff, ay_TUM_result_ff), axis=1)
    scaler_temp_TUM_ff = np.concatenate((scaler_temp_result_TUM_ff, scaler_temp_label), axis=0)
    scaler_temp_result_TUM_rr = np.concatenate(
        (vx_TUM_result_rr, vy_TUM_result_rr, yaw_TUM_result_rr, ax_TUM_result_rr, ay_TUM_result_rr), axis=1)
    scaler_temp_TUM_rr = np.concatenate((scaler_temp_result_TUM_rr, scaler_temp_label), axis=0)

    scaler_temp_label = np.concatenate((vx_label, vy_label, yaw_label), axis=1)
    scaler_temp_result_STAN = np.concatenate(
        (vx_STAN_result, vy_STAN_result, yaw_STAN_result), axis=1)
    scaler_temp_STAN = np.concatenate((scaler_temp_result_STAN, scaler_temp_label), axis=0)

    scaler_results_TUM_ff = scaler_results.fit(scaler_temp_TUM_ff)
    scaler_temp_result_TUM_ff = scaler_results.transform(scaler_temp_result_TUM_ff)
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

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: Test CRT')

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
    """

    print('\n')

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(yaw_TUM_result_ff, yaw_TUM_result_rr, yaw_STAN_result, yaw_label, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(vy_TUM_result_ff, vy_TUM_result_rr, vy_STAN_result, vy_label, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '_.png')
    plot_and_save(vx_TUM_result_ff, vx_TUM_result_rr, vx_STAN_result, vx_label, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '_.png')
    plot_and_save(ay_TUM_result_ff, ay_TUM_result_rr, [], ay_label, 'Lat. acc. ay [m/s2]',
                  filepath2plots + 'ay_test_' + str(counter) + '_.png')
    plot_and_save(ax_TUM_result_ff, ax_TUM_result_rr, [], ax_label, 'Long. acc. ax [m/s2]',
                  filepath2plots + 'ax_test_' + str(counter) + '_.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(TUM_ff, TUM_rr, STAN, label, value, savename):

    plt.figure(figsize=(25, 10))
    ax = plt.gca()  # Ottieni l'oggetto degli assi corrente

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


def complete_path(path: str, num: int) -> str:
    index_last_underscore = path.rfind('_')
    path = path[:index_last_underscore + 1]
    path += str(num) + '.csv'
    return path



TUM_path_to_results_ff = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_06/09_52_48/matfiles/prediction_result_feedforward_CRT_'
TUM_path_to_results_rr = 'NeuralNetwork_for_VehicleDynamicsModeling/outputs/2024_05_06/14_20_41/matfiles/prediction_result_recurrent_CRT_'
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


