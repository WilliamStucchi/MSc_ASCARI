import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def plot_run(path_to_results: str,
             path_to_labels: str,
             start: int,
             run_timespan: int,
             save_path: str,
             name: str):
    # Load results
    print(name + "[Loading data from: "  + path_to_results + ']')
    results = np.loadtxt(path_to_results, delimiter=' ')
    # Load labels
    print(name + "[Loading labels from: " + path_to_labels + ']')
    labels = np.loadtxt(path_to_labels, delimiter=',')

    if run_timespan == -1:
        run_timespan = results.shape[0]

    dyaw_result = results[:, 0][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    vx_result = results[:, 2][:, np.newaxis]

    dyaw_label = labels[start:run_timespan + start, 0][:, np.newaxis]
    vy_label = labels[start:run_timespan + start, 1][:, np.newaxis]
    vx_label = labels[start:run_timespan + start, 2][:, np.newaxis]

    dyaw_diff = dyaw_label - dyaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result

    scaler_results = MinMaxScaler(feature_range=(0, 1))

    scaler_temp_result = np.concatenate((dyaw_result, vx_result, vy_result), axis=1)
    scaler_temp_label = np.concatenate((dyaw_label, vx_label, vy_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    dyaw_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    vx_result_scaled = scaler_temp_result[:, 2]

    dyaw_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    vx_label_scaled = scaler_temp_label[:, 2]

    # print deviation from label

    round_digits = 5

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES: ' + 'Test ' + save_path)

    data = np.asarray([mean_squared_error(dyaw_label, dyaw_result),
                       mean_squared_error(vx_label, vx_result),
                       mean_squared_error(vy_label, vy_result),
                       mean_absolute_error(dyaw_label, dyaw_result),
                       mean_absolute_error(vx_label, vx_result),
                       mean_absolute_error(vy_label, vy_result)]).reshape(2, 3).round(round_digits)

    column_header = ['yaw rate', 'lat. vel. vy', 'long. vel. vx']
    row_header = ['MSE', 'MAE']

    row_format = "{:>15}" * (len(column_header) + 1)
    print(row_format.format("", *column_header))
    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('MSE AND MAE OF SCALED VALUES: ' + 'Test ' + save_path)

    data = np.asarray([mean_squared_error(dyaw_label_scaled, dyaw_result_scaled),
                       mean_squared_error(vx_label_scaled, vx_result_scaled),
                       mean_squared_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(dyaw_label_scaled, dyaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled), ]).reshape(2, 3).round(round_digits)

    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    print(name + '[Save path: ' + save_path + ']')
    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(dyaw_result, dyaw_label, dyaw_diff, 'Yaw rate [rad/s]',
                  save_path + 'images/yaw.png', True, True)
    plot_and_save(vy_result, vy_label, vy_diff, 'Lat. vel. vy [m/s]',
                  save_path + 'images/vy.png', True, True)
    plot_and_save(vx_result, vx_label, vx_diff, 'Long. vel. vx [m/s]',
                  save_path + 'images/vx.png', True, True)


def plot_and_save(inp_1,
                  inp_2,
                  inp_3,
                  value,
                  savename,
                  plot,
                  save):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')

    ax1.plot(inp_1, label='Result', color='tab:orange')
    ax1.plot(inp_2, label='Label', color='tab:blue')
    ax2.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.0)

    ax1.set_ylabel(value)
    ax2.set_ylabel('Difference label - result')
    ax1.set_xlabel('Time steps (8 ms)')
    ax2.set_xlabel('Time steps (8 ms)')
    ax1.legend()
    ax2.legend()

    plt.figure().set_figwidth(25)
    plt.figure().set_figheight(15)

    if plot:
        plt.show()

    if save:
        fig.savefig(savename, format='png')
        plt.close(fig)
