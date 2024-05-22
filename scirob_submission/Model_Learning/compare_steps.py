import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def plot(step_1: str,
         step_2: str,
         step_3: str,
         step_4: str,
         filepath2testdata: str,
         filepath2plots: str,
         counter: int):

    res_step_1, res_step_2, res_step_3, res_step_4 = None, None, None, None
    labels_all = None
    diff_step_1, diff_step_2, diff_step_3, diff_step_4 = None, None, None, None

    # Compute metrics
    row_header = ['MSE', 'RMSE', 'MAE', 'R2']
    column_header = ['long. vel. vx', 'lat. vel. vy', 'yaw rate']

    my_dict = {'MSE': mean_squared_error,
               'RMSE': mean_squared_error,
               'MAE': mean_absolute_error,
               'R2': r2_score
               }

    # load label data
    if filepath2testdata != 'None':
        with open(filepath2testdata, 'r') as fh:
            labels = np.loadtxt(fh, delimiter=',')

        labels_all = np.zeros((3, len(labels), 1))
        # Labels
        labels_all[0] = labels[:, 0][:, np.newaxis]
        labels_all[1] = labels[:, 1][:, np.newaxis]
        labels_all[2] = labels[:, 2][:, np.newaxis]

    # load results
    if step_1 != 'None':
        with open(step_1, 'r') as fh:
            step_1_data = np.loadtxt(fh)

        res_step_1 = np.zeros((3, len(step_1_data), 1))
        res_step_1[0] = step_1_data[:, 0][:, np.newaxis]
        res_step_1[1] = step_1_data[:, 1][:, np.newaxis]
        res_step_1[2] = step_1_data[:, 2][:, np.newaxis]
        diff_step_1 = labels_all - res_step_1
        res_scaled_step_1, lab_scaled_step_1 = scale_results(res_step_1, labels_all)

        print('\n')
        msg = 'Test CRT Step 1'
        print(msg)
        compute_metrics_matrix(row_header, column_header, my_dict, res_step_1, labels_all,
                               msg, filepath2plots, counter)

        msg = 'Test CRT Step 1 scaled'
        print(msg)
        compute_metrics_matrix_scaled(row_header, column_header, my_dict, res_scaled_step_1, lab_scaled_step_1,
                                      msg, filepath2plots, counter)

    if step_2 != 'None':
        with open(step_2, 'r') as fh:
            step_2_data = np.loadtxt(fh)

        res_step_2 = np.zeros((3, len(step_2_data), 1))
        res_step_2[0] = step_2_data[:, 0][:, np.newaxis]
        res_step_2[1] = step_2_data[:, 1][:, np.newaxis]
        res_step_2[2] = step_2_data[:, 2][:, np.newaxis]
        diff_step_2 = labels_all - res_step_2
        res_scaled_step_2, lab_scaled_step_2 = scale_results(res_step_2, labels_all)

        msg = 'Test CRT Step 2'
        print(msg)
        compute_metrics_matrix(row_header, column_header, my_dict, res_step_2, labels_all,
                               msg, filepath2plots, counter)

        msg = 'Test CRT Step 2 scaled'
        print(msg)
        compute_metrics_matrix_scaled(row_header, column_header, my_dict, res_scaled_step_2, lab_scaled_step_2,
                                      msg, filepath2plots, counter)

    if step_3 != 'None':
        with open(step_3, 'r') as fh:
            step_3_data = np.loadtxt(fh)

        res_step_3 = np.zeros((3, len(step_3_data), 1))
        res_step_3[0] = step_3_data[:, 0][:, np.newaxis]
        res_step_3[1] = step_3_data[:, 1][:, np.newaxis]
        res_step_3[2] = step_3_data[:, 2][:, np.newaxis]
        diff_step_3 = labels_all - res_step_3
        res_scaled_step_3, lab_scaled_step_3 = scale_results(res_step_3, labels_all)

        msg = 'Test CRT Step 3'
        print(msg)
        compute_metrics_matrix(row_header, column_header, my_dict, res_step_3, labels_all,
                               msg, filepath2plots, counter)

        msg = 'Test CRT Step 3 scaled'
        print(msg)
        compute_metrics_matrix_scaled(row_header, column_header, my_dict, res_scaled_step_3, lab_scaled_step_3,
                                      msg, filepath2plots, counter)

    if step_4 != 'None':
        with open(step_4, 'r') as fh:
            step_4_data = np.loadtxt(fh)

        res_step_4 = np.zeros((3, len(step_4_data), 1))
        res_step_4[0] = step_4_data[:, 0][:, np.newaxis]
        res_step_4[1] = step_4_data[:, 1][:, np.newaxis]
        res_step_4[2] = step_4_data[:, 2][:, np.newaxis]
        diff_step_4 = labels_all - res_step_4
        res_scaled_step_4, lab_scaled_step_4 = scale_results(res_step_4, labels_all)

        msg = 'Test CRT Step 4'
        print(msg)
        compute_metrics_matrix(row_header, column_header, my_dict, res_step_4, labels_all,
                               msg, filepath2plots, counter)

        msg = 'Test CRT Step 4 scaled'
        print(msg)
        compute_metrics_matrix_scaled(row_header, column_header, my_dict, res_scaled_step_4, lab_scaled_step_4,
                                      msg, filepath2plots, counter)

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(res_step_1[0], res_step_2[0], res_step_3[0], res_step_4[0], labels_all[0], 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_test_' + str(counter) + '.png')
    plot_and_save(res_step_1[1], res_step_2[1], res_step_3[1], res_step_4[1], labels_all[1], 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_test_' + str(counter) + '.png')
    plot_and_save(res_step_1[2], res_step_2[2], res_step_3[2], res_step_4[2], labels_all[2], 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_test_' + str(counter) + '.png')

    # Plot and save differences
    plot_and_save(diff_step_1[0], diff_step_2[0], diff_step_3[0], diff_step_4[0], None, 'Yaw rate [rad/s]',
                  filepath2plots + 'yaw_diff_' + str(counter) + '.png')
    plot_and_save(diff_step_1[1], diff_step_2[1], diff_step_3[1], diff_step_4[1], None, 'Lat. vel. vy [m/s]',
                  filepath2plots + 'vy_diff_' + str(counter) + '.png')
    plot_and_save(diff_step_1[2], diff_step_2[2], diff_step_3[2], diff_step_4[2], None, 'Long. vel. vx [m/s]',
                  filepath2plots + 'vx_diff_' + str(counter) + '.png')


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(step_1, step_2, step_3, step_4, label, value, savename):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()

    if 'yaw' not in savename:
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

    plt.plot(step_1, label='Step_1', color='tab:orange')
    plt.plot(step_2, label='Step_2', color='tab:red')
    plt.plot(step_3, label='Step_3', color='tab:green')
    plt.plot(step_4, label='Step_4', color='tab:purple')

    if label is not None:
        plt.plot(label, label='Ground Truth', color='tab:blue')

    plt.ylabel(value)
    plt.xlabel('Time steps (8 ms)')
    plt.legend()
    plt.grid()

    plt.savefig(savename, format='png', dpi=300)
    plt.ion()
    plt.close()


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

def complete_path(path: str, num: int) -> str:
    index_last_underscore = path.rfind('_')
    path = path[:index_last_underscore + 1]
    path += str(num) + '.csv'
    return path


# ----------------------------------------------------------------------------------------------------------------------


stan_1 = 'results/step_1/callbacks/2024_05_10/09_39_23/results_test_'
stan_2 = 'results/step_1/callbacks/2024_05_16/10_35_06/results_test_'
stan_3 = 'results/step_1/callbacks/2024_05_16/11_25_29/results_test_'
stan_4 = 'results/step_1/callbacks/2024_05_16/11_53_36/results_test_'
path_to_data = 'data/CRT/test_set_1.csv'
path_to_plots = '../../../stan combined/'

for num_tests in range(1, 2):
    stan_1 = complete_path(stan_1, num_tests)
    stan_2 = complete_path(stan_2, num_tests)
    stan_3 = complete_path(stan_3, num_tests)
    stan_4 = complete_path(stan_4, num_tests)

    # TUM_path_to_data = complete_path(path_to_data, num_tests)

    plot(step_1=stan_1,
         step_2=stan_2,
         step_3=stan_3,
         step_4=stan_4,
         filepath2testdata=path_to_data,
         filepath2plots=path_to_plots,
         counter=num_tests)
