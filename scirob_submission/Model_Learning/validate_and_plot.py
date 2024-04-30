import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from data_preprocessing.parameters.learning_params import *


def build_model():
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(units=Param['N1'], input_dim=Param['N_STATE_INPUT'] * Param['T'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N2'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N_TARGETS']))

    model.summary()

    return model


def data_to_run(input_shape: int, normalize_values: bool):
    data = np.loadtxt("data/data_to_run.csv", delimiter=",")

    # Extract values
    vx = data[:, 0].copy()
    vy = data[:, 1].copy()
    psi = data[:, 2].copy()
    delta = data[:, 5].copy()
    trl = data[:, 6].copy()
    trr = data[:, 7].copy()

    # Create final dataset
    result = np.zeros((data.shape[0], input_shape))

    # Put in final dataset
    result[:, 0] = psi
    result[:, 1] = vy
    result[:, 2] = vx
    result[:, 3] = delta

    # 0.05655 is the radius (in meters) of the wheel of a car
    # 177620 is the normalization factor adopted by the Stanford code
    result[:, 4] = (trl + trr) / (2 * 0.05 * 177620)
    # result[:, 4] = (0.5 * (trl + trr)) / 177620

    # normalize between -1 and 1
    if normalize_values:
        min_psi = result[:, 0].min()
        max_psi = result[:, 0].max()
        min_vy = result[:, 1].min()
        max_vy = result[:, 1].max()
        min_vx = result[:, 2].min()
        max_vx = result[:, 2].max()
        min_delta = result[:, 3].min()
        max_delta = result[:, 3].max()
        min_Fx = result[:, 4].min()
        max_Fx = result[:, 4].max()

        for i in range(0, result.shape[0]):
            result[i, 0] = 2 * (result[i, 0] - min_psi) / (max_psi - min_psi) - 1
            result[i, 1] = 2 * (result[i, 1] - min_vy) / (max_vy - min_vy) - 1
            result[i, 2] = 2 * (result[i, 2] - min_vx) / (max_vx - min_vx) - 1
            result[i, 3] = 2 * (result[i, 3] - min_delta) / (max_delta - min_delta) - 1
            result[i, 4] = 2 * (result[i, 4] - min_Fx) / (max_Fx - min_Fx) - 1

    """
    for i in range(100):
        #print('Data: ', data[i])
        print('-----------')
        print('Res: ', result[i])
        # input('Waiting...')
        print('-----------')
    input('wait')
    """

    return result


def dataset_separation(data_test: np.array, start: int, duration: int):
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']

    # Initial set of the test (first 4 time steps)
    initials = data_test[start:start + input_timesteps, :]
    initials = np.reshape(initials, (1, input_shape * input_timesteps))

    # Input signals
    delta = data_test[start + input_timesteps:start + duration, output_shape]
    Fx = data_test[start + input_timesteps:start + duration, output_shape + 1]

    return initials, delta, Fx


def run_test(loaded_model,
             start_point: int,
             counter: int,
             run_timespan: int,
             test_num: int):
    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']
    normalize_values = False

    # Load test data
    data = data_to_run(input_shape, normalize_values)
    # Scaling data?

    if start_point + run_timespan > data.shape[0]:
        print('test dataset fully covered -> exit main script')
        return 0

    initial, delta, Fx = dataset_separation(data, start_point, run_timespan)

    # Initialize structures
    results = np.zeros((len(delta) + input_timesteps, input_shape))
    # print('Results shape: ', results.shape)
    new_input = np.zeros((1, input_shape * input_timesteps))
    # print('New input shape: ', new_input.shape)
    # print('Initial: ', initial)

    # print('Results: ')
    for i in range(0, input_timesteps):
        results[i, 0:output_shape] = initial[:, i * input_shape:i * input_shape + output_shape]
        # print('Results: ', results)
        # print('Initials: ', initial)
        # print('------------')

        # Run test
    for i_count in tqdm(range(0, len(delta))):
        if i_count == 0:
            data_convert = initial
        else:
            data_convert = new_input

        # Predict
        result_process = loaded_model.predict(data_convert) * dt + data_convert[:, input_shape *
                                                                                   (input_timesteps - 1):input_shape * (
                input_timesteps - 1) + output_shape]

        if abs(result_process[:, 0]) > 1000 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print('Error in computing the prediction. Values are too big!')
            return 0

        """
        print("Data: ", data_convert)
        print("Data: ", data_convert[:, 0:5])
        print("Data: ", data_convert[:, 5:10])
        print("Data: ", data_convert[:, 10:15])
        print("Data: ", data_convert[:, 15:20])
        print("Pure prediction: ", loaded_model.predict(data_convert))
        print("Corrected prediction: ", loaded_model.predict(data_convert) * dt)
        print("Correction: ", result_process)
        input('Waiting...')
        """

        results[i_count + input_timesteps, 0:output_shape] = result_process

        # Create next input with the previous prediction
        temp = np.zeros((1, input_shape * input_timesteps))
        temp[:, 0:input_shape * (input_timesteps - 1)] = data_convert[0, input_shape:input_shape * input_timesteps]

        # Add prediction to the next input
        temp[:,
        input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] = result_process

        # Add control signals
        temp[:, input_shape * (input_timesteps - 1) + output_shape] = delta[i_count]
        temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = Fx[i_count]

        # print('Temp: ', temp)
        new_input = temp
        # input('Waiting...')

    results[:, output_shape:input_shape] = data[idx_start:idx_start + len(delta) + input_timesteps,
                                           output_shape:input_shape]

    for element in results:
        print(element)

    print('Saving test results in: \"results/gen_test_3_w/test_' + str(test_num) + '/prediction_result_' + str(counter) + '.csv\"')
    np.savetxt('results/gen_test_3_w/test_' + str(test_num) + '/prediction_result_' + str(counter) + '.csv', results)
    # input('wait')

    return 1


def plot_run(start: int, counter: int, run_timespan: int, i: int):
    path_to_results = 'results/gen_test_3_w/test_' + str(i) + '/prediction_result_' + str(counter) + '.csv'
    path_to_labels = 'data/data_to_run.csv'

    # Load results
    results = np.loadtxt(path_to_results, delimiter=' ')
    # Load labels
    labels = np.loadtxt(path_to_labels, delimiter=',')

    dyaw_result = results[:, 0][:, np.newaxis]
    vy_result = results[:, 1][:, np.newaxis]
    vx_result = results[:, 2][:, np.newaxis]

    dyaw_label = labels[start:run_timespan + start, 2][:, np.newaxis]
    vy_label = labels[start:run_timespan + start, 1][:, np.newaxis]
    vx_label = labels[start:run_timespan + start, 0][:, np.newaxis]

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
    print('MSE AND MAE OF UNSCALED VALUES: ' + 'Test No. ' + str(counter))

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

    print('MSE AND MAE OF SCALED VALUES: ' + 'Test No. ' + str(counter))

    data = np.asarray([mean_squared_error(dyaw_label_scaled, dyaw_result_scaled),
                       mean_squared_error(vx_label_scaled, vx_result_scaled),
                       mean_squared_error(vy_label_scaled, vy_result_scaled),
                       mean_absolute_error(dyaw_label_scaled, dyaw_result_scaled),
                       mean_absolute_error(vx_label_scaled, vx_result_scaled),
                       mean_absolute_error(vy_label_scaled, vy_result_scaled), ]).reshape(2, 3).round(round_digits)

    for row_head, row_data in zip(row_header, data):
        print(row_format.format(row_head, *row_data))

    print('\n')

    # plot and save comparison between NN predicted and actual vehicle state
    plot_and_save(dyaw_result, dyaw_label, dyaw_diff, 'Yaw rate [rad/s]',
                  'results/gen_test_3_w/test_' + str(i) + '/images/yaw' + str(counter) + '.png', True, True)
    plot_and_save(vy_result, vy_label, vy_diff, 'Lat. vel. vy [m/s]',
                  'results/gen_test_3_w/test_' + str(i) + '/images/vy' + str(counter) + '.png', True, True)
    plot_and_save(vx_result, vx_label, vx_diff, 'Long. vel. vx [m/s]',
                  'results/gen_test_3_w/test_' + str(i) + '/images/vx' + str(counter) + '.png', True, True)


def plot_and_save(inp_1,
                  inp_2,
                  inp_3,
                  value,
                  savename,
                  plot,
                  save):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    ax1.plot(inp_1, label='Result', color='tab:orange')
    ax1.plot(inp_2, label='Label', color='tab:blue')
    ax2.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.0)

    ax1.set_ylabel(value)
    ax2.set_ylabel('Difference label - result')
    ax1.set_xlabel('Time steps (8 ms)')
    ax2.set_xlabel('Time steps (8 ms)')
    ax1.legend()
    ax2.legend()

    if plot:
        plt.show()

    if save:
        fig.savefig(savename, format='png')
        plt.close(fig)


# Test parameters
n_test = 37
run_timestart = 0
iteration_step = 1250
run_timespan = 1500

for i in [1, 2, 3, 5, 6]:
    # Load model
    checkpoint_path = "saved_models/gen_test_3_w/exp_" + str(i) + "_mod/model.ckpt"
    loaded_model = build_model()
    loaded_model.load_weights(checkpoint_path)

    for counter in range(0, n_test):
        idx_start = run_timestart + counter * iteration_step

        print('Start test #', str(counter))
        print('------------------------------')
        if run_test(loaded_model, idx_start, counter, run_timespan, i):
            plot_run(idx_start, counter, run_timespan, i)
            # input('Waiting...')
