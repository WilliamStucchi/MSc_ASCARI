import sys

from tqdm import tqdm
from data_preprocessing.parameters.learning_params import *
import timeit
import tensorflow as tf


def run_test(path_to_test_data,
             loaded_model,
             scaler,
             start_point: int,
             run_timespan: int,
             path_save: str,
             name: str):
    tf.compat.v1.disable_eager_execution()

    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']

    # Load test data
    if path_to_test_data is not None:
        data = np.loadtxt(path_to_test_data, delimiter=',')
    else:
        print(name + '[No test set provided. Exiting.]')
        return 0

    if run_timespan == -1:
        run_timespan = len(data)

    if start_point + run_timespan > data.shape[0]:
        print(name + '[test dataset fully covered -> exit main script]')
        return 0

    initial, _, _, _, delta, fx = dataset_separation(input_shape, output_shape, input_timesteps,
                                                     data, start_point, run_timespan)

    # Initialize structures
    results = np.zeros((len(delta) + input_timesteps, input_shape))
    # print(name + '[Results shape: ', results.shape, ']')
    new_input = np.zeros((1, input_shape * input_timesteps))
    # print(name + '[New input shape: ', new_input.shape, ']')
    # print(name + 'Initial: ', initial)
    ay = np.zeros(len(delta) + input_timesteps)
    ax = np.zeros(len(delta) + input_timesteps)
    # grip = np.zeros(len(delta) + input_timesteps)

    # print(name + 'Results: ')
    for i in range(0, input_timesteps):
        results[i, 0:output_shape] = initial[:, i * input_shape:i * input_shape + output_shape]
        # print(name + '[Results: ' + results + ']')
        # print(name + '[Initials: ' + initial + ']')
        # print('------------')

        # Run test
    for i_count in tqdm(range(0, len(delta))):
        if i_count == 0:
            data_convert = initial
        else:
            data_convert = new_input

        # Predict
        if scaler is not None:
            prediction = loaded_model.predict(scaler.transform(data_convert), verbose=0)
        else:
            prediction = loaded_model.predict(data_convert, verbose=0)

        """print(prediction.shape)
        input('wait')"""
        # grip[i_count] = prediction[:, -4]
        yaw_acc = prediction[:, -3]
        ay[i_count] = prediction[:, -2]
        ax[i_count] = prediction[:, -1]

        """# Old version of model
        result_process = prediction * dt + data_convert[:, input_shape * (input_timesteps - 1):input_shape * (
                input_timesteps - 1) + output_shape]"""

        # New version of model
        # Euler integration
        yaw_rate = yaw_acc * dt + data_convert[:, input_shape * (input_timesteps - 1):
                                                  input_shape * (input_timesteps - 1) + 1]

        # vy_dot = ay - yaw_rate * vx
        dvy = ay[i_count] - yaw_rate * data_convert[:, input_shape * (input_timesteps - 1) + 2:
                                                       input_shape * (input_timesteps - 1) + 3]

        # vx_dot = ax + yaw_rate * vy
        dvx = ax[i_count] + yaw_rate * data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                                       input_shape * (input_timesteps - 1) + 2]

        # Euler integration x(t) = x(t-1) + x_dot(t-1) * DT
        vy = dvy * dt + data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                        input_shape * (input_timesteps - 1) + 2]
        vx = dvx * dt + data_convert[:, input_shape * (input_timesteps - 1) + 2:
                                        input_shape * (input_timesteps - 1) + 3]

        result_process = np.array([yaw_rate, vy, vx]).reshape(1, 3)

        if abs(result_process[:, 0]) > 100 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print(name + '[Error in computing the prediction. Values are too big!]')
            return 0, -1

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
        temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] = result_process

        # Add control signals
        temp[:, input_shape * (input_timesteps - 1) + output_shape] = delta[i_count]
        temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = fx[i_count]

        # print('Temp: ', temp)
        new_input = temp
        # input('Waiting...')

    results[:, output_shape:input_shape] = data[start_point:start_point + len(delta) + input_timesteps,
                                           output_shape: input_shape]
    """
    for element in results:
        print(element)
    """
    # Save results
    print(name + '[Saving test results in: ' + path_save + ']')
    np.savetxt(path_save, results)
    # input('wait')

    # Save Ax and Ay
    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ax.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ax)

    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ay.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ay)

    """# Save grip
    index_last_slash = path_save.rfind('.')
    save_grip = path_save[:index_last_slash] + '_grip.csv'
    print(name + '[Saving test results in: ' + save_grip + ']')
    np.savetxt(save_grip, grip)"""

    return 1, len(data)


# ----------------------------------------------------------------------------------------------------------------------

def run_test_inject_vy(path_to_test_data,
                       loaded_model,
                       scaler,
                       start_point: int,
                       run_timespan: int,
                       path_save: str,
                       name: str):
    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']
    sum_inj = 0

    # Load test data
    if path_to_test_data is not None:
        data = np.loadtxt(path_to_test_data, delimiter=',')
    else:
        print(name + '[No test set provided. Exiting.]')
        return 0

    if run_timespan == -1:
        run_timespan = len(data)

    if start_point + run_timespan > data.shape[0]:
        print(name + '[test dataset fully covered -> exit main script]')
        return 0

    initial, _, _, _, delta, fx = dataset_separation(input_shape, output_shape, input_timesteps,
                                                     data, start_point, run_timespan, no_vy=False)

    # Initialize structures
    results = np.zeros((len(delta) + input_timesteps, input_shape))
    # print(name + '[Results shape: ', results.shape, ']')
    new_input = np.zeros((1, input_shape * input_timesteps))
    # print(name + '[New input shape: ', new_input.shape, ']')
    # print(name + 'Initial: ', initial)
    ay = np.zeros(len(delta) + input_timesteps)
    ax = np.zeros(len(delta) + input_timesteps)

    # print(name + 'Results: ')
    for i in range(0, input_timesteps):
        results[i, 0:output_shape] = initial[:, i * input_shape:i * input_shape + output_shape]
        # print(name + '[Results: ' + results + ']')
        # print(name + '[Initials: ' + initial + ']')
        # print('------------')

        # Run test
    for i_count in tqdm(range(0, len(delta))):
        if i_count == 0:
            data_convert = initial
        else:
            data_convert = new_input

        # possibility for vy to be 0
        prob = 2.0
        injected_input = data_convert.copy()
        # Generate a random number
        random_number = np.random.rand()
        # Change the second element to 0 with the given probability
        if random_number <= prob:
            injected_input[:, 1] = 0
            injected_input[:, 6] = 0
            injected_input[:, 11] = 0
            injected_input[:, 16] = 0
            sum_inj += 1

        # Predict
        if scaler is not None:
            prediction = loaded_model.predict(scaler.transform(injected_input), verbose=0)
        else:
            prediction = loaded_model.predict(injected_input, verbose=0)
        """print(prediction.shape)
        input('wait')"""
        yaw_acc = prediction[:, 0]
        ay[i_count] = prediction[:, 1]
        ax[i_count] = prediction[:, 2]
        """result_process = prediction * dt + data_convert[:, input_shape * (input_timesteps - 1):input_shape * (
                input_timesteps - 1) + output_shape]"""

        # Euler integration
        yaw_rate = yaw_acc * dt + data_convert[:, input_shape * (input_timesteps - 1):
                                                  input_shape * (input_timesteps - 1) + 1]

        # vy_dot = ay - yaw_rate * vx
        dvy = ay[i_count] - yaw_rate * data_convert[:, input_shape * (input_timesteps - 1) + 2:
                                                       input_shape * (input_timesteps - 1) + 3]

        # vx_dot = ax + yaw_rate * vy
        dvx = ax[i_count] + yaw_rate * data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                                       input_shape * (input_timesteps - 1) + 2]

        # Euler integration
        vy = dvy * dt + data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                        input_shape * (input_timesteps - 1) + 2]
        vx = dvx * dt + data_convert[:, input_shape * (input_timesteps - 1) + 2:
                                        input_shape * (input_timesteps - 1) + 3]

        result_process = np.array([yaw_rate, vy, vx]).reshape(1, 3)

        if abs(result_process[:, 0]) > 100 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print(name + '[Error in computing the prediction. Values are too big!]')
            return 0, -1

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
        temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] = result_process

        # Add control signals
        temp[:, input_shape * (input_timesteps - 1) + output_shape] = delta[i_count]
        temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = fx[i_count]

        # print('Temp: ', temp)
        new_input = temp
        # input('Waiting...')

    results[:, output_shape:input_shape] = data[start_point:start_point + len(delta) + input_timesteps,
                                           output_shape: input_shape]
    """
    for element in results:
        print(element)
    """
    # Save results
    print(name + '[Saving test results in: ' + path_save + ']')
    np.savetxt(path_save, results)
    # input('wait')

    # Save Ax and Ay
    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ax.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ax)

    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ay.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ay)

    print(name + '[Total length of test: ' + str(len(delta)) + ']')
    print(name + '[Total injections: ' + str(sum_inj) + ']')

    return 1, len(data)


# ----------------------------------------------------------------------------------------------------------------------

def run_test_no_vy(path_to_test_data,
                   loaded_model,
                   scaler,
                   start_point: int,
                   run_timespan: int,
                   path_save: str,
                   name: str):
    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT'] + 2
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']

    # Load test data
    if path_to_test_data is not None:
        data = np.loadtxt(path_to_test_data, delimiter=',')
    else:
        print(name + '[No test set provided. Exiting.]')
        return 0

    if run_timespan == -1:
        run_timespan = len(data)

    if start_point + run_timespan > data.shape[0]:
        print(name + '[test dataset fully covered -> exit main script]')
        return 0

    initial, _, vy, _, delta, ffr, ffl, frr, frl = dataset_separation_no_vy(input_shape, output_shape, input_timesteps,
                                                                data, start_point, run_timespan, no_vy=True)

    # Initialize structures
    results = np.zeros((len(delta) + input_timesteps, input_shape))
    # print(name + '[Results shape: ', results.shape, ']')
    new_input = np.zeros((1, input_shape * input_timesteps))
    """print(name + '[New input shape: ', new_input.shape, ']')
    print(name + '[Initial: ', initial, ']')"""
    ay = np.zeros(len(delta) + input_timesteps)
    ax = np.zeros(len(delta) + input_timesteps)

    # print(name + 'Results: ')
    for i in range(0, input_timesteps):
        results[i, 0] = initial[:, i * input_shape]
        results[i, 1] = vy[i]
        results[i, 2] = initial[:, i * input_shape + 1]
        """print(name + '[Results: ', results, ']')
        print(name + '[Initials: ', initial, ']')
        print('------------')"""

        # Run test
    for i_count in tqdm(range(0, len(delta))):
        if i_count == 0:
            data_convert = initial
        else:
            data_convert = new_input

        # Predict
        if scaler is not None:
            prediction = loaded_model.predict(scaler.transform(data_convert), verbose=0)
        else:
            prediction = loaded_model.predict(data_convert, verbose=0)

        # input('wait')
        yaw_acc = prediction[:, 0]
        ay[i_count] = prediction[:, 1]
        ax[i_count] = prediction[:, 2]
        """result_process = prediction * dt + data_convert[:, input_shape * (input_timesteps - 1):input_shape * (
                input_timesteps - 1) + output_shape]"""

        # Euler integration
        yaw_rate = yaw_acc * dt + data_convert[:, input_shape * (input_timesteps - 1):
                                               input_shape * (input_timesteps - 1) + 1]

        # vy_dot = ay - yaw_rate * vx
        dvy = ay[i_count] - yaw_rate * data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                                    input_shape * (input_timesteps - 1) + 2]

        # vx_dot = ax + yaw_rate * vy
        dvx = ax[i_count] + yaw_rate * results[i_count + input_timesteps - 1, 1]  # TODO: controlla che questo sia corretto

        # Euler integration
        vy = dvy * dt + results[i_count + input_timesteps - 1, 1]
        vx = dvx * dt + data_convert[:, input_shape * (input_timesteps - 1) + 1:
                                        input_shape * (input_timesteps - 1) + 2]

        result_process = np.array([yaw_rate, vy, vx]).reshape(1, 3)

        if abs(result_process[:, 0]) > 100 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print(name + '[Error in computing the prediction. Values are too big!]')
            return 0, -1

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
        temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + 2] = (
            np.delete(result_process, 1, axis=1))

        # Add control signals
        temp[:, input_shape * (input_timesteps - 1) + 2] = delta[i_count]
        temp[:, input_shape * (input_timesteps - 1) + 3] = ffr[i_count]
        temp[:, input_shape * (input_timesteps - 1) + 4] = ffl[i_count]
        temp[:, input_shape * (input_timesteps - 1) + 5] = frr[i_count]
        temp[:, input_shape * (input_timesteps - 1) + 6] = frl[i_count]

        # print('Temp: ', temp)
        new_input = temp
        # input('Waiting...')

    results[:, output_shape:input_shape] = data[start_point:start_point + len(delta) + input_timesteps,
                                           output_shape: input_shape]
    """
    for element in results:
        print(element)
    """
    # Save results
    print(name + '[Saving test results in: ' + path_save + ']')
    np.savetxt(path_save, results)
    # input('wait')

    # Save Ax and Ay
    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ax.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ax)

    index_last_slash = path_save.rfind('.')
    save_accel = path_save[:index_last_slash] + '_ay.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ay)

    return 1, len(data)


# ----------------------------------------------------------------------------------------------------------------------

def run_test_input_reordered(path_to_test_data,
                             loaded_model,
                             scaler,
                             start_point: int,
                             run_timespan: int,
                             path_save: str,
                             name: str):
    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']

    # Load test data
    if path_to_test_data is not None:
        data = np.loadtxt(path_to_test_data, delimiter=',')
    else:
        print(name + '[No test set provided. Exiting.]')
        return 0

    if run_timespan == -1:
        run_timespan = len(data)

    if start_point + run_timespan > data.shape[0]:
        print(name + '[test dataset fully covered -> exit main script]')
        return 0

    initial, _, _, _, delta, fx = dataset_separation_new(data, start_point, run_timespan)

    # Initialize structures
    results = np.zeros((len(delta) + input_timesteps, input_shape))
    # print(name + '[Results shape: ', results.shape, ']')
    new_input = np.zeros((1, input_shape * input_timesteps))
    # print(name + '[New input shape: ', new_input.shape, ']')
    # print(name + 'Initial: ', initial)
    ay = np.zeros(len(delta) + input_timesteps)
    ax = np.zeros(len(delta) + input_timesteps)

    # print(name + 'Results: ')
    for i in range(0, input_timesteps):
        for j in range(0, 3):
            results[i, j] = initial[:, j * input_timesteps + i]

    """print(name, '{Results: ', results, '}')
    print(name, '{Initials: ', initial, '}')
    input('wait...')
    print('------------')"""

    # Run test
    for i_count in tqdm(range(0, len(delta))):
        if i_count == 0:
            data_convert = initial
        else:
            data_convert = new_input

        # Predict
        if scaler is not None:
            prediction = loaded_model.predict(scaler.transform(data_convert), verbose=0)
        else:
            prediction = loaded_model.predict(data_convert, verbose=0)

        """print(prediction.shape)
        input('wait')"""
        yaw_acc = prediction[:, 0]
        ay[i_count] = prediction[:, 1]
        ax[i_count] = prediction[:, 2]
        """result_process = prediction * dt + data_convert[:, input_shape * (input_timesteps - 1):input_shape * (
                input_timesteps - 1) + output_shape]"""

        # Euler integration
        # yaw_rate(t) = yaw_acc(t) * dt + yaw_rate(t-1)
        yaw_rate = yaw_acc * dt + data_convert[:, input_timesteps - 1:input_timesteps]

        # vy_dot = ay - yaw_rate * vx
        dvy = ay[i_count] - yaw_rate * data_convert[:, 3 * input_timesteps - 1:3 * input_timesteps]

        # vx_dot = ax + yaw_rate * vy
        dvx = ax[i_count] + yaw_rate * data_convert[:, 2 * input_timesteps - 1:2 * input_timesteps]

        # Euler integration
        vy = dvy * dt + data_convert[:, 2 * input_timesteps - 1:2 * input_timesteps]
        vx = dvx * dt + data_convert[:, 3 * input_timesteps - 1:3 * input_timesteps]

        result_process = np.array([yaw_rate, vy, vx]).reshape(1, 3)

        if abs(result_process[:, 0]) > 1000 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print(name + '[Error in computing the prediction. Values are too big!]')
            return 0, -1

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
        # temp[:, 0:input_shape * (input_timesteps - 1)] = data_convert[0, input_shape:input_shape * input_timesteps]
        for i in range(input_shape):
            temp[:, i * input_timesteps:i * input_timesteps + output_shape] = \
                data_convert[:, (i * input_timesteps) + 1:(i * input_timesteps) + output_shape + 1]

        # Add prediction to the next input
        # temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] = result_process
        for i in range(1, output_shape + 1):
            temp[:, i * input_timesteps - 1] = result_process[:, i - 1]

        # Add control signals
        # temp[:, input_shape * (input_timesteps - 1) + output_shape] = delta[i_count]
        # temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = fx[i_count]
        temp[:, input_shape * input_timesteps - 5] = delta[i_count]
        temp[:, input_shape * input_timesteps - 1] = fx[i_count]

        """print(name, '{Data convert: ', data_convert, '}')
        print(name, '{Result_process: ', result_process, '}')
        print(name, '{Temp: ', temp, '}')
        input('wait...')
        print('------------')"""

        # print('Temp: ', temp)
        new_input = temp
        # input('Waiting...')

    results[:, output_shape:input_shape] = data[start_point:start_point + len(delta) + input_timesteps,
                                           output_shape: input_shape]
    """
    for element in results:
        print(element)
    """
    # Save results
    print(name + '[Saving test results in: ' + path_save + ']')
    np.savetxt(path_save, results)
    # input('wait')

    # Save Ax and Ay
    index_last_slash = path_save.rfind('_')
    save_accel = path_save[:index_last_slash + 2] + '_ax.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ax)

    index_last_slash = path_save.rfind('_')
    save_accel = path_save[:index_last_slash + 2] + '_ay.csv'
    print(name + '[Saving test results in: ' + save_accel + ']')
    np.savetxt(save_accel, ay)

    return 1, len(data)


# ----------------------------------------------------------------------------------------------------------------------

def dataset_separation(input_shape: int, output_shape: int, input_timesteps: int,
                       data_test: np.array, start: int, duration: int, no_vy: bool = False):
    # Initial set of the test (first 4 time steps)
    if not no_vy:
        initials = data_test[start:start + input_timesteps, :]
        initials = np.reshape(initials, (1, input_shape * input_timesteps))
    else:
        first = data_test[start:start + input_timesteps, :]
        initials = np.delete(first, 1, axis=1)  # delete column 1 in each row (remove vy information)
        initials = np.reshape(initials, (1, input_shape * input_timesteps))

    # Input signals
    delta = data_test[start + input_timesteps:start + duration, output_shape]
    fx = data_test[start + input_timesteps:start + duration, output_shape + 1]

    # Others
    vx = data_test[start + input_timesteps:start + duration, 2]
    vy = data_test[start + input_timesteps:start + duration, 1]
    yaw_rate = data_test[start + input_timesteps:start + duration, 0]

    return initials, yaw_rate, vy, vx, delta, fx


# ----------------------------------------------------------------------------------------------------------------------

def dataset_separation_no_vy(input_shape: int, output_shape: int, input_timesteps: int,
                             data_test: np.array, start: int, duration: int, no_vy: bool = False):
    # Initial set of the test (first 4 time steps)
    initials = data_test[start:start + input_timesteps, :-1]
    initials = np.reshape(initials, (1, input_shape * input_timesteps))

    # Input signals
    delta = data_test[start + input_timesteps:start + duration, 2]
    ffr = data_test[start + input_timesteps:start + duration, 3]
    ffl = data_test[start + input_timesteps:start + duration, 4]
    frr = data_test[start + input_timesteps:start + duration, 5]
    frl = data_test[start + input_timesteps:start + duration, 6]

    # Others
    vx = data_test[start + input_timesteps:start + duration, 1]
    vy = data_test[start + input_timesteps:start + duration, -1]
    yaw_rate = data_test[start + input_timesteps:start + duration, 0]

    return initials, yaw_rate, vy, vx, delta, ffr, ffl, frr, frl


# ----------------------------------------------------------------------------------------------------------------------

def dataset_separation_new(data_test: np.array, start: int, duration: int):
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']

    # Initial set of the test (first 4 time steps)
    yaw_init = data_test[start:start + input_timesteps, 0]
    vy_init = data_test[start:start + input_timesteps, 1]
    vx_init = data_test[start:start + input_timesteps, 2]
    steer_init = data_test[start:start + input_timesteps, 3]
    fx_init = data_test[start:start + input_timesteps, 4]

    initials = np.concatenate([yaw_init, vy_init, vx_init, steer_init, fx_init], axis=0)
    initials = np.reshape(initials, (1, input_shape * input_timesteps))

    # Input signals
    delta = data_test[start + input_timesteps:start + duration, output_shape]
    fx = data_test[start + input_timesteps:start + duration, output_shape + 1]

    # Others
    vx = data_test[start + input_timesteps:start + duration, 2]
    vy = data_test[start + input_timesteps:start + duration, 1]
    yaw_rate = data_test[start + input_timesteps:start + duration, 0]

    return initials, yaw_rate, vy, vx, delta, fx
