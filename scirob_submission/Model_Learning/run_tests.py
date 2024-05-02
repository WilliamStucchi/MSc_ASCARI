from tqdm import tqdm
from data_preprocessing.parameters.learning_params import *


def run_test(path_to_test_data,
             loaded_model,
             start_point: int,
             run_timespan: int,
             path_save: str):
    # Load data and model parameters
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']
    dt = Param['DT']
    data = []

    # Load test data
    if path_to_test_data is not None:
        data = np.loadtxt(path_to_test_data, delimiter=',')

    if run_timespan == -1:
        run_timespan = len(data)

    if start_point + run_timespan > data.shape[0]:
        print('test dataset fully covered -> exit main script')
        return 0

    initial, delta, fx = dataset_separation(data, start_point, run_timespan)

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
        result_process = loaded_model.predict(data_convert, verbose=0) * dt + data_convert[:, input_shape *
                                                                                (input_timesteps - 1):input_shape * (
                                                                                 input_timesteps - 1) + output_shape]

        if abs(result_process[:, 0]) > 1000 or abs(result_process[:, 1]) > 1000 or abs(result_process[:, 2]) > 1000:
            print('Error in computing the prediction. Values are too big!')
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

    print('Saving test results in: ' + path_save)
    np.savetxt(path_save, results)
    # input('wait')

    return 1, len(data)


def dataset_separation(data_test: np.array, start: int, duration: int):
    input_shape = Param['N_STATE_INPUT']
    output_shape = Param['N_TARGETS']
    input_timesteps = Param['T']

    # Initial set of the test (first 4 time steps)
    initials = data_test[start:start + input_timesteps, :]
    initials = np.reshape(initials, (1, input_shape * input_timesteps))

    # Input signals
    delta = data_test[start + input_timesteps:start + duration, output_shape]
    fx = data_test[start + input_timesteps:start + duration, output_shape + 1]

    return initials, delta, fx
