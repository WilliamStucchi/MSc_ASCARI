import numpy as np
import sys
from tqdm import tqdm
import os.path
from tensorflow import keras
import tensorflow as tf

# custom modules
from .prepare_data import *

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""

# SET FLOATING POINT PRECISION
np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})

# ----------------------------------------------------------------------------------------------------------------------

def run_nn(path_dict: dict,
           params_dict: dict,
           path_to_model: str,
           startpoint: float,
           counter: int,
           nn_mode: str):
    """Runs the neural network to test its predictions against actual vehicle data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param startpoint:          row index where to start using provided test data
    :type startpoint: float
    :param counter:             number of current testing loop (only used for naming of output files)
    :type counter: int
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforward" or "recurrent"')

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2inputs_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2inputs_trainedmodel_recurr']

    else:
        if path_to_model == '':
            path2scaler = path_dict['filepath2scaler_save']
        else:
            path2scaler = path_to_model[:path_to_model.rfind('/') + 1] + 'scaler.plk'

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2results_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2results_trainedmodel_recurr']

    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    if startpoint + params_dict['Test']['run_timespan'] > data.shape[0]:
        sys.exit("test dataset fully covered -> exit main script")

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # scale dataset the vanish effects of different input data quantities
    data = scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar = \
        create_dataset_separation_run(data, params_dict, startpoint,
                                                       params_dict['Test']['run_timespan'], nn_mode)

    # load neural network model
    if path_to_model == '':
        model = keras.models.load_model(path2model)
    else:
        model = keras.models.load_model(path_to_model)

    results = np.zeros((len(torqueRR_Nm) + input_timesteps, input_shape))

    if nn_mode == "feedforward":
        new_input = np.zeros((1, input_shape * input_timesteps))

        for m in range(0, input_timesteps):
            results[m, 0:output_shape] = initial[:, m * input_shape:m * input_shape + output_shape]

    elif nn_mode == "recurrent":
        new_input = np.zeros((1, input_timesteps, input_shape))

        results[0:input_timesteps, :] = initial[0, :, :]

    for i_count in tqdm(range(0, len(torqueRR_Nm))):

        if i_count == 0:
            data_convert = initial

        else:
            data_convert = new_input

        result_process = model.predict(data_convert, verbose=0)
        results[i_count + input_timesteps, 0:output_shape] = result_process

        # convert test data
        if nn_mode == "feedforward":
            temp = np.zeros((1, input_shape * input_timesteps))
            temp[:, 0:input_shape * (input_timesteps - 1)] = data_convert[0, input_shape:input_shape * input_timesteps]

            temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] \
                = result_process

            temp[:, input_shape * (input_timesteps - 1) + output_shape] = steeringangle_rad[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = torqueRL_Nm[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 2] = torqueRR_Nm[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 3] = brakepresF_bar[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 4] = brakepresR_bar[i_count]

        elif nn_mode == "recurrent":
            temp = np.zeros((1, input_timesteps, input_shape))
            temp[0, 0:input_timesteps - 1, :] = data_convert[0, 1:input_timesteps, :]

            temp[0, input_timesteps - 1, 0:output_shape] = result_process
            temp[0, input_timesteps - 1, output_shape] = steeringangle_rad[i_count]
            temp[0, input_timesteps - 1, output_shape + 1] = torqueRL_Nm[i_count]
            temp[0, input_timesteps - 1, output_shape + 2] = torqueRR_Nm[i_count]
            temp[0, input_timesteps - 1, output_shape + 3] = brakepresF_bar[i_count]
            temp[0, input_timesteps - 1, output_shape + 4] = brakepresR_bar[i_count]

        new_input = temp  # new input of the nn is the output with the current control inputs

    results[:, output_shape:input_shape] = data[startpoint:startpoint + len(steeringangle_rad) + input_timesteps,
                                           output_shape:input_shape]

    results = scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    if path_to_model == '':
        np.savetxt(
            os.path.join(path_dict['path2results_matfiles'], 'prediction_result_' + nn_mode + str(counter) + '.csv'),
            results)
    else:
        index_last_slash = path_to_model.rfind('/')
        output_path = path_to_model[:index_last_slash] + 'prediction_result_' + nn_mode + str(counter) + '.csv'
        np.savetxt(output_path, results)


# ----------------------------------------------------------------------------------------------------------------------

def run_test_CRT(path_dict: dict,
                 params_dict: dict,
                 path_to_model: object,
                 path_to_data: object,
                 nn_mode: str,
                 test_type: str,
                 counter: str):
    """ Runs the neural network to test its predictions against actual vehicle data from CarRealTime

        :param counter:
        :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
        :type path_dict: dict
        :param params_dict:         dictionary which contains all parameters necessary to run this module
        :type params_dict: dict
    """

    tf.compat.v1.disable_eager_execution()

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforward" or "recurrent"')

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2inputs_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2inputs_trainedmodel_recurr']

    else:
        if path_to_model is not None:
            path2scaler = path_to_model[:path_to_model.rfind('/') + 1] + 'scaler.plk'
        else:
            path2scaler = path_dict['filepath2scaler_save']

        if nn_mode == "feedforward":
            path2model = path_dict['filepath2results_trainedmodel_ff']
        elif nn_mode == "recurrent":
            path2model = path_dict['filepath2results_trainedmodel_recurr']

    if path_to_data is not None:
        with open(str(path_to_data) + str(counter) + '.csv', 'r') as fh:
            data = np.loadtxt(fh, delimiter=',')
    else:
        with open(path_dict['filepath2inputs_testdata_CRT'] + '_' + str(counter) + '.csv', 'r') as fh:
            data = np.loadtxt(fh, delimiter=',')

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # scale dataset the vanish effects of different input data quantities
    data = scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar = \
        create_dataset_separation_run_for_testing(data, params_dict, nn_mode)

    # load neural network model
    if path_to_model is not None:
        model = keras.models.load_model(path_to_model)
    else:
        model = keras.models.load_model(path2model)

    results = np.zeros((len(torqueRR_Nm) + input_timesteps, input_shape))

    if nn_mode == "feedforward":
        new_input = np.zeros((1, input_shape * input_timesteps))

        for m in range(0, input_timesteps):
            results[m, 0:output_shape] = initial[:, m * input_shape:m * input_shape + output_shape]

    elif nn_mode == "recurrent":
        new_input = np.zeros((1, input_timesteps, input_shape))

        results[0:input_timesteps, :] = initial[0, :, :]

    for i_count in tqdm(range(0, len(torqueRR_Nm))):

        if i_count == 0:
            data_convert = initial

        else:
            data_convert = new_input

        result_process = model.predict(data_convert, verbose=0)
        results[i_count + input_timesteps, 0:output_shape] = result_process

        # convert test data
        if nn_mode == "feedforward":
            temp = np.zeros((1, input_shape * input_timesteps))
            temp[:, 0:input_shape * (input_timesteps - 1)] = data_convert[0, input_shape:input_shape * input_timesteps]

            temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] \
                = result_process

            temp[:, input_shape * (input_timesteps - 1) + output_shape] = steeringangle_rad[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = torqueRL_Nm[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 2] = torqueRR_Nm[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 3] = brakepresF_bar[i_count]
            temp[:, input_shape * (input_timesteps - 1) + output_shape + 4] = brakepresR_bar[i_count]

        elif nn_mode == "recurrent":
            temp = np.zeros((1, input_timesteps, input_shape))
            temp[0, 0:input_timesteps - 1, :] = data_convert[0, 1:input_timesteps, :]

            temp[0, input_timesteps - 1, 0:output_shape] = result_process
            temp[0, input_timesteps - 1, output_shape] = steeringangle_rad[i_count]
            temp[0, input_timesteps - 1, output_shape + 1] = torqueRL_Nm[i_count]
            temp[0, input_timesteps - 1, output_shape + 2] = torqueRR_Nm[i_count]
            temp[0, input_timesteps - 1, output_shape + 3] = brakepresF_bar[i_count]
            temp[0, input_timesteps - 1, output_shape + 4] = brakepresR_bar[i_count]

        new_input = temp  # new input of the nn is the output with the current control inputs

    results[:, output_shape:input_shape] = data[0:len(steeringangle_rad) + input_timesteps, output_shape:input_shape]

    results = scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    if path_to_model == '':
        np.savetxt(
            os.path.join(path_dict['path2results_matfiles'],
                         'prediction_result_' + nn_mode + '_CRT_' + str(counter)+'.csv'),
            results)
    else:
        index_last_slash = path_to_model.rfind('/')
        output_path = (path_to_model[:index_last_slash + 1] + '/matfiles/results_test_' + test_type + '.csv')
        np.savetxt(output_path, results)
