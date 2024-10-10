import sys
from random import random

import numpy as np

import src
import helper_funcs_NN
import visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error


np.random.seed(7)

# ----------------------------------------------------------------------------------------------------------------------
# Manage Paths ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# create a dictionary which contains paths to all relevant folders and files
path_dict = helper_funcs_NN.src.manage_paths.manage_paths()

# ----------------------------------------------------------------------------------------------------------------------
# Run Tests ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# rete recurrent
# path_to_model = 'outputs/2024_05_20/11_30_21/keras_model_recurrent.h5'
# path_to_results = 'outputs/2024_05_20/11_30_21/'


# rete feedforward
path_to_model = 'outputs/2024_05_19/17_00_32/keras_model.h5'
path_to_results = 'outputs/2024_05_19/17_00_32/'

path_to_data = 'inputs/trainingdata/new/test_set_'

# create a dictionary which contains all parameters
params_dict = helper_funcs_NN.src.handle_params.handle_params(path_dict=path_dict)

for count in range(1, 4):

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=path_to_data,
                                           nn_mode="feedforward",
                                           test_type='perf_' + str(count),
                                           counter=count)

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=path_to_data,
                                           nn_mode='recurrent',
                                           test_type='perf_' + str(count),
                                           counter=count)

    # save and plot results (if activated in parameter file)
    visualization.plot_results.plot_run_test_CRT(path_dict=path_dict,
                                                 params_dict=params_dict,
                                                 path_to_model=path_to_model,
                                                 path_to_results=path_to_results,
                                                 counter='perf_' + str(count))


grips = ['mu_1', 'mu_06', 'mu_08', 'mu_0603', 'mu_06045', 'mu_0806', 'mu_0804']
for grip_ in grips:

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=path_to_data,
                                           nn_mode="feedforward",
                                           test_type=str(grip_),
                                           counter=grip_)

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=path_to_data,
                                           nn_mode='recurrent',
                                           test_type=str(grip_),
                                           counter=grip_)

    # save and plot results (if activated in parameter file)
    visualization.plot_results.plot_run_test_CRT(path_dict=path_dict,
                                                 params_dict=params_dict,
                                                 path_to_model=path_to_model,
                                                 path_to_results=path_to_results,
                                                 counter=grip_)

"""
index_last_slash = path_to_model.rfind('/')
path_to_results = path_to_model[:index_last_slash + 1]

for count in range(1, 3):

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=None,
                                           nn_mode="feedforward",
                                           counter=count)

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')

        src.run_neuralnetwork.run_test_CRT(path_dict=path_dict,
                                           params_dict=params_dict,
                                           path_to_model=path_to_model,
                                           path_to_data=None,
                                           nn_mode='recurrent',
                                           counter=count)

    # save and plot results (if activated in parameter file)
    visualization.plot_results.plot_run_test_CRT(path_dict=path_dict,
                                                 params_dict=params_dict,
                                                 path_to_results=path_to_results,
                                                 counter=count)

print('---------------------------------------------------------------------------------------------------------------')

# exit python if evaluation is disabled (NeuralNetwork_Settings.run_file_mode == 0)
if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 0:
    sys.exit('SYSTEM EXIT: exit due to run_file_mode is set to zero to avoid testing the neural network against '
             + 'vehicle sensor data')

for i_count in range(0, params_dict['Test']['n_test']):

    idx_start = params_dict['Test']['run_timestart'] + i_count * params_dict['Test']['iteration_step']

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     path_to_model=path_to_model,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="feedforward")

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')

        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     path_to_model=path_to_model,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="recurrent")

    # save and plot results (if activated in parameter file)
    visualization.plot_results.plot_run(path_dict=path_dict,
                                        params_dict=params_dict,
                                        path_to_results=path_to_results,
                                        counter=i_count,
                                        start=idx_start)



"""