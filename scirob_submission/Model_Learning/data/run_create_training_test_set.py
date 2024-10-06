from create_training_test_set import *

# ----------------------------------------------------------------------------------------------------------------------

path_to_data = '../../../../CRT_data/'

"""
path_to_output = 'piste_training_complete/'
create_piste_training_complete(path_to_data, path_to_output, 32)
"""

path_to_output = 'handling/'
# create_training_set_car_perf(path_to_data, path_to_output, 16, 1)
# create_training_set_road_grip(path_to_data, path_to_output, 17, 1)
# create_training_set_mass(path_to_data, path_to_output, 1, 1)
# create_training_scheduled_sampling(path_to_data, path_to_output, 16, 1)
create_test_step_steer(path_to_data, path_to_output, 15, 10)
"""create_test_ramp_steer(path_to_data, path_to_output, 4, 300)
create_test_impulse_steer(path_to_data, path_to_output, 6, 300)
create_test_sine_steer(path_to_data, path_to_output, 2, 300)
create_test_sweep_steer(path_to_data, path_to_output, 1, 300)"""
# create_test_set_param_study(path_to_data, path_to_output)

# create_test_set_car_perf(path_to_data, path_to_output, 4)
# create_test_set_road_grip(path_to_data, path_to_output, 7)
# create_test_set_mass(path_to_data, path_to_output, 6)
