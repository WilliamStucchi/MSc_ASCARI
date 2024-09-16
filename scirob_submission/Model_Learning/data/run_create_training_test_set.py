from create_training_test_set import *

# ----------------------------------------------------------------------------------------------------------------------

path_to_data = '../../../../CRT_data/'

"""
path_to_output = 'piste_training_complete/'
create_piste_training_complete(path_to_data, path_to_output, 32)
"""

path_to_output = 'new/'
# create_training_set_car_perf(path_to_data, path_to_output, 16, 1, include_grip=True)
# create_training_set_road_grip(path_to_data, path_to_output, 17, 1)
# create_training_set_mass(path_to_data, path_to_output, 1, 1)
# create_training_scheduled_sampling(path_to_data, path_to_output, 16, 1)
create_test_static_equilibrium(path_to_output, 90)
# create_test_set_param_study(path_to_data, path_to_output)

# create_test_set_car_perf(path_to_data, path_to_output, 4)
# create_test_set_road_grip(path_to_data, path_to_output, 7)
# create_test_set_mass(path_to_data, path_to_output, 6)
