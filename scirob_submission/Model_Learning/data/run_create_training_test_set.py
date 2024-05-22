from create_training_test_set import *

# ----------------------------------------------------------------------------------------------------------------------

path_to_data = '../../../../CRT_data/'

"""
path_to_output = 'piste_training_complete/'
create_piste_training_complete(path_to_data, path_to_output, 32)
"""
"""
path_to_output = 'CRT/'
create_training_set_v1(path_to_data, path_to_output, 32, 1)
create_training_set_v1(path_to_data, path_to_output, 32, 2)
create_training_set_v1(path_to_data, path_to_output, 32, 3)
create_training_set_v1(path_to_data, path_to_output, 32, 4)
"""

path_to_output = 'new/'
create_training_set(path_to_data, path_to_output, 17, 1)
"""create_training_set(path_to_data, path_to_output, 32, 2)
create_training_set(path_to_data, path_to_output, 32, 3)
create_training_set(path_to_data, path_to_output, 32, 4)"""

create_test_set(path_to_data, path_to_output, 4)
