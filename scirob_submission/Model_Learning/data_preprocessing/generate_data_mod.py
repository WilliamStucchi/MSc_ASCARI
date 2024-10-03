# Nathan Spielberg
# DDL 10.17.2018

# Generate Data from Bike model for a variety of cases

# Run all required imports
from new_data_generation_functions import *
from parameters.learning_params import *
import numpy as np

# Set the Random Seed for Repeatability!
np.random.seed(1)

#######################################################################
# Case 1: Regular Bike Model High Friction
#######################################################################
print('Generating experiment 1 data!')
# Generate Data
data = gen_data_mod(Param, Veh)

print(data.shape)

"""for elem in data:
    print(len(elem))
    print(elem[0:5])
    print(elem[5:10])
    print(elem[10:15])
    print(elem[15:20])
    input('Waiting....')"""

np.savetxt('../data/new/bicycle_model_360.csv', data, delimiter=',')

# Shuffle and Divide into datasets
# train, dev, test = shuffle_and_divide(data, Param)
# train, dev, test = shuffle_and_divide_modified(data, Param)

# Write out the first generated dataset , dev=dev, test=test
"""np.savez("../data/gen/exp_1_mod", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])"""
exit('Creation completed')
#######################################################################
# Case 2: Longitudinal WT
#######################################################################
print('Generating experiment 2 data!')
# Set Relevant Parameters
Param["WEIGHT_TRANSFER"] = True

# Generate Data
data = gen_data_mod(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide_modified(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_2_mod", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 3: Tire Relaxation Length
#######################################################################
print('Generating experiment 3 data!')

Param["WEIGHT_TRANSFER"] = False
Param["RELAX_LENGTH"] = True

# Generate Data
data = gen_data_mod(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide_modified(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_3_mod", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 4: Tire Force Coupling
#######################################################################

#######################################################################
# Case 5: Snow and Dry Friction
#######################################################################
print('Generating experiment 5 data!')

Param["TWO_FRIC"] = True
Param["WEIGHT_TRANSFER"] = False
Param["RELAX_LENGTH"] = False

# Generate Data
data = gen_data_mod(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide_modified(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_5_mod", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 6: All of the Effects
#######################################################################
print('Generating experiment 6 data!')

Param["TWO_FRIC"] = True
Param["WEIGHT_TRANSFER"] = True
Param["RELAX_LENGTH"] = True

# Generate Data
data = gen_data_mod(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide_modified(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_6_mod", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 7: Experimental Data in Ice and Snow
#######################################################################
"""
#load npz file
File_Direct = "../data/exp/exp_data/ice/"

#get the data from it
exp_traj    = load_exp_traj(File_Direct, Param)

#shuffle and divide it

train, dev, test = shuffle_and_divide(exp_traj, Param)

#write it out
np.savez("../data/exp/exp_data/ice", train_f = train[0], 
	                          train_t = train[1],
	                          dev_f   = dev[0],
	                          dev_t   = dev[1],
	                          test_f  = test[0],
	                          test_t  = test[1])
"""
"""
####################################################################
#Case 8: Expperimental Data in Dry Friction Conditions
#######################################################################
#load npz file
File_Direct = "../data/exp/exp_data/dry/"

#get the data from it
exp_traj    = load_exp_traj(File_Direct, Param)

#shuffle and divide it

train, dev, test = shuffle_and_divide(exp_traj, Param)

#write it out
np.savez("../data/exp/exp_data/dry", train_f = train[0], 
	                          train_t = train[1],
	                          dev_f   = dev[0],
	                          dev_t   = dev[1],
	                          test_f  = test[0],
	                          test_t  = test[1])

#######################################################################
#Case 9: Experimental Data in Ice, Snow and Dry Friction
#######################################################################
"""
"""
#load npz file
File_Direct = "../data/exp/exp_data/combined/"

#get the data from it
exp_traj    = load_exp_traj(File_Direct, Param)

#shuffle and divide it

train, dev, test = shuffle_and_divide(exp_traj, Param)

#write it out
np.savez("../data/exp/exp_data/combined", train_f = train[0], 
	                          train_t = train[1],
	                          dev_f   = dev[0],
	                          dev_t   = dev[1],
	                          test_f  = test[0],
	                          test_t  = test[1])


"""
