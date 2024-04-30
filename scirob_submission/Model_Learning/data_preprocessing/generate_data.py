# Nathan Spielberg
# DDL 10.17.2018

# Generate Data from Bike model for a variety of cases

# Run all required imports
from data_generation_functions import *
from parameters.learning_params import *
import numpy as np

# Set the Random Seed for Repeatability!
np.random.seed(1)

#######################################################################
# Case 1: Regular Bike Model High Friction
#######################################################################

# Generate Data
data = gen_data(Param, Veh)

# Shuffle and Divide into datasets
# train, dev, test = shuffle_and_divide(data, Param)
train, dev, test = shuffle_and_divide(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_1", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])
#######################################################################
# Case 2: Longitudinal WT
#######################################################################
# Set Relevant Pameters
Param["WEIGHT_TRANSFER"] = True

# Generate Data
data = gen_data(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_2_w", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 3: Tire Relaxation Length
#######################################################################
Param["WEIGHT_TRANSFER"] = False
Param["RELAX_LENGTH"] = True

# Generate Data
data = gen_data(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_3_w", train_f=train[0],
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
Param["TWO_FRIC"] = True
Param["WEIGHT_TRANSFER"] = False
Param["RELAX_LENGTH"] = False

# Generate Data
data = gen_data(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_5_w", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 6: All of the Effects
#######################################################################
Param["TWO_FRIC"] = True
Param["WEIGHT_TRANSFER"] = True
Param["RELAX_LENGTH"] = True

# Generate Data
data = gen_data(Param, Veh)
# Shuffle and Divide into datasets
train, dev, test = shuffle_and_divide(data, Param)

# Write out the first generated dataset , dev=dev, test=test
np.savez("../data/gen/exp_6_w", train_f=train[0],
         train_t=train[1],
         dev_f=dev[0],
         dev_t=dev[1],
         test_f=test[0],
         test_t=test[1])

#######################################################################
# Case 7: Expperimental Data in Ice and Snow
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
