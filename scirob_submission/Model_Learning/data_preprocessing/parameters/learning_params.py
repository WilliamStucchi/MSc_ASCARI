#Nathan Spielberg
#DDL 7.2.2018

import numpy as np
import os

Veh           = {} #For the Vehicle Model
Param         = {} #For Everything Else

#Define Constants

#Vehicle Model Parameters
Veh["a"]                 = 1.194 
Veh["b"]                 = 1.437   
Veh["l"]                 = Veh["a"] + Veh["b"] 
Veh["h"]                 = 0.55
Veh["m"]                 = 1776.2  
Veh["Izz"]               = 2763.49 
Veh["Cf"]                = 150.0e3 
Veh["Cr"]                = 170.0e3 
Veh["mu"]                = 1.0 #Nominal Friction value
Veh["mu_2"]              = 0.3 #Low Friction value
Veh["del_lim"]           = 27.0*np.pi/180.0
Veh["p_lim"]             = 147*1e3 #Engine Power Limit
Veh["b_bias"]            = .66       
Veh["sig_f"]             = 0.4
Veh["sig_r"]             = 0.4
Veh["g"]                 = 9.81

#NN Model Parameters
Param["N1"]              = 128      #Neurons first layer
Param["N2"]              = 128      #Neurons second layer
Param["N_STATE_INPUT"]   = 5        # r Uy Ux delta Fxf
Param["T"]               = 4        #Number of delay states in the NN Model 
Param["T_MODEL"]         = 4
Param["FX_NORM"]         = Veh["m"] * 100 #Normalization of acceleration input 
Param["NN_DT"]           = 0.01

#RNN Model Parameters
Param["HIDDEN"]          = 1 #Hidden state dimension

#Vehicle Optimication Parameters
#For initial parameter guesses model- mean and std dev for gaussian sampling around 1.0
Param["LOC"]             = 1.1
Param["SCALE"]           = 0.1

#Training Parameters
Param["EPOCHS"]          = 1500
Param["LEARNING_RATE"]   = 0.0001 #retrain all with lower lr 1500 epochs at .0001
Param["TRAIN_PERCENT"]   = 0.8
Param["DEV_PERCENT"]     = 0.15
Param["BATCH_SIZE"]      = 1000

#Training Options
Param["RESTORE"]         = False 
Param["SAVE"]            = True

#Data Generation Parameters
Param["DT"]              = 0.01   #Sampling Time
Param["N_SAMPLES"]       = 200000 #Number of state transition trajectories.
Param["ADD_NOISE"]       = True
Param["TWO_FRIC"]        = False
Param["WEIGHT_TRANSFER"] = False
Param["RELAX_LENGTH"]    = False
Param["UX_LIM"]          = 35.0 
Param["N_FEATURES"]      = Param["T"]*Param["N_STATE_INPUT"] 
Param["N_TARGETS"]       = 3  # 2

#Real Data Parameters
#Filter for filtering out higher order dynamics
Param["FS"]              = (1.0/Param["DT"])*2.0*np.pi
Param["CUTOFF"]          = 6.0*2.0*np.pi

