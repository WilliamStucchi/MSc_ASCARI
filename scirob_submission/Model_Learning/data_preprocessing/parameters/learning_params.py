#Nathan Spielberg
#DDL 7.2.2018

import numpy as np
import os

Veh           = {} #For the Vehicle Model
Param         = {} #For Everything Else

#Define Constants

#Vehicle Model Parameters
Veh["a"]                 = 1.51
Veh["b"]                 = 1.23
Veh["l"]                 = Veh["a"] + Veh["b"] 
Veh["h"]                 = 0.55
Veh["m"]                 = 1554.32
Veh["Izz"]               = 2198

Veh["Cf"]                = 16.1
Veh["Cr"]                = 20.1
Veh['Tf']                = 0.020
Veh['Tr']                = 0.022
Veh['Hf']                = 0.51
Veh['Hr']                = 0.57
Veh['Pf']                = 0.7
Veh['Pr']                = 0.7
Veh['Peakyf']            = np.tan(np.radians(10))
Veh['Peakyr']            = np.tan(np.radians(9))
Veh['muyf']              = 0.0115
Veh['muyr']              = 0.018

Veh['front_normal_load'] = 6827.5
Veh['rear_normal_load']  = 8420.4

Veh["mu"]                = 1.0     #Nominal Friction value
Veh["mu_2"]              = 0.6     #Low Friction value
Veh["del_lim"]           = 360 * np.pi / 180.0
Veh['SW_rate']           = 15.56

Veh["p_lim"]             = 147*1e3 #Engine Power Limit

Veh["b_bias"]            = .66       
Veh["sig_f"]             = 0.4
Veh["sig_r"]             = 0.4
Veh["g"]                 = 9.81

Veh['Fzf0']              = 6827.5
Veh['Fzr0']              = 8420.4
Veh['CxA']               = 0.728
Veh['CzfA']              = 0.0855
Veh['CzrA']              = 0.2899
Veh['rho_air']           = 1.225

#NN Model Parameters
Param["N1"]              = 128      #Neurons first layer
Param["N2"]              = 128      #Neurons second layer
Param["N_STATE_INPUT"]   = 5        # r Uy Ux delta Fxf
Param["T"]               = 4        #Number of delay states in the NN Model 
Param["T_MODEL"]         = 4
Param["FX_NORM"]         = Veh["m"] #Normalization of acceleration input
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
Param["UX_LIM"]          = 87.0
Param["N_FEATURES"]      = Param["T"]*Param["N_STATE_INPUT"] 
Param["N_TARGETS"]       = 3  # 2

#Real Data Parameters
#Filter for filtering out higher order dynamics
Param["FS"]              = (1.0/Param["DT"])*2.0*np.pi
Param["CUTOFF"]          = 6.0*2.0*np.pi

