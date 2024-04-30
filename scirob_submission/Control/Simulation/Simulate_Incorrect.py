import matplotlib.pyplot as plt
from utils.vehicles import *
from utils.velocityprofiles import *
from utils.tiremodels import *
from utils.simulation import *
from utils.paths import *
#from utils import paths
from utils.control import *
import sys

import numpy as np

#This defines the road-tire friction coefficient
roadMu = 1.0
incMu  = 0.6 #Average Incorrect friction value

#This defines the desired speed profile (1.0 is the modeled friction limit)
#The experiments tested on the vehicle reached the limits at 0.9
speedMu = roadMu * 0.9 
weightTransferType = 'steadystate'


veh_cor     = Vehicle(vehicleName = "niki")

#Create a vehicle with the wrong parameters
veh_inc     = Vehicle(vehicleName = "niki")
veh_inc.muF = incMu
veh_inc.muR = incMu

controllerVeh = Vehicle(vehicleName = "niki") #naively set parameters

#Create path object
track = Path()
track.loadFromMAT("maps/Skidpad_Oval.mat")
print(track)

# Create speed profile
speedProfile = RacingProfile(veh_cor, track, friction = speedMu, vMax = 99)
#Generate a speed profile with the corrrect parameters(we will overshoot)
speedProfile = RacingProfile(veh_cor, track, friction = speedMu, vMax = 99)


controller_1 = LaneKeepingController(track, veh_cor, speedProfile)
controller_2 = LaneKeepingController(track, veh_inc, speedProfile)

#simulate
bikeSim_1 = Simulation(veh_cor, controller_1, path = track, profile = speedProfile,
	mapMatchType = "closest", tires = 'fiala', weightTransferType = None) 
logFile_1 = bikeSim_1.simulate()

#simulate with the correct vehicle model
bikeSim_2 = Simulation(veh_cor, controller_2, path = track, profile = speedProfile,
	mapMatchType = "closest", tires = 'fiala', weightTransferType = None) 
logFile_2 = bikeSim_2.simulate()


plt.figure()
ax11 = plt.subplot(4, 1, 1)
ax11.plot(logFile_1["s"], logFile_1["e"], 'k', linewidth = 2, label = "Physics Model")
ax11.plot(logFile_2["s"], logFile_2["e"], 'r',linewidth = 2, label = "Physics Model with Incorrect Parameters")
plt.title("Comparison of Controllers on High Friction Oval")
plt.legend()
plt.grid(True)
plt.ylabel('Tracking Error (m)')

ax12 = plt.subplot(4, 1, 2, sharex = ax11)
ax12.plot(logFile_1["s"], logFile_1["deltaFFW"] * 180 / np.pi, 'k',linewidth = 2, label = 'Physics Model')
ax12.plot(logFile_2["s"], logFile_2["deltaFFW"] * 180 / np.pi, 'r',linewidth = 2, label = "Physics Model with Incorrect Parameters")
plt.ylabel('Steering ffw (deg)')

ax13 = plt.subplot(4, 1, 3, sharex = ax11)
ax13.plot(logFile_1["s"], logFile_1["deltaFB"] * 180 / np.pi,  'k',linewidth = 2, label = 'Physics Model')
ax13.plot(logFile_2["s"], logFile_2["deltaFB"] * 180 / np.pi,  'r',linewidth = 2, label = "Physics Model with Incorrect Parameters")
plt.ylabel('Steering fb (deg)')

ax14 = plt.subplot(4, 1, 4, sharex = ax11)
ax14.plot(logFile_1["s"], logFile_1["betaFFW"] * 180 / np.pi,  'k',linewidth = 2, label = 'Physics Model')
ax14.plot(logFile_2["s"], logFile_2["betaFFW"] * 180 / np.pi,  'r',linewidth = 2, label = "Physics Model with Incorrect Parameters")
plt.ylabel('Beta FFW (deg)')

plt.grid(True)
plt.xlabel("s (m)")


plt.show()


