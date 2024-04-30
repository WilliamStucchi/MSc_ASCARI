#Nathan Spielberg and Nitin Kapania

import numpy as np
import utils.tiremodels as tm
import utils.vehicles
import utils.paths
import math
import tensorflow as tf
from casadi import *

#Controller from Nitin Kapania's PhD thesis - lookahead with augmented sideslip for
#steering feedback, longitudinal is simple feedforward and feedback control

##############################################################################################################################################
#This controller is used as a baseline using the physics-based bicycle model
##############################################################################################################################################

class LaneKeepingController():
    def __init__(self, path, vehicle, profile):
        self.path = path
        self.vehicle = vehicle
        self.profile = profile
        self.xLA = 14.2    #lookahead distance, meters
        self.kLK = 0.0538  #proportional gain , rad / meter
        self.kSpeed = 3000.0 #Speed proportional gain - N / (m/s)
        self.alphaFlim = 7.0 * np.pi / 180 #steering limits for feedforward controller
        self.alphaRlim = 5.0 * np.pi / 180 #steering limits for feedforward controller
        
        #Initialize force lookup tables for feedforward
        numTableValues = 250

        #values where car is sliding
        alphaFslide = np.abs(np.arctan(3*vehicle.muF*vehicle.m*vehicle.b/vehicle.L*vehicle.g/vehicle.Cf)) 
        alphaRslide = np.abs(np.arctan(3*vehicle.muR*vehicle.m*vehicle.a/vehicle.L*vehicle.g/vehicle.Cr))

        alphaFtable = np.linspace(-alphaFslide, alphaFslide, numTableValues)
        alphaRtable = np.linspace(-alphaRslide, alphaRslide, numTableValues) # vector of rear alpha (rad)
        
        FyFtable = tm.fiala(vehicle.Cf, vehicle.muF, vehicle.muF, alphaFtable, vehicle.FzF)
        FyRtable = tm.fiala(vehicle.Cr, vehicle.muR, vehicle.muR, alphaRtable, vehicle.FzR)

        #flip arrays so Fy is increasing - important for numpy interp!!
        self.alphaFtable = np.flip(alphaFtable, 0)
        self.alphaRtable = np.flip(alphaRtable, 0)
        self.FyFtable = np.flip(FyFtable, 0) 
        self.FyRtable = np.flip(FyRtable, 0)



    def getDeltaFB(self, localState, betaFFW):
        kLK = self.kLK
        xLA = self.xLA
        e = localState.e
        deltaPsi = localState.deltaPsi
        # + betaFFW
        deltaFB = -kLK * (e + xLA * np.sin(deltaPsi + betaFFW))
        #NS Modified to not have betaFFW
        #deltaFB = -kLK * (e + xLA * np.sin(deltaPsi))
        return deltaFB


    def speedTracking(self, localState):

        #note - interp requires rank 0 arrays
        AxTable = self.profile.Ax
        UxTable = self.profile.Ux
        sTable = self.profile.s
        m = self.vehicle.m
        fdrag = self.vehicle.dragCoeff
        frr = self.vehicle.rollResistance

        s = localState.s
        Ux = localState.Ux

        AxDes = np.interp(s, sTable, AxTable) #run interp every time - this is slow, but we may be able to get away with
        UxDes = np.interp(s, sTable, UxTable) #run interp every time - this is slow, but we may be able to get away with


        FxFFW = m*AxDes + np.sign(Ux)*fdrag*Ux ** 2 + frr*np.sign(Ux) # Feedforward
        FxFB = -self.kSpeed*(Ux - UxDes) # Feedback
        FxCommand = FxFFW + FxFB
        return FxCommand, UxDes, AxDes, FxFFW, FxFB


    def getDeltaFFW(self, localState, K):
        a = self.vehicle.a
        b = self.vehicle.b
        L = self.vehicle.L
        m = self.vehicle.m
        Ux = localState.Ux


        FyFdes = b / L * m * Ux**2 * K
        FyRdes = a / b * FyFdes

        alphaFdes = _force2alpha(self.FyFtable, self.alphaFtable, FyFdes)
        alphaRdes = _force2alpha(self.FyRtable, self.alphaRtable, FyRdes)

        betaFFW = alphaRdes + b * K 
        deltaFFW = K * L + alphaRdes - alphaFdes

        return deltaFFW, betaFFW, FyFdes, FyRdes, alphaFdes, alphaRdes  
        
    def lanekeeping(self, localState):
        #note - interp requires rank 0 arrays
        sTable = self.path.s
        kTable = self.path.curvature

        K = np.interp(localState.s, sTable, kTable) #run interp every time - this is slow, but we may be able to get away with    
        deltaFFW, betaFFW, FyFdes, FyRdes, alphaFdes, alphaRdes = self.getDeltaFFW(localState, K)
        deltaFB = self.getDeltaFB(localState, betaFFW)
        delta = deltaFFW + deltaFB
        return delta, deltaFFW, deltaFB, K, alphaFdes, alphaRdes, betaFFW

    def updateInput(self, localState, controlInput):
        delta, deltaFFW, deltaFB, K, alphaFdes, alphaRdes, betaFFW = self.lanekeeping(localState)
        Fx, UxDes, AxDes, FxFFW, FxFB = self.speedTracking(localState)
        controlInput.update(delta, Fx)
        auxVars = {'K': K , 'UxDes': UxDes, 'AxDes': AxDes, 'alphaFdes': alphaFdes,
        'alphaRdes': alphaRdes, 'deltaFFW': deltaFFW, 'deltaFB': deltaFB, 'betaFFW': betaFFW}

        return auxVars


class OpenLoopControl():
    def __init__(self, vehicle, delta = 2 * np.pi / 180, Fx = 100.):
        self.delta = delta
        self.Fx = Fx
            

    #Note, Local state not needed for open loop control, no feedback!    
    def updateInput(self, localState, controlInput):
        
        delta = self.delta
        Fx = self.Fx
            
        #Curvature is 0 for open loop control - no path to track 
        auxVars = {'K': 0., 'UxDes': 0.}
        controlInput.update(delta, Fx)

        return auxVars


class ControlInput:
    def __init__(self):
        self.delta = 0.0
        self.Fx = 0.0

    def update(self, delta, Fx):
        self.delta = delta
        self.Fx = Fx



##############################################################################################################################################
#Generate a Neural Network Feedforward and Feedback Controller Based on the Pretrained Network
##############################################################################################################################################

class RNN_Inv_Feedforward():
    def __init__(self, path, vehicle, profile, init_delta):
        self.path = path
        self.vehicle = vehicle
        self.profile = profile
        self.xLA = 14.2    #lookahead distance, meters
        self.kLK = 0.0538  #proportional gain , rad / meter
        self.kSpeed = 3000.0 #Speed proportional gain - N / (m/s)
        self.init_delta = init_delta

        self.NUM_TARGETS   = 2
        self.N_S           = 5 
        self.T             = 4

        self.Fx_norm       = vehicle.m * 100

        data = np.load('Pretrained_Weights/Network_Weights.npz') #try this first

        self.w1     = data["arr_0"][0].T
        self.b1     = data["arr_0"][1]
        self.w2     = data["arr_0"][2].T
        self.b2     = data["arr_0"][3]
        self.w3     = data["arr_0"][4].T
        self.b3     = data["arr_0"][5]


        #Start of opti stuff
        self.opti = casadi.Opti() #get the opti toolbox 

        #Set up the opti problem and all that stuff
        self.delta_opt      = self.opti.variable(1)
        self.uy_opt         = self.opti.variable(1)

        #Set the Parameters
        self.ux_opt         = self.opti.parameter(1)
        self.K_opt          = self.opti.parameter(1)

        #Defined by others.
        self.r_opt          = self.ux_opt*self.K_opt
        self.fxf_opt        = -vehicle.m*self.uy_opt*self.r_opt #self.opti.parameter(1, self.T)

        #Create Vars to Store Last good controls
        self.last_delta     = 0
        self.last_beta      = 0

        #Here we should pass in p cont dyn and other things. 
        p      = np.array( [ self.K_opt, self.ux_opt, self.fxf_opt] ) #k ux fxf

        y_in   = vertcat( self.delta_opt, self.r_opt, self.uy_opt)

        #self.opti.subject_to( self.continuous_dyn(y_in, p) == 0)
        y_dot  = self.continuous_dyn(y_in, p)

        #pull out what we want to optimize (now solving approx problem)
        r_dot  = y_dot[0]
        uy_dot = y_dot[1]

        #Cost Function here- use the 1 norm
        J = r_dot**2 + uy_dot**2

        #Need to minimize it!
        self.opti.minimize(J)

        #Need To Set the constraints here to get reasonable inputs
        self.opti.subject_to(self.delta_opt > -25.0*np.pi/180.0)
        self.opti.subject_to(self.delta_opt <  25.0*np.pi/180.0)
        self.opti.subject_to(self.uy_opt > -10.0)
        self.opti.subject_to(self.uy_opt <  10.0)


    def continuous_dyn(self, y, p):

        L         = self.vehicle.L
        fx_norm   = self.Fx_norm
        T         = self.T

        # load NN weights
        w1        = self.w1
        b1        = self.b1
        w2        = self.w2
        b2        = self.b2
        w3        = self.w3
        b3        = self.b3


        delta     = y[0] #now is T x 1
        r         = y[1]
        uy        = y[2]

        k         = p[0]
        ux        = p[1]
        fxf       = p[2]
        

        for j in range(T):
            if j==0:
                n_in      =  vertcat(r, uy, ux, delta, fxf / fx_norm)
            else:
                n_in      =  vertcat(n_in, r, uy, ux, delta, fxf / fx_norm)

        a1        =  mtimes(w1, n_in) + b1

        for i in range(len(b1)):
            a1[i] = log(1 + exp(a1[i]) )

        a2        = mtimes(w2, a1) + b2

        for i in range(len(b2)):
            a2[i] = log(1 + exp(a2[i]) )

        output    = mtimes(w3, a2) + b3


        #pull out the velocity state derivs

        r_dot     = output[0]

        uy_dot    = output[1]


        y_dot = vertcat(r_dot, uy_dot)


        return y_dot

    def getDeltaFB(self, localState, betaFFW):
        kLK = self.kLK
        xLA = self.xLA
        e = localState.e
        deltaPsi = localState.deltaPsi
        #here we dont use the beta ffw + betaFFW
        deltaFB = -kLK * (e + xLA * np.sin(deltaPsi + betaFFW))
        #deltaFB = 0
        return deltaFB


    def speedTracking(self, localState):

        #note - interp requires rank 0 arrays
        AxTable = self.profile.Ax
        UxTable = self.profile.Ux
        sTable = self.profile.s
        m = self.vehicle.m
        fdrag = self.vehicle.dragCoeff
        frr = self.vehicle.rollResistance

        s = localState.s
        Ux = localState.Ux

        AxDes = np.interp(s, sTable, AxTable) #run interp every time - this is slow, but we may be able to get away with
        UxDes = np.interp(s, sTable, UxTable) #run interp every time - this is slow, but we may be able to get away with


        FxFFW = m*AxDes + np.sign(Ux)*fdrag*Ux ** 2 + frr*np.sign(Ux) # Feedforward
        FxFB = -self.kSpeed*(Ux - UxDes) # Feedback
        FxCommand = FxFFW + FxFB
        return FxCommand, UxDes, AxDes, FxFFW, FxFB

    def getDeltaFFW(self, localState, K, delta, ax):
    
        self.opti.set_value(self.ux_opt,  localState.Ux)
        self.opti.set_value(self.K_opt,   K)

        #set initial guesses
        if self.init_delta == 'kinematic':
            self.opti.set_initial(self.delta_opt , K*self.vehicle.L)
        elif self.init_delta == 'rand':
            self.opti.set_initial(self.delta_opt ,  np.random.uniform(low = -10*np.pi/180.0 , high = 10*np.pi/180.0))
        else:
            self.opti.set_initial(self.delta_opt , 0)

        #Try Solving
        self.opti.solver('ipopt')
        p_opts = {'expand'   : False, 'print_time' : False}
        s_opts = {'max_iter' : 1000,'print_level' : 0} #this is not great need to look into

        self.opti.solver('ipopt', p_opts, s_opts)

        sol = self.opti.solve()

        status = sol.stats()['return_status']
        
        print('{}: cost function error = {}'.format(status, np.sqrt(sol.value(self.opti.f))))
        
        #Print casadi warnings? need to figure out, solving but bad solution i think.
        #look at casadi documentation at airport.

        #These lines are for code generation for the vehicle
        #This generates a shared object for real time usage
        """
        print("generating c code... ", end='')
        sys.stdout.flush()
        self.opti.advanced.casadi_solver.generate_dependencies("libsentinel.c")
        print("done")
        print("compiling library... ", end='')
        sys.stdout.flush()
        os.system('gcc -fPIC -O3 -shared libsentinel.c -o libsentinel.so');
        print("done")
        raise
        """

        if status != 'Solve_Succeeded' and status != 'Solved_To_Acceptable_Levels':
            deltaFFW = self.last_delta
            betaFFW  = self.last_beta
            print("ERROR ERROR ERROR ERROR ERROR ERROR ")
        else:
            deltaFFW        = sol.value(self.delta_opt)
            betaFFW         = np.arctan(sol.value(self.uy_opt) / localState.Ux) 
            self.last_delta = deltaFFW
            self.last_beta  = betaFFW 

        FyFdes = 0
        FyRdes = 0
        alphaFdes = 0
        alphaRdes = 0
        return deltaFFW, betaFFW, FyFdes, FyRdes, alphaFdes, alphaRdes 

    def lanekeeping(self, localState, controlInput):
        #note - interp requires rank 0 arrays
        sTable = self.path.s
        kTable = self.path.curvature

        K = np.interp(localState.s, sTable, kTable) #run interp every time - this is slow, but we may be able to get away with    
        deltaFFW, betaFFW, FyFdes, FyRdes, alphaFdes, alphaRdes = self.getDeltaFFW(localState, K, controlInput.delta, controlInput.Fx / self.vehicle.m)
        deltaFB = self.getDeltaFB(localState, betaFFW)
        delta = deltaFFW + deltaFB
        return delta, deltaFFW, deltaFB, K, alphaFdes, alphaRdes, betaFFW

    def updateInput(self, localState, controlInput):
        delta, deltaFFW, deltaFB, K, alphaFdes, alphaRdes, betaFFW = self.lanekeeping(localState, controlInput)
        Fx, UxDes, AxDes, FxFFW, FxFB = self.speedTracking(localState)
        controlInput.update(delta, Fx)
        auxVars = {'K': K , 'UxDes': UxDes, 'AxDes': AxDes, 'alphaFdes': alphaFdes,
        'alphaRdes': alphaRdes, 'deltaFFW': deltaFFW, 'deltaFB': deltaFB, 'betaFFW': betaFFW}

        return auxVars


##################################HELPER FUNCTIONS ##############################################

def _force2alpha(forceTable, alphaTable, Fdes):
        if Fdes > max(forceTable):
             Fdes = max(forceTable) - 1

        elif Fdes < min(forceTable):
             Fdes = min(forceTable) + 1

        #note - need to slice to rank 0 for np to work
        #note - x values must be increasing in numpy interp!!!
        alpha = np.interp(Fdes, forceTable ,alphaTable)
        

        return alpha







