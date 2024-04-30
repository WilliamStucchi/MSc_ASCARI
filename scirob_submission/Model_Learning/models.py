# Nathan Spielberg
# DDL 10.17.2018

# Build the Models for Training

# from data_preprocessing.parameters import learning_params
import numpy as np
import tensorflow.compat.v1 as tf


# First make the bike model class
class Bike_Model():
    def __init__(self, Param, Veh, inputs, labels):
        # input an label are placeholders.
        self.inputs = inputs
        self.labels = labels
        self.fiala_tf
        self.prediction
        self.optimize
        self.mse

        # Model Params
        self.N_STATE_INPUT = Param["N_STATE_INPUT"]
        self.T = Param["T"]
        self.BATCH_SIZE = Param["BATCH_SIZE"]
        self.DT = Param["DT"]
        self.Fx_norm = Param["FX_NORM"]

        # Learning Params
        self.LR = Param["LEARNING_RATE"]

        # Bike Params
        self.a_tf = tf.constant(Veh["a"], dtype=tf.float64)
        self.b_tf = tf.constant(Veh["b"], dtype=tf.float64)
        self.L = tf.constant(Veh["a"] + Veh["b"], dtype=tf.float64)
        self.g = tf.constant(Veh["g"], dtype=tf.float64)

        self.m_const = tf.constant(Veh["m"], dtype=tf.float64)
        self.I_const = tf.constant(Veh["Izz"], dtype=tf.float64)
        self.Ca_f_const = tf.constant(Veh["Cf"], dtype=tf.float64)
        self.Ca_r_const = tf.constant(Veh["Cr"], dtype=tf.float64)
        self.mu_const = tf.constant(Veh["mu"], dtype=tf.float64)

        self.loc = Param["LOC"]  # This defines best first guess mean of parameter multipliers
        self.scale = Param["SCALE"]  # This is the variance on our initial param percentage estimate

        self.m_norm = tf.Variable(np.random.normal(self.loc, self.scale), name='m', dtype=tf.float64)
        self.I_norm = tf.Variable(np.random.normal(self.loc, self.scale), name='I', dtype=tf.float64)
        self.Ca_f_norm = tf.Variable(np.random.normal(self.loc, self.scale), name='Ca_f', dtype=tf.float64)
        self.Ca_r_norm = tf.Variable(np.random.normal(self.loc, self.scale), name='Ca_r', dtype=tf.float64)
        self.mu_norm = tf.Variable(np.random.normal(self.loc, self.scale), name='mu', dtype=tf.float64)

        self.m_tf = self.m_const
        self.I_tf = self.I_const
        self.Ca_f = self.Ca_f_norm * self.Ca_f_const
        self.Ca_r = self.Ca_r_norm * self.Ca_r_const
        self.mu_tf = self.mu_norm * self.mu_const

        # Create the Model
        # first slice out inputs
        r = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 5)], [-1, 1])
        uy = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 4)], [-1, 1])
        ux = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 3)], [-1, 1])
        delta = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 2)], [-1, 1])
        fxf = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 1)], [-1, 1]) * self.Fx_norm

        fz_f = (1.0 / self.L) * (self.m_tf * self.b_tf * self.g)
        fz_r = (1.0 / self.L) * (self.m_tf * self.a_tf * self.g)
        alpha_f = tf.atan(tf.divide((uy + self.a_tf * r), ux)) - delta
        alpha_r = tf.atan(tf.divide((uy - self.b_tf * r), ux))
        fy_f = self.fiala_tf(alpha_f, self.Ca_f, self.mu_tf, fz_f)
        fy_r = self.fiala_tf(alpha_r, self.Ca_r, self.mu_tf, fz_r)

        # Calc Derivs
        r_dot = (1 / self.I_tf) * (
                self.a_tf * fy_f * tf.cos(delta) + self.a_tf * fxf * tf.sin(delta) - self.b_tf * fy_r)
        uy_dot = (1 / self.m_tf) * (fy_f * tf.cos(delta) + fxf * tf.sin(delta) + fy_r) - r * ux
        ux_dot = (1 / self.m_tf) * (fxf * tf.cos(delta)) + r * uy

        # Predict Next State
        model_next_r = r + self.DT * r_dot
        model_next_uy = uy + self.DT * uy_dot
        model_next_ux = ux + self.DT * ux_dot

        self.pred_bike = tf.concat([model_next_r, model_next_uy, model_next_ux], 1)

        # And for training
        self.loss_bike = tf.losses.mean_squared_error(self.pred_bike, self.labels)
        self.train_bike = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_bike)

    # tire model for tensorflow
    def fiala_tf(self, alpha, Ca, mu, fz):
        alpha_slide = tf.abs(tf.atan(3 * mu * fz / Ca))
        fy = tf.where(tf.abs(alpha) < alpha_slide,
                      (-Ca * tf.tan(alpha) + ((Ca ** 2) / (3 * mu * fz)) * (tf.abs(tf.tan(alpha))) * tf.tan(alpha) -
                       ((Ca ** 3) / (9 * (mu ** 2) * (fz ** 2))) * (tf.tan(alpha) ** 3) * (1 - 2 * mu / (3 * mu))),
                      -mu * fz * tf.sign(alpha))
        return fy

    def prediction(self):
        # Predict next states for bike model

        return self.pred_bike

    def optimize(self):
        return self.train_bike

    def mse(self):
        return self.loss_bike


class NN_Model():
    def __init__(self, Param, inputs, labels):
        # input an label are placeholders.
        self.inputs = inputs
        self.labels = labels
        self.prediction
        self.optimize
        self.mse

        # Model Params
        self.N_STATE_INPUT = Param["N_STATE_INPUT"]
        self.T = Param["T"]
        self.T_MODEL = Param["T_MODEL"]
        self.BATCH_SIZE = Param["BATCH_SIZE"]
        self.DT = Param["DT"]
        self.Fx_norm = Param["FX_NORM"]
        self.N1 = Param["N1"]
        self.N2 = Param["N2"]
        self.N_TARGETS = Param["N_TARGETS"]

        # Learning Params
        self.LR = Param["LEARNING_RATE"]

        # Model Creation
        # first slice out inputs
        r = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 5)], [-1, 1])
        uy = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 4)], [-1, 1])

        nn_inputs = tf.slice(self.inputs, (0, (self.N_STATE_INPUT * self.T - self.N_STATE_INPUT * self.T_MODEL)),
                             (-1, self.N_STATE_INPUT * self.T_MODEL))
        net_nn = tf.keras.layers.Dense(self.N1, activation=tf.nn.softplus)(
            nn_inputs)  # pass the first value from iter.get_next() as input
        net_nn = tf.keras.layers.Dense(self.N2, activation=tf.nn.softplus)(net_nn)  # could also use tf.nn.relu

        # w: Euler integration of step Dt
        net_out = tf.keras.layers.Dense(self.N_TARGETS)(net_nn)
        print(type(net_out))
        net_out = tf.cast(net_out, dtype=tf.float64)
        self.pred_nn = tf.concat([r, uy], 1) + self.DT * net_out

        # Training Setup
        self.loss_nn = tf.losses.mean_squared_error(self.pred_nn, self.labels)
        self.train_nn = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_nn)

    def prediction(self):
        return self.pred_nn

    def optimize(self):
        return self.train_nn

    def mse(self):
        return self.loss_nn


class NN_Model_mod():
    def __init__(self, Param, inputs, labels):
        # input an label are placeholders.
        self.inputs = inputs
        self.labels = labels
        self.prediction
        self.optimize
        self.mse

        # Model Params
        self.N_STATE_INPUT = Param["N_STATE_INPUT"]
        self.T = Param["T"]
        self.T_MODEL = Param["T_MODEL"]
        self.BATCH_SIZE = Param["BATCH_SIZE"]
        self.DT = Param["DT"]
        self.Fx_norm = Param["FX_NORM"]
        self.N1 = Param["N1"]
        self.N2 = Param["N2"]
        self.N_TARGETS = Param["N_TARGETS"]

        # Learning Params
        self.LR = Param["LEARNING_RATE"]

        # Model Creation
        # first slice out inputs
        r = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 5)], [-1, 1])  # returns the last known value of r
        uy = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 4)], [-1, 1])  # returns the last known value of uy
        ux = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - 3)], [-1, 1])  # returns the last known value of ux

        nn_inputs = tf.slice(self.inputs, [0, (self.N_STATE_INPUT * self.T - self.N_STATE_INPUT * self.T_MODEL)],
                             [-1, self.N_STATE_INPUT * self.T_MODEL])  # output of this instruction should be 20 (as 5*4 is the dimension of the input)
        net_nn = tf.keras.layers.Dense(self.N1, activation=tf.nn.softplus)(nn_inputs)  # pass the first value from iter.get_next() as input
        net_nn = tf.keras.layers.Dense(self.N2, activation=tf.nn.softplus)(net_nn)  # could also use tf.nn.relu

        # w: Euler integration of step Dt
        net_out = tf.keras.layers.Dense(self.N_TARGETS)(net_nn)
        print(type(net_out))
        net_out = tf.cast(net_out, dtype=tf.float64)
        self.pred_nn = tf.concat([r, uy, ux], 1) + self.DT * net_out

        # Training Setup
        self.loss_nn = tf.losses.mean_squared_error(self.pred_nn, self.labels)
        self.train_nn = tf.train.AdamOptimizer(learning_rate=self.LR).minimize(self.loss_nn)

    def prediction(self):
        return self.pred_nn

    def optimize(self):
        return self.train_nn

    def mse(self):
        return self.loss_nn
