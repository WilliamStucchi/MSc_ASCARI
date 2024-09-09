import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from scirob_submission.Model_Learning.data_preprocessing.parameters.learning_params import *


class NN_Model_V2():
    def __init__(self):
        self.units_first_layer = Param['N1']
        self.units_second_layer = Param['N2']
        self.input_dim = Param['N_STATE_INPUT'] * Param['T']
        self.output_dim = Param['N_TARGETS'] + 1
        self.LR = Param["LEARNING_RATE"]

    def build_model(self, seed):
        # Set tf seed
        tf.random.set_seed(seed)

        # Input layer
        input_layer = tfkl.Input(shape=(self.input_dim, ), name='input_layer')

        # Hidden layers
        x = tfkl.Dense(units=self.units_first_layer, activation='relu')(input_layer)
        x = tfkl.Dense(units=self.units_second_layer, activation='relu')(x)

        # Output layer
        output_layer = tfkl.Dense(units=self.output_dim)(x)

        model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model_v2')

        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(weight_decay=5e-4),
                      weighted_metrics=['mse', 'mae'])

        model.summary()

        return model



