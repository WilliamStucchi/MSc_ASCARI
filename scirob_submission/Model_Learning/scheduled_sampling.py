import tensorflow as tf
import numpy as np
import pandas as pd
from scirob_submission.Model_Learning.models_v2.models_V2 import *
import datetime
import math
from pathlib import Path
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------

def get_set(dataset, perc_valid):
    train_features, train_lab = get_training(dataset, perc_valid)
    assert_creation(train_features, train_lab)
    val_features, val_lab = get_validation(dataset, perc_valid)
    assert_creation(val_features, val_lab)

    return train_features, train_lab, val_features, val_lab


# ----------------------------------------------------------------------------------------------------------------------

def get_training(dataset, perc_valid):
    features = dataset[:math.floor(len(dataset) * (1 - perc_valid)), :-3]
    labels = dataset[:math.floor(len(dataset) * (1 - perc_valid)), -3:]
    return features, labels


# ----------------------------------------------------------------------------------------------------------------------

def get_validation(dataset, perc_valid):
    features = dataset[math.floor(len(dataset) * (1 - perc_valid)):, :-3]
    labels = dataset[math.floor(len(dataset) * (1 - perc_valid)):, -3:]
    return features, labels


# ----------------------------------------------------------------------------------------------------------------------

def assert_creation(features, labels):
    assert len(features) != 0
    assert len(labels) != 0
    assert len(features) == len(labels)


# ----------------------------------------------------------------------------------------------------------------------

def train_step(x, y, model, input_shape, input_timesteps, dt):
    cumulative_loss = 0.0
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    bs = tf.shape(x)[0]
    # y = tf.reshape(y, [bs, 3])

    yaw_rate = np.zeros(bs)
    vy = np.zeros(bs)
    vx = np.zeros(bs)

    mod_x = np.array(x)
    # mod_x = np.zeros((bs, input_shape * input_timesteps))

    with (tf.device('/GPU:0')):
        for k in range(5):
            with tf.GradientTape() as tape:
                if k != 0:
                    # print(mod_x[:, k])
                    for t in range(bs):
                        mod_x[t, k, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + 1] = yaw_rate[t]
                        mod_x[t, k, input_shape * (input_timesteps - 1) + 1:input_shape * (input_timesteps - 1) + 2] = vy[t]
                        mod_x[t, k, input_shape * (input_timesteps - 1) + 2:input_shape * (input_timesteps - 1) + 3] = vx[t]

                new_x = tf.convert_to_tensor(mod_x)
                inp = new_x[:, k]
                """print(inp)
                input('wait')"""

                prediction = model(inp, training=True)

                for j in range(bs):
                    corresponding_input = np.array(new_x[j:j + 1, k])
                    bs_prediction = prediction[j:j + 1]

                    yaw_acc = float(bs_prediction[:, 0])
                    ay = float(bs_prediction[:, 1])
                    ax = float(bs_prediction[:, 2])

                    # Euler integration
                    yaw_rate[j] = yaw_acc * dt + corresponding_input[:, input_shape * (input_timesteps - 1):
                                                              input_shape * (input_timesteps - 1) + 1]

                    # vy_dot = ay - yaw_rate * vx
                    dvy = ay - yaw_rate[j] * corresponding_input[:, input_shape * (input_timesteps - 1) + 2:
                                                                   input_shape * (input_timesteps - 1) + 3]
                    # print('dvy ', dvy)

                    # vx_dot = ax + yaw_rate * vy
                    dvx = ax + yaw_rate[j] * corresponding_input[:, input_shape * (input_timesteps - 1) + 1:
                                                                   input_shape * (input_timesteps - 1) + 2]
                    # print('dvx ', dvx)

                    # Euler integration
                    vy[j] = dvy * dt + corresponding_input[:, input_shape * (input_timesteps - 1) + 1:
                                                    input_shape * (input_timesteps - 1) + 2]
                    vx[j] = dvx * dt + corresponding_input[:, input_shape * (input_timesteps - 1) + 2:
                                                    input_shape * (input_timesteps - 1) + 3]

                    """if j == 0:
                        print('Shape ', corresponding_input.shape)
                        print('Input ', corresponding_input)
                        print('Prediction ', bs_prediction)
    
                        print('yaw rate ', yaw_rate[j])
                        print('vy ', vy[j])
                        print('vx ', vx[j])
                        input('wait')"""


                loss_sampling = loss_fn(y[:, k], prediction)
                cumulative_loss += loss_sampling

                gradients = tape.gradient(loss_sampling, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return cumulative_loss / 5


# ----------------------------------------------------------------------------------------------------------------------

input_shape = 5
input_timesteps = 4
dt = 0.01

model_path = 'saved_models/step_1/callbacks/2024_07_18/13_02_33/'
model = tf.keras.models.load_model(model_path + 'keras_model.h5')

# GET SCHEDULED SAMPLING DATA
dataset_scheduling = np.loadtxt('data/scheduled_sampling/train_data_step_no_shuffle_1_sched.csv', delimiter=',')
print('TOTAL LENGTH OF THE DATASET: ', len(dataset_scheduling))
print(dataset_scheduling.shape)

sequences = []
labels = []
for i in range(len(dataset_scheduling) - 4):
    temp = []
    temp2 = []
    sequences.append(temp)
    labels.append(temp2)

for i in range(len(dataset_scheduling)):
    if i + 4 < len(dataset_scheduling):
        sequences[i].append(dataset_scheduling[i, :-3])
        sequences[i].append(dataset_scheduling[i + 1, :-3])
        sequences[i].append(dataset_scheduling[i + 2, :-3])
        sequences[i].append(dataset_scheduling[i + 3, :-3])
        sequences[i].append(dataset_scheduling[i + 4, :-3])

        labels[i].append(dataset_scheduling[i, -3:])
        labels[i].append(dataset_scheduling[i + 1, -3:])
        labels[i].append(dataset_scheduling[i + 2, -3:])
        labels[i].append(dataset_scheduling[i + 3, -3:])
        labels[i].append(dataset_scheduling[i + 4, -3:])

sequences = np.array(sequences)
labels = np.array(labels)

indices = np.random.permutation(len(sequences))

sequences = sequences[indices]
labels = labels[indices]

batch_size = 1000
dataset = tf.data.Dataset.from_tensor_slices((sequences, labels)).batch(batch_size)

epochs = 10
sum_losses = 0.0
for epoch in tqdm(range(epochs)):
    for x_batch, y_batch in tqdm(dataset):
        sum_losses += train_step(x_batch, y_batch, model, input_shape, input_timesteps, dt)
    print(f'Epoch {epoch + 1}, Loss: {sum_losses.numpy() / len(dataset)}')

tf.keras.models.save_model(model, 'saved_models/step_1/callbacks/2024_07_18/13_02_33/keras_scheduled.h5')
tf.keras.models.save_model(model, 'saved_models/step_1/callbacks/2024_07_18/13_02_33/keras_scheduled.keras')