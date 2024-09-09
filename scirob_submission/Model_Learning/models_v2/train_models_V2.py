from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scirob_submission.Model_Learning.models_v2.models_V2 import *
import tensorflow as tf
import numpy as np
import datetime
from joblib import dump
import math
from pathlib import Path


def apply_scaler(features, labels, n_samples, n_timesteps):
    features_reshaped = features.reshape(-1, n_timesteps)

    # Initialize scalers for inputs
    input_scalers = {
        "longitudinal_velocity": MinMaxScaler(feature_range=(0, 1)),
        "lateral_velocity": MinMaxScaler(feature_range=(0, 1)),
        "yaw_rate": StandardScaler(),
        "steering_angle": StandardScaler(),
        "longitudinal_force": StandardScaler()
    }

    # Create a list of arrays for scaled inputs
    scaled_inputs_list = []

    # Scale each feature independently
    for i in range(n_timesteps):  # 4 timesteps
        start_idx = i * 5
        end_idx = start_idx + 5
        scaled_inputs_timestep = np.zeros_like(features_reshaped[:, start_idx:end_idx])

        scaled_inputs_timestep[:, 0] = input_scalers["longitudinal_velocity"].fit_transform(
            features_reshaped[:, start_idx].reshape(-1, 1)).flatten()
        scaled_inputs_timestep[:, 1] = input_scalers["lateral_velocity"].fit_transform(
            features_reshaped[:, start_idx + 1].reshape(-1, 1)).flatten()
        scaled_inputs_timestep[:, 2] = input_scalers["yaw_rate"].fit_transform(
            features_reshaped[:, start_idx + 2].reshape(-1, 1)).flatten()
        scaled_inputs_timestep[:, 3] = input_scalers["steering_angle"].fit_transform(
            features_reshaped[:, start_idx + 3].reshape(-1, 1)).flatten()
        scaled_inputs_timestep[:, 4] = input_scalers["longitudinal_force"].fit_transform(
            features_reshaped[:, start_idx + 4].reshape(-1, 1)).flatten()

        scaled_inputs_list.append(scaled_inputs_timestep)

    # Concatenate the scaled inputs
    scaled_inputs = np.concatenate(scaled_inputs_list, axis=1)

    # Reshape back to the original input shape
    scaled_inputs = scaled_inputs.reshape(n_samples, 20)

    print('Features: ', features.shape)
    print('Scaled inputs: ', scaled_inputs.shape)
    input('Wait...')
    """scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    val_features = scaler.transform(val_features)"""


# ----------------------------------------------------------------------------------------------------------------------

def get_set(dataset, perc_valid):
    train_features, train_lab = get_training(dataset, perc_valid)
    assert_creation(train_features, train_lab)
    val_features, val_lab = get_validation(dataset, perc_valid)
    assert_creation(val_features, val_lab)

    scaler = None
    if APPLY_SCALER:
        n_samples = len(train_features)
        n_timesteps = 4
        apply_scaler(train_features, train_lab, n_samples, n_timesteps)

    return train_features, train_lab, val_features, val_lab, scaler


# ----------------------------------------------------------------------------------------------------------------------

def get_training(dataset, perc_valid):
    features = dataset[:math.floor(len(dataset) * (1 - perc_valid)), :-4]
    labels = dataset[:math.floor(len(dataset) * (1 - perc_valid)), -4:]
    return features, labels


# ----------------------------------------------------------------------------------------------------------------------

def get_validation(dataset, perc_valid):
    features = dataset[math.floor(len(dataset) * (1 - perc_valid)):, :-4]
    labels = dataset[math.floor(len(dataset) * (1 - perc_valid)):, -4:]
    return features, labels


# ----------------------------------------------------------------------------------------------------------------------

def assert_creation(features, labels):
    assert len(features) != 0
    assert len(labels) != 0
    assert len(features) == len(labels)


# ----------------------------------------------------------------------------------------------------------------------

def compute_sample_weights(acc, th=0, base_weight=1.0, scale_factor=2.0):
    """
    Compute custom weights for samples based on lateral acceleration.

    Parameters:
    - acc: numpy array of lateral acceleration values.
    - th: The threshold for lateral acceleration.
    - base_weight: The base weight for samples below the threshold.
    - scale_factor: The factor by which weights increase for samples above the threshold.

    Returns:
    - numpy array of sample weights.
    """
    # Initialize weights with the base weight
    weights = np.full_like(acc, base_weight, dtype=float)

    # Set threshold
    th = max(acc) * 0.8
    print('MAX: ', max(acc))
    print('TH: ', th)

    # Scale weights for samples with lateral acceleration greater than the threshold
    above_threshold = (np.abs(acc) >= th)
    weights[above_threshold] = base_weight + scale_factor * (np.abs(acc[above_threshold]) - th)
    # weights[above_threshold] = base_weight + np.exp(scale_factor * (abs(acc[above_threshold]) - th))

    # Normalize the weights to balance the dataset
    weights_sum = np.sum(weights)
    weights_normalized = weights / weights_sum * len(weights)

    print('Total length of training set: ', len(weights))
    print('Total number of elements above ' + str(th) + ' m/s2: ', np.sum(above_threshold))

    return weights_normalized


# -----------------------------------------------------------------------------------------------------------------------

APPLY_SCALER = False
TRAIN_S1 = True
# create datetime-dependent paths
path_day = datetime.datetime.now().strftime('%Y_%m_%d')
path_datetime = datetime.datetime.now().strftime('%H_%M_%S')
save_path_initial = '../saved_models/'

# GET TRAINING DATA
dataset_step_1 = np.loadtxt('../data/new/train_data_step_1_gripest.csv', delimiter=',')
print('TOTAL LENGTH OF THE DATASET: ', len(dataset_step_1))

perc_validation = 0.2

# 1 step
train_features_step_1, train_labels_step_1, val_features_step_1, val_labels_step_1, scaler_1 = get_set(dataset_step_1,
                                                                                                       perc_validation)
"""
# custom weight
lateral_accelerations = train_labels_step_1[:, 1]
threshold = 5.0
scale_factor = 2.0

# Compute sample weights
sample_weights = compute_sample_weights(lateral_accelerations, threshold, scale_factor=scale_factor)
"""

"""
set1 = np.loadtxt('../data/single_sets/train_data_step_1_mu_0.csv', delimiter=',')
set2 = np.loadtxt('../data/single_sets/train_data_step_1_mu_1.csv', delimiter=',')
set3 = np.loadtxt('../data/single_sets/train_data_step_1_mu_2.csv', delimiter=',')

perc_validation = 0.2

# 1 step
train_features_set1, train_labels_set1, val_features_set1, val_labels_set1, _ = get_set(set1, perc_validation)
train_features_set2, train_labels_set2, val_features_set2, val_labels_set2, _ = get_set(set2, perc_validation)
train_features_set3, train_labels_set3, val_features_set3, val_labels_set3, _ = get_set(set3, perc_validation)

# custom weight
ay_set1 = train_labels_set1[:, 1]
ay_set2 = train_labels_set2[:, 1]
ay_set3 = train_labels_set3[:, 1]
scale_factor = 2.0

# Compute sample weights
w_set1 = compute_sample_weights(ay_set1, scale_factor=scale_factor)
w_set2 = compute_sample_weights(ay_set2, scale_factor=scale_factor)
w_set3 = compute_sample_weights(ay_set3, scale_factor=scale_factor)
input('wait')

train_features_step1 = np.concatenate((train_features_set1, train_features_set2, train_features_set3))
train_labels_step1 = np.concatenate((train_labels_set1, train_labels_set2, train_labels_set3))
val_features_step1 = np.concatenate((val_features_set1, val_features_set2, val_features_set3))
val_labels_step1 = np.concatenate((val_labels_set1, val_labels_set2, val_labels_set3))
sample_weights = np.concatenate((w_set1, w_set2, w_set3))

# random shuffle
indices = np.random.permutation(len(train_features_step1))

train_features_step1 = train_features_step1[indices]
train_labels_step1 = train_labels_step1[indices]
sample_weights = sample_weights[indices]

indices = np.random.permutation(len(val_features_step1))

val_features_step1 = val_features_step1[indices]
val_labels_step1 = val_labels_step1[indices]
"""

"""print('Funziona?')
for i in range(11):
    print('input -> {')
    print(train_features_step_1[i])
    print(train_labels_step_1[i])
    print(sample_weights[i])
    print('}')
input('wait')"""

"""# change input shape for lstm layer
train_features_step_1 = np.reshape(train_features_step_1, (train_labels_step_1.shape[0], 4, 4))
val_features_step_1 = np.reshape(val_features_step_1, (val_labels_step_1.shape[0], 4, 4))"""

# Step 1 training
if TRAIN_S1:
    mod = NN_Model_V2()
    model_step_1 = mod.build_model(seed=1)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_mse',
                                          mode='min',
                                          verbose=1,
                                          patience=60)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse',
                                                          factor=0.8,
                                                          patience=10,
                                                          verbose=1,
                                                          mode='min',
                                                          min_delta=0.000005)

    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path_initial + 'step_1/callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
        monitor='val_mse',
        mode='min',
        verbose=1,
        save_best_only=True)
    # os.mkdir('results/step_1/callbacks/' + path_day + '/' + path_datetime + '/images')

    Path(save_path_initial + 'step_1/callbacks/' + path_day + '/' + path_datetime + "/").mkdir(parents=True,
                                                                                               exist_ok=True)
    """if APPLY_SCALER:
        scaler_filename = save_path_initial + 'step_1/callbacks/' + path_day + '/' + path_datetime + '/scaler.plk'
        dump(scaler_1, scaler_filename)"""

    with tf.device('/GPU:0'):
        history_mod = model_step_1.fit(x=train_features_step_1,
                                       y=train_labels_step_1,
                                       # sample_weight=sample_weights,
                                       batch_size=1000,
                                       validation_data=(val_features_step_1, val_labels_step_1),
                                       epochs=2000,
                                       verbose=1,
                                       shuffle=False,
                                       callbacks=[reduce_lr_loss, es, mc],
                                       use_multiprocessing=True)
