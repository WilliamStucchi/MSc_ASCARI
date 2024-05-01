from models_V2 import *
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os.path
import datetime

CALLBACKS = False
path_trained_model = 'saved_models/test_CRT/'
# create datetime-dependent paths
path_day = datetime.datetime.now().strftime('%Y_%m_%d')
path_datetime = datetime.datetime.now().strftime('%H_%M_%S')

mod = NN_model_V2()
model = mod.build_model(seed=1)

# GET TRAINING DATA
perc_validation = 0.2

dataset_step_1 = np.loadtxt('data/CRT/train_data_step1.csv')
dataset_step_2 = np.loadtxt('data/CRT/train_data_step2.csv')
dataset_step_3 = np.loadtxt('data/CRT/train_data_step3.csv')
dataset_step_4 = np.loadtxt('data/CRT/train_data_step4.csv')

# 1 step
train_features_step_1 = dataset_step_1[:len(dataset_step_1) * (1 - perc_validation), :-3]
val_features_step_1 = dataset_step_1[len(dataset_step_1) * (1 - perc_validation):, :-3]
train_labels_step_1 = dataset_step_1[:len(dataset_step_1) * (1 - perc_validation), -3:]
val_labels_step_1 = dataset_step_1[len(dataset_step_1) * (1 - perc_validation):, -3:]

# 2 step
train_features_step_2 = dataset_step_2[:len(dataset_step_2) * (1 - perc_validation), :-3]
val_features_step_2 = dataset_step_2[len(dataset_step_2) * (1 - perc_validation):, :-3]
train_labels_step_2 = dataset_step_2[:len(dataset_step_2) * (1 - perc_validation), -3:]
val_labels_step_2 = dataset_step_2[len(dataset_step_2) * (1 - perc_validation):, -3:]

# 3 step
train_features_step_3 = dataset_step_3[:len(dataset_step_3) * (1 - perc_validation), :-3]
val_features_step_3 = dataset_step_3[len(dataset_step_3) * (1 - perc_validation):, :-3]
train_labels_step_3 = dataset_step_3[:len(dataset_step_3) * (1 - perc_validation), -3:]
val_labels_step_3 = dataset_step_3[len(dataset_step_3) * (1 - perc_validation):, -3:]

# 4 step
train_features_step_4 = dataset_step_4[:len(dataset_step_4) * (1 - perc_validation), :-3]
val_features_step_4 = dataset_step_4[len(dataset_step_4) * (1 - perc_validation):, :-3]
train_labels_step_4 = dataset_step_4[:len(dataset_step_4) * (1 - perc_validation), -3:]
val_labels_step_4 = dataset_step_4[len(dataset_step_4) * (1 - perc_validation):, -3:]

# Step 1 training
if CALLBACKS:
    es = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                          mode='min',
                                          verbose=1,
                                          patience=60)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                                          factor=0.8,
                                                          patience=10,
                                                          verbose=1,
                                                          mode='min',
                                                          min_delta=0.000005)

    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_1/callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)

    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_1,
                                y=train_labels_step_1,
                                batch_size=1000,
                                validation_data=(val_features_step_1, val_labels_step_1),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[reduce_lr_loss, es, mc],
                                use_multiprocessing=True)
else:
    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_1/no_callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)
    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_1,
                                y=train_labels_step_1,
                                batch_size=1000,
                                validation_data=(val_features_step_1, val_labels_step_1),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[mc],
                                use_multiprocessing=True)

print(history_mod.history.keys())

# Step 2 training
if CALLBACKS:
    es = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                          mode='min',
                                          verbose=1,
                                          patience=60)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                                          factor=0.8,
                                                          patience=10,
                                                          verbose=1,
                                                          mode='min',
                                                          min_delta=0.000005)

    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_2/callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)

    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_2,
                                y=train_labels_step_2,
                                batch_size=1000,
                                validation_data=(val_features_step_2, val_labels_step_2),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[reduce_lr_loss, es, mc],
                                use_multiprocessing=True)
else:
    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_2/no_callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)
    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_2,
                                y=train_labels_step_2,
                                batch_size=1000,
                                validation_data=(val_features_step_2, val_labels_step_2),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[mc],
                                use_multiprocessing=True)

print(history_mod.history.keys())

# Step 3 training
if CALLBACKS:
    es = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                          mode='min',
                                          verbose=1,
                                          patience=60)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                                          factor=0.8,
                                                          patience=10,
                                                          verbose=1,
                                                          mode='min',
                                                          min_delta=0.000005)

    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_3/callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)

    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_3,
                                y=train_labels_step_3,
                                batch_size=1000,
                                validation_data=(val_features_step_3, val_labels_step_3),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[reduce_lr_loss, es, mc],
                                use_multiprocessing=True)
else:
    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_3/no_callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)
    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_3,
                                y=train_labels_step_3,
                                batch_size=1000,
                                validation_data=(val_features_step_3, val_labels_step_3),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[mc],
                                use_multiprocessing=True)

print(history_mod.history.keys())

# Step 4 training
if CALLBACKS:
    es = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
                                          mode='min',
                                          verbose=1,
                                          patience=60)

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='mean_squared_error',
                                                          factor=0.8,
                                                          patience=10,
                                                          verbose=1,
                                                          mode='min',
                                                          min_delta=0.000005)

    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_4/callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)

    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_4,
                                y=train_labels_step_4,
                                batch_size=1000,
                                validation_data=(val_features_step_4, val_labels_step_4),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[reduce_lr_loss, es, mc],
                                use_multiprocessing=True)
else:
    mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/step_4/no_callbacks/' + path_day + '/' + path_datetime,
                                            monitor='mean_squared_error',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True)
    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_features_step_1,
                                y=train_labels_step_1,
                                batch_size=1000,
                                validation_data=(val_features_step_1, val_labels_step_1),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[mc],
                                use_multiprocessing=True)

print(history_mod.history.keys())
