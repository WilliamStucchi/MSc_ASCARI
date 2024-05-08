import os

from scirob_submission.Model_Learning.models_v2.models_V2 import *
import tensorflow as tf
import numpy as np
import math
import datetime

CALLBACKS = True
# create datetime-dependent paths
path_day = datetime.datetime.now().strftime('%Y_%m_%d')
path_datetime = datetime.datetime.now().strftime('%H_%M_%S')
save_path_initial = '..saved_models/'

mod = NN_Model_V2()
model_step_1 = mod.build_model(seed=1)
model_step_2 = mod.build_model(seed=1)
model_step_3 = mod.build_model(seed=1)
model_step_4 = mod.build_model(seed=1)

# GET TRAINING DATA
perc_validation = 0.2

dataset_step_1 = np.loadtxt('../data/CRT/old/train_data_step1.csv', delimiter=',')
dataset_step_2 = np.loadtxt('../data/CRT/old/train_data_step2.csv', delimiter=',')
dataset_step_3 = np.loadtxt('../data/CRT/old/train_data_step3.csv', delimiter=',')
dataset_step_4 = np.loadtxt('../data/CRT/old/train_data_step4.csv', delimiter=',')

# 1 step
train_features_step_1 = dataset_step_1[:math.floor(len(dataset_step_1) * (1 - perc_validation)), :-3]
val_features_step_1 = dataset_step_1[math.floor(len(dataset_step_1) * (1 - perc_validation)):, :-3]
train_labels_step_1 = dataset_step_1[:math.floor(len(dataset_step_1) * (1 - perc_validation)), -3:]
val_labels_step_1 = dataset_step_1[math.floor(len(dataset_step_1) * (1 - perc_validation)):, -3:]

# 2 step
train_features_step_2 = dataset_step_2[:math.floor(len(dataset_step_2) * (1 - perc_validation)), :-3]
val_features_step_2 = dataset_step_2[math.floor(len(dataset_step_2) * (1 - perc_validation)):, :-3]
train_labels_step_2 = dataset_step_2[:math.floor(len(dataset_step_2) * (1 - perc_validation)), -3:]
val_labels_step_2 = dataset_step_2[math.floor(len(dataset_step_2) * (1 - perc_validation)):, -3:]

# 3 step
train_features_step_3 = dataset_step_3[:math.floor(len(dataset_step_3) * (1 - perc_validation)), :-3]
val_features_step_3 = dataset_step_3[math.floor(len(dataset_step_3) * (1 - perc_validation)):, :-3]
train_labels_step_3 = dataset_step_3[:math.floor(len(dataset_step_3) * (1 - perc_validation)), -3:]
val_labels_step_3 = dataset_step_3[math.floor(len(dataset_step_3) * (1 - perc_validation)):, -3:]

# 4 step
train_features_step_4 = dataset_step_4[:math.floor(len(dataset_step_4) * (1 - perc_validation)), :-3]
val_features_step_4 = dataset_step_4[math.floor(len(dataset_step_4) * (1 - perc_validation)):, :-3]
train_labels_step_4 = dataset_step_4[:math.floor(len(dataset_step_4) * (1 - perc_validation)), -3:]
val_labels_step_4 = dataset_step_4[math.floor(len(dataset_step_4) * (1 - perc_validation)):, -3:]

for i in range(0, 1):
    # Step 1 training
    if CALLBACKS:
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
            filepath=save_path_initial +'/step_1/callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_1/callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_1.fit(x=train_features_step_1,
                                           y=train_labels_step_1,
                                           batch_size=1000,
                                           validation_data=(val_features_step_1, val_labels_step_1),
                                           epochs=1500,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[reduce_lr_loss, es, mc],
                                           use_multiprocessing=True)
    else:
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path_initial +'/step_1/no_callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_1/no_callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_1.fit(x=train_features_step_1,
                                           y=train_labels_step_1,
                                           batch_size=1000,
                                           validation_data=(val_features_step_1, val_labels_step_1),
                                           epochs=1000,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[mc],
                                           use_multiprocessing=True)

    # Step 2 training
    if CALLBACKS:
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
            filepath=save_path_initial +'/step_2/callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_2/callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_2.fit(x=train_features_step_2,
                                           y=train_labels_step_2,
                                           batch_size=1000,
                                           validation_data=(val_features_step_2, val_labels_step_2),
                                           epochs=1500,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[reduce_lr_loss, es, mc],
                                           use_multiprocessing=True)
    else:
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path_initial +'/step_2/no_callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_2/no_callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_2.fit(x=train_features_step_2,
                                           y=train_labels_step_2,
                                           batch_size=1000,
                                           validation_data=(val_features_step_2, val_labels_step_2),
                                           epochs=1000,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[mc],
                                           use_multiprocessing=True)

    # Step 3 training
    if CALLBACKS:
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
            filepath=save_path_initial +'/step_3/callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_3/callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_3.fit(x=train_features_step_3,
                                           y=train_labels_step_3,
                                           batch_size=1000,
                                           validation_data=(val_features_step_3, val_labels_step_3),
                                           epochs=1500,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[reduce_lr_loss, es, mc],
                                           use_multiprocessing=True)
    else:
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path_initial +'/step_3/no_callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_3/no_callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_3.fit(x=train_features_step_3,
                                           y=train_labels_step_3,
                                           batch_size=1000,
                                           validation_data=(val_features_step_3, val_labels_step_3),
                                           epochs=1000,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[mc],
                                           use_multiprocessing=True)

    # Step 4 training
    if CALLBACKS:
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
            filepath=save_path_initial +'/step_4/callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_4/callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_4.fit(x=train_features_step_4,
                                           y=train_labels_step_4,
                                           batch_size=1000,
                                           validation_data=(val_features_step_4, val_labels_step_4),
                                           epochs=1500,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[reduce_lr_loss, es, mc],
                                           use_multiprocessing=True)
    else:
        mc = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path_initial +'/step_4/no_callbacks/' + path_day + '/' + path_datetime + "/keras_model.h5",
            monitor='val_mse',
            mode='min',
            verbose=1,
            save_best_only=True)
        # os.mkdir('results/step_4/no_callbacks/' + path_day + '/' + path_datetime + '/images')

        with tf.device('/GPU:0'):
            history_mod = model_step_4.fit(x=train_features_step_1,
                                           y=train_labels_step_1,
                                           batch_size=1000,
                                           validation_data=(val_features_step_1, val_labels_step_1),
                                           epochs=1000,
                                           verbose=1,
                                           shuffle=True,
                                           callbacks=[mc],
                                           use_multiprocessing=True)

    CALLBACKS = False
