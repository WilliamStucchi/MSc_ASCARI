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


# CREATE CALLBACKS
mc = tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/'+path_day+'/'+path_datetime,
                         monitor='mean_squared_error',
                         mode='min',
                         verbose=1,
                         save_best_only=True)

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

    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_data[0],
                                y=train_data[1],
                                batch_size=1000,
                                validation_data=(val_data[0], val_data[1]),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[reduce_lr_loss, es, mc],
                                use_multiprocessing=True)
else:
    with tf.device('/GPU:0'):
        history_mod = model.fit(x=train_data[0],
                                y=train_data[1],
                                batch_size=1000,
                                validation_data=(val_data[0], val_data[1]),
                                epochs=1500,
                                verbose=1,
                                shuffle=True,
                                callbacks=[mc],
                                use_multiprocessing=True)

print(history_mod.history.keys())
