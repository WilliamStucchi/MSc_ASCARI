# Nathan Spielberg
# DDL 10.17.2018

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from data_preprocessing.parameters.learning_params import *
from models import *
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

print('Num GPUs Available: ', tf.config.list_physical_devices('GPU'))

# remove later
# import matplotlib.pyplot as plt

# set the random seed for repeatable results
np.random.seed(1)
tf.set_random_seed(1)

# need to change iterator and retrain for all make two iterators
# for each model.
# probably some work to verify and then retrain still have old results
# saved


##########################################################################
# Experiment 1
##########################################################################

# first Load in the data
data = np.load("data/gen/exp_1_w.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])

"""
# w: print all elements passed to the network in one input of the training

count = 0
correspondence = 0

for element in train_data[0]:
    for count in range(0, 5):
        print('e' + str(count + 1) + ': ' + str(element[count]) + ' ' + str(element[count + 5]) + ' ' + str(
            element[count + 10]) + ' ' + str(element[count + 15]))
    print('Output: ' + str(train_data[1][correspondence][0]) + ', ' + str(train_data[1][correspondence][1]))
    correspondence += 1

    variable = input()
"""

dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

tf.disable_eager_execution()

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_1 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_1 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_1/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_1.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_1.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_1.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_1.mse(), NN_1.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_1.mse(), NN_1.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_1.optimize(), NN_1.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_1.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_1.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_1.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_1.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_1.mse(), NN_1.mse()])

        # write out Experimental Results
        np.savez("results/gen_test_2_w/exp_1",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/gen_test_2_w/exp_1/model.ckpt")
        print("Model for Exp 1 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()

##########################################################################
# Experiment 2
##########################################################################


# first Load in the data
data = np.load("data/gen/exp_2_w.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_2 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_2 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_2/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_2.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_2.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_2.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_2.mse(), NN_2.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_2.mse(), NN_2.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_2.optimize(), NN_2.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_2.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_2.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_2.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_2.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_2.mse(), NN_2.mse()])

        # write out Experimental Results
        np.savez("results/gen_test_2_w/exp_2",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/gen_test_2_w/exp_2/model.ckpt")
        print("Model for Exp 2 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()

##########################################################################
# Experiment 3
##########################################################################

# first Load in the data
data = np.load("data/gen/exp_3_w.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_3 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_3 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_3/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_3.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_3.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_3.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_3.mse(), NN_3.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_3.mse(), NN_3.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_3.optimize(), NN_3.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_3.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_3.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_3.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_3.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_3.mse(), NN_3.mse()])

        # write out Experimental Results
        np.savez("results/gen_test_2_w/exp_3",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/gen_test_2_w/exp_3/model.ckpt")
        print("Model for Exp 3 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()

##########################################################################
# Experiment 5
##########################################################################

# first Load in the data
data = np.load("data/gen/exp_5_w.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_5 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_5 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_5/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_5.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_5.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_5.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_5.mse(), NN_5.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_5.mse(), NN_5.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_5.optimize(), NN_5.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_5.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_5.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_5.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_5.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_5.mse(), NN_5.mse()])

        # write out Experimental Results
        np.savez("results/gen_test_2_w/exp_5",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/gen_test_2_w/exp_5/model.ckpt")
        print("Model for Exp 5 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()
##########################################################################
# Experiment 6
##########################################################################


# first Load in the data
data = np.load("data/gen/exp_6_w.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_6 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_6 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_6/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_6.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_6.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_6.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_6.mse(), NN_6.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_6.mse(), NN_6.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_6.optimize(), NN_6.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_6.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_6.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_6.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_6.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_6.mse(), NN_6.mse()])

        # write out Experimental Results
        np.savez("results/gen_test_2_w/exp_6",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/gen_test_2_w/exp_6/model.ckpt")
        print("Model for Exp 6 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()

##########################################################################
# Begin Experiments on Experimental Data!
##########################################################################

"""
##########################################################################
#Experiment 7 Ice Experimental Data
##########################################################################

#first Load in the data
data = np.load("data/exp/exp_data/ice.npz")

#Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data   = (data["dev_f"],   data["dev_t"])
test_data  = (data["test_f"],  data["test_t"])


#Define Number of batches per epoch
n_batches  = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"],2))
dev_cost   = np.zeros(shape=(Param["EPOCHS"],2))


#Create necessary placeholders for the data.
x, y           = tf.placeholder(tf.float64, shape=[None,Param["N_FEATURES"] ]), tf.placeholder(tf.float64, shape=[None,Param["N_TARGETS"] ])

#Create a placeholder to dyn switch between batch sizes for test and train...
batch_size     = tf.placeholder(tf.int64)


bike_dataset   = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter      = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset        = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter           = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels    = nn_iter.get_next()

#Create Models first
bike_7         = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_7           = NN_Model(Param, nn_inputs, nn_labels)

#Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_7/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_7.Ca_f))  )
        print("The Initial Cr Values is: " + str(sess.run(bike_7.Ca_r))  )
        print("The Initial mu Values is: " + str(sess.run(bike_7.mu_tf)) )

        for i in tqdm(range( Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_7.mse(), NN_7.mse()] )
            sess.run(bike_iter.initializer, feed_dict={ x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={ x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_7.mse(), NN_7.mse()] )

            sess.run(bike_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_7.optimize(), NN_7.optimize()] )


        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_7.Ca_f)) + " And the Actual is: " + str( Veh["Cf"])  )
        print("The Final Cr Values is: " + str(sess.run(bike_7.Ca_r)) + " And the Actual is: " + str( Veh["Cr"])  )
        print("The Final mu Values is: " + str(sess.run(bike_7.mu_tf))+ " And the Actual is: " + str( Veh["mu"])  )
        print("The Final Izz Values is: " + str(sess.run(bike_7.I_tf))+ " And the Actual is: " + str( Veh["Izz"]) )

        #test
        sess.run(bike_iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_7.mse(), NN_7.mse()])

        #write out Experimental Results
        np.savez("results/exp_test_w/exp_7",
            test_mse     = (test_mse_b, test_mse_nn),
            delay_states = Param["T_MODEL"],
            train_cost   = train_cost,
            dev_cost     = dev_cost)

        #And save the session!
        saver.save(sess, "saved_models/exp_test_w/exp_7/model.ckpt")
        print("Model for Exp 7 Saved")

        #Close and reset the session
        #tf.reset_default_graph()
        sess.close()

"""
##########################################################################
# Experiment 8 Dry Experimental Data
##########################################################################
# first Load in the data
data = np.load("data/exp/exp_data/dry.npz")

# Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data = (data["dev_f"], data["dev_t"])
test_data = (data["test_f"], data["test_t"])

# Define Number of batches per epoch
n_batches = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"], 2))
dev_cost = np.zeros(shape=(Param["EPOCHS"], 2))

tf.disable_eager_execution()

# Create necessary placeholders for the data.
x, y = tf.placeholder(tf.float64, shape=[None, Param["N_FEATURES"]]), tf.placeholder(tf.float64,
                                                                                     shape=[None, Param["N_TARGETS"]])

# Create a placeholder to dyn switch between batch sizes for test and train...
batch_size = tf.placeholder(tf.int64)

bike_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels = nn_iter.get_next()

# Create Models first
bike_8 = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_8 = NN_Model(Param, nn_inputs, nn_labels)

# Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_8/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_8.Ca_f)))
        print("The Initial Cr Values is: " + str(sess.run(bike_8.Ca_r)))
        print("The Initial mu Values is: " + str(sess.run(bike_8.mu_tf)))

        for i in tqdm(range(Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_8.mse(), NN_8.mse()])
            sess.run(bike_iter.initializer,
                     feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_8.mse(), NN_8.mse()])

            sess.run(bike_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer,
                     feed_dict={x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_8.optimize(), NN_8.optimize()])

        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_8.Ca_f)) + " And the Actual is: " + str(Veh["Cf"]))
        print("The Final Cr Values is: " + str(sess.run(bike_8.Ca_r)) + " And the Actual is: " + str(Veh["Cr"]))
        print("The Final mu Values is: " + str(sess.run(bike_8.mu_tf)) + " And the Actual is: " + str(Veh["mu"]))
        print("The Final Izz Values is: " + str(sess.run(bike_8.I_tf)) + " And the Actual is: " + str(Veh["Izz"]))

        # test
        sess.run(bike_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_8.mse(), NN_8.mse()])

        # write out Experimental Results
        np.savez("results/exp_test_2_w/exp_8",
                 test_mse=(test_mse_b, test_mse_nn),
                 delay_states=Param["T_MODEL"],
                 train_cost=train_cost,
                 dev_cost=dev_cost)

        # And save the session!
        saver.save(sess, "saved_models/exp_test_2_w/exp_8/model.ckpt")
        print("Model for Exp 8 Saved")

        # Close and reset the session
        # tf.reset_default_graph()
        sess.close()

"""
##########################################################################
# Experiment 9 Combined Experimental Data
##########################################################################
#first Load in the data
data   = np.load("data/exp/exp_data/combined.npz")

data_1 = np.load("data/exp/exp_data/combined_unshuffled.npz")


#Output the Data From the Experiment
train_data = (data["train_f"], data["train_t"])
dev_data   = (data["dev_f"],   data["dev_t"])
test_data  = (data["test_f"],  data["test_t"])
print("The Training data is: ", len(train_data[0]) )


test_data_1= (data_1["test_f"],  data_1["test_t"])


#Define Number of batches per epoch
n_batches  = len(train_data[0]) // Param["BATCH_SIZE"] + 1
train_cost = np.zeros(shape=(Param["EPOCHS"],2))
dev_cost   = np.zeros(shape=(Param["EPOCHS"],2))
print("The Number of Batches are: ", n_batches)


#Create necessary placeholders for the data.
x, y           = tf.placeholder(tf.float64, shape=[None,Param["N_FEATURES"] ]), tf.placeholder(tf.float64, shape=[None,Param["N_TARGETS"] ])

#Create a placeholder to dyn switch between batch sizes for test and train...
batch_size     = tf.placeholder(tf.int64)
init_state     = tf.placeholder(tf.float64, [None, Param["HIDDEN"] ])

bike_dataset   = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
bike_iter      = bike_dataset.make_initializable_iterator()
bike_inputs, bike_labels = bike_iter.get_next()

nn_dataset        = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).repeat()
nn_iter           = nn_dataset.make_initializable_iterator()
nn_inputs, nn_labels    = nn_iter.get_next()

#Create Models first
bike_9         = Bike_Model(Param, Veh, bike_inputs, bike_labels)

NN_9           = NN_Model(Param, nn_inputs, nn_labels)

#Add ops to save and restore all the variables
saver = tf.train.Saver()

with tf.device('/GPU:0'):
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if Param["RESTORE"]:
            saver.restore(sess, "saved_models/exp_9/model.ckpt")
            print("Model restored from saved model! ")

        print("The Initial Cf Values is: " + str(sess.run(bike_9.Ca_f))  )
        print("The Initial Cr Values is: " + str(sess.run(bike_9.Ca_r))  )
        print("The Initial mu Values is: " + str(sess.run(bike_9.mu_tf)) )

        for i in tqdm(range( Param["EPOCHS"])):
            # initialize iterator with train data
            sess.run(bike_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: train_data[0].shape[0]})
            train_cost[i] = sess.run([bike_9.mse(), NN_9.mse()] )
            sess.run(bike_iter.initializer, feed_dict={ x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            sess.run(nn_iter.initializer, feed_dict={ x: dev_data[0], y: dev_data[1], batch_size: dev_data[0].shape[0]})
            dev_cost[i] = sess.run([bike_9.mse(), NN_9.mse()] )

            sess.run(bike_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            sess.run(nn_iter.initializer, feed_dict={ x: train_data[0], y: train_data[1], batch_size: Param["BATCH_SIZE"]})
            for _ in range(n_batches):
                _ = sess.run([bike_9.optimize(), NN_9.optimize()] )


        print("The Oracle has spoken!")
        print("The Final Cf Values is: " + str(sess.run(bike_9.Ca_f)) + " And the Actual is: " + str( Veh["Cf"])  )
        print("The Final Cr Values is: " + str(sess.run(bike_9.Ca_r)) + " And the Actual is: " + str( Veh["Cr"])  )
        print("The Final mu Values is: " + str(sess.run(bike_9.mu_tf))+ " And the Actual is: " + str( Veh["mu"])  )
        print("The Final Izz Values is: " + str(sess.run(bike_9.I_tf))+ " And the Actual is: " + str( Veh["Izz"]) )

        #test
        sess.run(bike_iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        sess.run(nn_iter.initializer, feed_dict={ x: test_data[0], y: test_data[1], batch_size: test_data[0].shape[0]})
        test_mse_b, test_mse_nn = sess.run([bike_9.mse(), NN_9.mse()])


        #label_true      = sess.run(labels)
        #write out Experimental Results

        np.savez("results/exp_test_w/exp_9",
            test_mse     = (test_mse_b, test_mse_nn),
            delay_states = Param["T_MODEL"],
            train_cost   = train_cost,
            dev_cost     = dev_cost)

        #And save the session!
        saver.save(sess, "saved_models/exp_test_w/exp_9/model.ckpt")
        print("Model for Exp 9 Saved")

        #Close and reset the session
        sess.close()
        
"""
