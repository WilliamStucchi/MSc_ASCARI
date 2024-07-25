from scirob_submission.Model_Learning.models_v2.models_V2 import *
import math
from tqdm import tqdm
import random


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

def train_step(x, y, model, input_shape_, input_timesteps_, dt, loss_fn_, optimizer_, epsilon_):
    cumulative_loss = 0.0
    bs = tf.shape(x)[0]
    evaluation = random.choices([0, 1], weights=[1 - epsilon_, epsilon_], k=bs)

    yaw_rate = np.zeros(bs)
    vy = np.zeros(bs)
    vx = np.zeros(bs)

    mod_x = np.array(x)

    with ((tf.device('/GPU:0'))):
        for k in range(tf.shape(x)[1]):
            with tf.GradientTape() as tape:
                if k != 0:
                    for b in range(bs):
                        if evaluation[b] == 1:
                            mod_x[b, k, :input_shape_ * (input_timesteps_ - 1)] = mod_x[b, k - 1, input_shape_:]

                            mod_x[b, k,
                                  input_shape_ * (input_timesteps_ - 1):input_shape_ * (input_timesteps_ - 1) + 1] = yaw_rate[b]
                            mod_x[b, k,
                                  input_shape_ * (input_timesteps_ - 1) + 1:input_shape_ * (input_timesteps_ - 1) + 2] = vy[b]
                            mod_x[b, k,
                                  input_shape_ * (input_timesteps_ - 1) + 2:input_shape_ * (input_timesteps_ - 1) + 3] = vx[b]

                new_x = tf.convert_to_tensor(mod_x)
                inp = new_x[:, k]

                prediction = model(inp, training=False)

                for j in range(bs):
                    corresponding_input = np.array(new_x[j:j + 1, k])
                    bs_prediction = prediction[j:j + 1]

                    yaw_acc = float(bs_prediction[:, 0])
                    ay = float(bs_prediction[:, 1])
                    ax = float(bs_prediction[:, 2])

                    # Euler integration
                    yaw_rate[j] = yaw_acc * dt + corresponding_input[:, input_shape_ * (input_timesteps_ - 1):
                                                                     input_shape_ * (input_timesteps_ - 1) + 1].item()

                    # vy_dot = ay - yaw_rate * vx
                    dvy = ay - yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                 input_shape_ * (input_timesteps_ - 1) + 3]

                    # vx_dot = ax + yaw_rate * vy
                    dvx = ax + yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                 input_shape_ * (input_timesteps_ - 1) + 2]

                    # Euler integration
                    vy[j] = dvy.item() * dt + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                  input_shape_ * (input_timesteps_ - 1) + 2].item()
                    vx[j] = dvx.item() * dt + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                  input_shape_ * (input_timesteps_ - 1) + 3].item()

                loss_sampling = loss_fn_(y[:, k], prediction)
                cumulative_loss += loss_sampling

                gradients = tape.gradient(loss_sampling, model.trainable_variables)
                optimizer_.apply_gradients(zip(gradients, model.trainable_variables))

    return cumulative_loss / float(tf.shape(x)[1])


# ----------------------------------------------------------------------------------------------------------------------

input_shape = 5
input_timesteps = 4
dt = 0.01
steps = 5

model_path = 'saved_models/step_1/callbacks/2024_07_24/18_50_11/'
model = tf.keras.models.load_model(model_path + 'keras_model.h5')

# GET SCHEDULED SAMPLING DATA
dataset_scheduling = np.loadtxt('data/scheduled_sampling/train_data_step_no_shuffle_1_sched.csv', delimiter=',')
print('TOTAL LENGTH OF THE DATASET: ', len(dataset_scheduling))
print(dataset_scheduling.shape)

sequences = []
labels = []
for i in range(len(dataset_scheduling) - steps):
    temp = []
    temp2 = []
    sequences.append(temp)
    labels.append(temp2)

for i in range(len(dataset_scheduling)):
    if i + steps < len(dataset_scheduling):
        for t in range(steps):
            sequences[i].append(dataset_scheduling[i + t, :-3])
            labels[i].append(dataset_scheduling[i + t, -3:])

sequences = np.array(sequences)
labels = np.array(labels)

indices = np.random.permutation(len(sequences))

sequences = sequences[indices]
labels = labels[indices]

batch_size = 1000
dataset = tf.data.Dataset.from_tensor_slices((sequences, labels)).batch(batch_size)

learning_rate = 5e-8
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()
epsilon = 0.1

epochs = 5
sum_losses = 0.0
for epoch in tqdm(range(epochs)):
    for x_batch, y_batch in tqdm(dataset):
        sum_losses += train_step(x_batch, y_batch, model, input_shape, input_timesteps, dt, loss_fn, optimizer, epsilon)

    print(f'Epoch {epoch + 1}, Loss: {sum_losses.numpy() / len(dataset)}')
    learning_rate = learning_rate - 0.25e-8
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    tf.keras.models.save_model(model,
                               'saved_models/step_1/callbacks/2024_07_24/18_50_11/keras_scheduled_' + str(epoch) + '.h5')
    tf.keras.models.save_model(model,
                               'saved_models/step_1/callbacks/2024_07_24/18_50_11/keras_scheduled_' + str(epoch) + '.keras')

    epsilon = epsilon + 0.2

