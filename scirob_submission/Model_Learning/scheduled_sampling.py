from scirob_submission.Model_Learning.models_v2.models_V2 import *
from tqdm import tqdm
import random


# ----------------------------------------------------------------------------------------------------------------------

def train_step(x, y, model_, input_shape_, input_timesteps_, dt_, loss_fn_, optimizer_, epsilon_):
    cumulative_loss = 0.0
    bs = tf.shape(x)[0]
    # probability of an entire path to be replaced by the network prediction
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

                prediction = model_(inp, training=False)

                for j in range(bs):
                    corresponding_input = np.array(new_x[j:j + 1, k])
                    bs_prediction = prediction[j:j + 1]

                    yaw_acc = float(bs_prediction[:, 0])
                    ay = float(bs_prediction[:, 1])
                    ax = float(bs_prediction[:, 2])

                    # Euler integration
                    yaw_rate[j] = yaw_acc * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1):
                                                                     input_shape_ * (input_timesteps_ - 1) + 1].item()

                    # vy_dot = ay - yaw_rate * vx
                    dvy = ay - yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                 input_shape_ * (input_timesteps_ - 1) + 3]

                    # vx_dot = ax + yaw_rate * vy
                    dvx = ax + yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                 input_shape_ * (input_timesteps_ - 1) + 2]

                    # Euler integration
                    vy[j] = dvy.item() * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                  input_shape_ * (input_timesteps_ - 1) + 2].item()
                    vx[j] = dvx.item() * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                  input_shape_ * (input_timesteps_ - 1) + 3].item()

                loss_sampling = loss_fn_(y[:, k], prediction)
                cumulative_loss += loss_sampling

                gradients = tape.gradient(loss_sampling, model_.trainable_variables)
                optimizer_.apply_gradients(zip(gradients, model_.trainable_variables))

                """lr_ += 2e-7
                optimizer_ = tf.keras.optimizers.Adam(learning_rate=lr_)"""

    return cumulative_loss / float(tf.shape(x)[1])


# ----------------------------------------------------------------------------------------------------------------------

def val_step(x, y, model_, input_shape_, input_timesteps_, dt_, loss_fn_):
    cumulative_loss = 0.0
    bs = tf.shape(x)[0]

    yaw_rate = np.zeros(bs)
    vy = np.zeros(bs)
    vx = np.zeros(bs)

    mod_x = np.array(x)

    with ((tf.device('/GPU:0'))):
        for k in range(tf.shape(x)[1]):
            with tf.GradientTape():
                if k != 0:
                    for b in range(bs):
                        mod_x[b, k, :input_shape_ * (input_timesteps_ - 1)] = mod_x[b, k - 1, input_shape_:]

                        mod_x[b, k,
                              input_shape_ * (input_timesteps_ - 1):input_shape_ * (input_timesteps_ - 1) + 1] = yaw_rate[b]
                        mod_x[b, k,
                              input_shape_ * (input_timesteps_ - 1) + 1:input_shape_ * (input_timesteps_ - 1) + 2] = vy[b]
                        mod_x[b, k,
                              input_shape_ * (input_timesteps_ - 1) + 2:input_shape_ * (input_timesteps_ - 1) + 3] = vx[b]

                new_x = tf.convert_to_tensor(mod_x)
                inp = new_x[:, k]

                prediction = model_(inp, training=False)

                for j in range(bs):
                    corresponding_input = np.array(new_x[j:j + 1, k])
                    bs_prediction = prediction[j:j + 1]

                    yaw_acc = float(bs_prediction[:, 0])
                    ay = float(bs_prediction[:, 1])
                    ax = float(bs_prediction[:, 2])

                    # Euler integration
                    yaw_rate[j] = yaw_acc * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1):
                                                                     input_shape_ * (input_timesteps_ - 1) + 1].item()

                    # vy_dot = ay - yaw_rate * vx
                    dvy = ay - yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                 input_shape_ * (input_timesteps_ - 1) + 3]

                    # vx_dot = ax + yaw_rate * vy
                    dvx = ax + yaw_rate[j] * corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                 input_shape_ * (input_timesteps_ - 1) + 2]

                    # Euler integration
                    vy[j] = dvy.item() * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 1:
                                                                  input_shape_ * (input_timesteps_ - 1) + 2].item()
                    vx[j] = dvx.item() * dt_ + corresponding_input[:, input_shape_ * (input_timesteps_ - 1) + 2:
                                                                  input_shape_ * (input_timesteps_ - 1) + 3].item()

                loss_sampling = loss_fn_(y[:, k], prediction)
                cumulative_loss += loss_sampling

    return cumulative_loss / float(tf.shape(x)[1])


# ----------------------------------------------------------------------------------------------------------------------

input_shape = 5
input_timesteps = 4
dt = 0.01
steps = 5

model_path = 'saved_models/step_1/callbacks/2024_07_22/13_34_22/'
model = tf.keras.models.load_model(model_path + 'keras_model.h5')

model.layers[1].trainable = False

model.summary()

# GET SCHEDULED SAMPLING DATA
dataset_scheduling = np.loadtxt('data/scheduled_sampling/train_data_step_no_shuffle_1_sched_cplt.csv', delimiter=',')
print('TOTAL LENGTH OF THE DATASET: ', len(dataset_scheduling))
print(dataset_scheduling.shape)

# Create sequence for training
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

# Training and validation
dataset_size = len(sequences)
train_size = int(0.9 * dataset_size)

feat_training = sequences[:train_size]
feat_validation = sequences[train_size:]
label_training = labels[:train_size]
label_validation = labels[train_size:]

# Create batches
batch_size = 1000
dataset_training = tf.data.Dataset.from_tensor_slices((feat_training, label_training)).batch(batch_size)
dataset_validation = tf.data.Dataset.from_tensor_slices((feat_validation, label_validation)).batch(batch_size)

# Learning parameters
learning_rate = 2e-8
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.MeanSquaredError()
epsilon = 1

# Learning loop
epochs = 8
loss_train = 0.0
loss_val = 0.0
old_loss_val = float('inf')

for epoch in tqdm(range(epochs)):
    for x_batch, y_batch in tqdm(dataset_training):
        loss_train += train_step(x_batch, y_batch, model, input_shape, input_timesteps, dt,
                                 loss_fn, optimizer, epsilon)

    for x_batch, y_batch in tqdm(dataset_validation):
        loss_val += val_step(x_batch, y_batch, model, input_shape, input_timesteps, dt, loss_fn)

    print(f'Epoch {epoch}, Loss: {loss_train / len(dataset_training)}, '
          f'Val Loss: {loss_val / len(dataset_validation)}')

    if loss_val < old_loss_val:
        old_loss_val = loss_val

        tf.keras.models.save_model(model,
                                   'saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_2_eps.h5')
        tf.keras.models.save_model(model,
                               'saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_2_eps.keras')

        print('MODEL SAVED AT [saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_eps.keras]')

    loss_train = 0.0
    loss_val = 0.0

