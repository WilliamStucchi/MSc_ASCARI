import tensorflow as tf
from run_tests import *
from plot_tests import *


def build_model():
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(units=Param['N1'], input_dim=Param['N_STATE_INPUT'] * Param['T'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N2'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N_TARGETS']))

    model.summary()

    return model


def data_to_run(input_shape: int, normalize_values: bool):
    data = np.loadtxt("data/data_to_run.csv", delimiter=",")

    # Extract values
    vx = data[:, 0].copy()
    vy = data[:, 1].copy()
    psi = data[:, 2].copy()
    delta = data[:, 5].copy()
    trl = data[:, 6].copy()
    trr = data[:, 7].copy()

    # Create final dataset
    result = np.zeros((data.shape[0], input_shape))

    # Put in final dataset
    result[:, 0] = psi
    result[:, 1] = vy
    result[:, 2] = vx
    result[:, 3] = delta

    # 0.05655 is the radius (in meters) of the wheel of a car
    # 177620 is the normalization factor adopted by the Stanford code
    result[:, 4] = (trl + trr) / (2 * 0.05 * 177620)
    # result[:, 4] = (0.5 * (trl + trr)) / 177620

    # normalize between -1 and 1
    if normalize_values:
        min_psi = result[:, 0].min()
        max_psi = result[:, 0].max()
        min_vy = result[:, 1].min()
        max_vy = result[:, 1].max()
        min_vx = result[:, 2].min()
        max_vx = result[:, 2].max()
        min_delta = result[:, 3].min()
        max_delta = result[:, 3].max()
        min_fx = result[:, 4].min()
        max_fx = result[:, 4].max()

        for i in range(0, result.shape[0]):
            result[i, 0] = 2 * (result[i, 0] - min_psi) / (max_psi - min_psi) - 1
            result[i, 1] = 2 * (result[i, 1] - min_vy) / (max_vy - min_vy) - 1
            result[i, 2] = 2 * (result[i, 2] - min_vx) / (max_vx - min_vx) - 1
            result[i, 3] = 2 * (result[i, 3] - min_delta) / (max_delta - min_delta) - 1
            result[i, 4] = 2 * (result[i, 4] - min_fx) / (max_fx - min_fx) - 1

    """
    for i in range(100):
        #print('Data: ', data[i])
        print('-----------')
        print('Res: ', result[i])
        # input('Waiting...')
        print('-----------')
    input('wait')
    """

    return result


def run_plot_save_v2(test_path_, model, save_data):
    outcome, len_data = run_test(test_path_, model, 0, -1, save_data)
    if outcome != 0:
        save_plots = save_data.replace('results.csv', '')
        plot_run(save_data, test_path, 0, len_data, save_plots)


"""
# Test parameters
n_test = 37
run_timestart = 0
iteration_step = 1250
run_timespan = 1500

for i in [1, 2, 3, 5, 6]:
    # Load model
    checkpoint_path = "saved_models/gen_test_3_w/exp_" + str(i) + "_mod/model.ckpt"
    loaded_model = build_model()
    loaded_model.load_weights(checkpoint_path)

    for counter in range(0, n_test):
        idx_start = run_timestart + counter * iteration_step

        print('Start test #', str(counter))
        print('------------------------------')
        if run_test(None, loaded_model, idx_start, counter, run_timespan, i):
            plot_run(idx_start, counter, run_timespan, i)
            # input('Waiting...')
"""

# Step 1
# No callbacks
model_path = 'saved_models/step_1/no_callbacks/2024_05_02/09_14_05/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_1/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_1/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 1
# Callbacks
model_path = 'saved_models/step_1/callbacks/2024_05_02/09_59_22/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_1/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_1/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 2
# No callbacks
model_path = 'saved_models/step_2/no_callbacks/2024_05_02/09_14_05/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_2/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_2/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 2
# Callbacks
model_path = 'saved_models/step_2/callbacks/2024_05_02/09_59_22/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_2/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_2/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 3
# No callbacks
model_path = 'saved_models/step_3/no_callbacks/2024_05_02/09_14_05/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_3/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_3/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 3
# Callbacks
model_path = 'saved_models/step_3/callbacks/2024_05_02/09_59_22/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_3/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_3/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 4
# No callbacks
model_path = 'saved_models/step_4/no_callbacks/2024_05_02/09_14_05/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_4/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_4/no_callbacks/2024_05_02/09_14_05/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

# Step 4
# Callbacks
model_path = 'saved_models/step_1/callbacks/2024_05_02/09_59_22/keras_model.h5'
loaded_model = tf.keras.models.load_model(model_path)

test_path = 'data/CRT/test_set_1.csv'
save_path = 'results/step_4/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)

test_path = 'data/CRT/test_set_2.csv'
save_path = 'results/step_4/callbacks/2024_05_02/09_59_22/results.csv'
run_plot_save_v2(test_path, loaded_model, save_path)
