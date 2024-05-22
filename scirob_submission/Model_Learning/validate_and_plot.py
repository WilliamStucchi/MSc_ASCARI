import tensorflow as tf
from run_tests import *
from plot_tests import *
import threading as th
from joblib import load


def build_model():
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Dense(units=Param['N1'], input_dim=Param['N_STATE_INPUT'] * Param['T'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N2'], activation='softplus'))
    model.add(tf.keras.layers.Dense(units=Param['N_TARGETS']))

    model.summary()

    return model


# ----------------------------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------------------------------------------------

def run_plot_save_v2(test_path_, model_path_, save_data_path_, scaler_present, name_):
    # Load NN model
    print(name_ + '[Loading model from: ' + model_path_ + 'keras_model.h5 ]')
    loaded_model = tf.keras.models.load_model(model_path_ + 'keras_model.h5')
    if scaler_present is not None:
        scaler = load(model_path_ + 'scaler.plk')
    else:
        scaler = None

    # Run test
    outcome, len_data = run_test(test_path_, loaded_model, scaler, 0, -1, save_data_path_, name_)

    # Plot results and save plots
    if outcome != 0:
        index_last_slash = save_data_path_.rfind('/')
        save_plots = save_data_path_[:index_last_slash + 1]
        plot_run(save_data_path_, test_path_, 0, len_data, save_plots, name_)

# ----------------------------------------------------------------------------------------------------------------------

def create_thread(name, test_path, m_path, sv_path, scaler_present):
    # Create new thread for the test to be executed
    return th.Thread(name=name,
              target=run_plot_save_v2,
              args=(test_path, m_path, sv_path, scaler_present, name))

# ----------------------------------------------------------------------------------------------------------------------

def start_parallel_threads(threads):
    for el in threads:
        el.start()
    for el in threads:
        el.join()


# ----------------------------------------------------------------------------------------------------------------------


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

test_set_0 = 'data/new/test_set_0.csv'
test_set_1 = 'data/new/test_set_1.csv'
test_set_2 = 'data/new/test_set_2.csv'
test_set_3 = 'data/new/test_set_3.csv'


"""
checkpoint_path = "saved_models/step_4/v1/model.ckpt"
loaded_model_4 = build_model()
loaded_model_4.load_weights(checkpoint_path)
_, leng = run_test(test_set_1, loaded_model_4, 0, -1, 'results/step_4/v1/results_test_1.csv', 'v1_41')
plot_run('results/step_4/v1/results_test_1.csv', test_set_1, 0, leng, 'results/step_4/v1/', 'v1_41')

_, leng = run_test(test_set_2, loaded_model_4, 0, -1, 'results/step_4/v1/results_test_2.csv', 'v1_42')
plot_run('results/step_4/v1/results_test_2.csv', test_set_2, 0, leng, 'results/step_4/v1/', 'v1_42')
"""

# Step 1
# Callbacks
save_path = 'results/step_1/callbacks/2024_05_17/12_29_37/'
model_path = 'saved_models/step_1/callbacks/2024_05_17/12_29_37/'
p2 = create_thread('t2', test_set_0, model_path, save_path + 'results_test_0.csv', None)
p3 = create_thread('t3', test_set_1, model_path, save_path + 'results_test_1.csv', None)
p4 = create_thread('t4', test_set_2, model_path, save_path + 'results_test_2.csv', None)

# Step 2
# Callbacks
save_path = 'results/step_2/callbacks/2024_05_15/15_25_37/'
model_path = 'saved_models/step_2/callbacks/2024_05_15/15_25_37/keras_model.h5'
p6 = create_thread('t6', test_set_0, model_path, save_path + 'results_test_0.csv', None)
p7 = create_thread('t7', test_set_1, model_path, save_path + 'results_test_1.csv', None)
p8 = create_thread('t8', test_set_2, model_path, save_path + 'results_test_2.csv', None)

# Step 3
# Callbacks
save_path = 'results/step_3/callbacks/2024_05_15/11_33_38/'
model_path = 'saved_models/step_3/callbacks/2024_05_15/11_33_38/keras_model.h5'
p10 = create_thread('t10', test_set_0, model_path, save_path + 'results_test_0.csv', None)
p11 = create_thread('t11', test_set_1, model_path, save_path + 'results_test_1.csv', None)
p12 = create_thread('t12', test_set_2, model_path, save_path + 'results_test_2.csv', None)

# Step 4
# Callbacks
save_path = 'results/step_4/callbacks/2024_05_15/15_25_37/'
model_path = 'saved_models/step_4/callbacks/2024_05_15/15_25_37/keras_model.h5'
p14 = create_thread('t14', test_set_0, model_path, save_path + 'results_test_0.csv', None)
p15 = create_thread('t15', test_set_1, model_path, save_path + 'results_test_1.csv', None)
p16 = create_thread('t16', test_set_2, model_path, save_path + 'results_test_2.csv', None)

start_parallel_threads([p2, p3, p4])
