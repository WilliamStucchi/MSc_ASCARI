import tensorflow as tf
from run_tests import *
from plot_tests import *
import threading as th
from joblib import load
import tensorflow as tf


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
    tf.compat.v1.disable_eager_execution()

    """# Old version of model
    loaded_model = build_model()
    loaded_model.load_weights('saved_models/step_1/test_old/model.ckpt')"""

    # New version of model
    print(name_ + '[Loading model from: ' + model_path_ + ']')
    # loaded_model = tf.keras.models.load_model(model_path_ + 'keras_model.h5')
    loaded_model = tf.keras.models.load_model(model_path_)

    if scaler_present is not None:
        print(name_ + '[Loading scaler]')
        scaler = load(model_path_ + 'scaler.plk')
    else:
        print(name_ + '[NO SCALER FOUND]')
        scaler = None

    print(name_ + '[START TEST]')
    # Run test
    outcome, len_data = run_test(test_path_, loaded_model, scaler, 0, -1, save_data_path_, name_)

    print(name_ + '[TEST ENDED]')
    print(name_ + '[PLOTTING]')
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
test_set_1_rrff= 'data/new/test_set_1_rrff.csv'
test_set_2_rrff = 'data/new/test_set_2_rrff.csv'
test_set_3_rrff = 'data/new/test_set_3_rrff.csv'
test_set_3 = 'data/new/test_set_3.csv'
test_set_4 = 'data/new/test_set_4.csv'

test_set_mu_1 = 'data/new/test_set_mu_1.csv'
test_set_mu_08 = 'data/new/test_set_mu_08.csv'
test_set_mu_0806 = 'data/new/test_set_mu_0806.csv'
test_set_mu_0804 = 'data/new/test_set_mu_0804.csv'
test_set_mu_06 = 'data/new/test_set_mu_06.csv'
test_set_mu_06045 = 'data/new/test_set_mu_06045.csv'
test_set_mu_0603 = 'data/new/test_set_mu_0603.csv'

test_set_mass_0 = 'data/new/test_set_mass_0.csv'
test_set_mass_1 = 'data/new/test_set_mass_1.csv'
test_set_mass_2 = 'data/new/test_set_mass_2.csv'
test_set_mass_3 = 'data/new/test_set_mass_3.csv'
test_set_mass_4 = 'data/new/test_set_mass_4.csv'
test_set_mass_5 = 'data/new/test_set_mass_5.csv'

test_set_cplt_1 = 'data/piste_training_complete/test_set_1.csv'

test_set_stepsteer_fx100_0deg = 'data/handling/test_set_stepsteer_fx100_0deg.csv'
test_set_stepsteer_fx100_1deg = 'data/handling/test_set_stepsteer_fx100_1deg.csv'
test_set_stepsteer_fx100_2deg = 'data/handling/test_set_stepsteer_fx100_2deg.csv'
test_set_stepsteer_fx100_3deg = 'data/handling/test_set_stepsteer_fx100_3deg.csv'
test_set_stepsteer_fx100_4deg = 'data/handling/test_set_stepsteer_fx100_4deg.csv'
test_set_stepsteer_fx100_5deg = 'data/handling/test_set_stepsteer_fx100_5deg.csv'
test_set_stepsteer_fx100_10deg = 'data/handling/test_set_stepsteer_fx100_10deg.csv'
test_set_stepsteer_fx100_15deg = 'data/handling/test_set_stepsteer_fx100_15deg.csv'
test_set_stepsteer_fx100_20deg = 'data/handling/test_set_stepsteer_fx100_20deg.csv'
test_set_stepsteer_fx100_25deg = 'data/handling/test_set_stepsteer_fx100_25deg.csv'
test_set_stepsteer_fx100_30deg = 'data/handling/test_set_stepsteer_fx100_30deg.csv'
test_set_stepsteer_fx100_45deg = 'data/handling/test_set_stepsteer_fx100_45deg.csv'
test_set_stepsteer_fx100_60deg = 'data/handling/test_set_stepsteer_fx100_60deg.csv'
test_set_stepsteer_fx100_75deg = 'data/handling/test_set_stepsteer_fx100_75deg.csv'
test_set_stepsteer_fx100_90deg = 'data/handling/test_set_stepsteer_fx100_90deg.csv'
test_set_stepsteer_fx100_100deg = 'data/handling/test_set_stepsteer_fx100_100deg.csv'

test_set_stepsteer_fx25_0deg = 'data/handling/test_set_stepsteer_fx25_0deg.csv'
test_set_stepsteer_fx25_1deg = 'data/handling/test_set_stepsteer_fx25_1deg.csv'
test_set_stepsteer_fx25_2deg = 'data/handling/test_set_stepsteer_fx25_2deg.csv'
test_set_stepsteer_fx25_3deg = 'data/handling/test_set_stepsteer_fx25_3deg.csv'
test_set_stepsteer_fx25_4deg = 'data/handling/test_set_stepsteer_fx25_4deg.csv'
test_set_stepsteer_fx25_5deg = 'data/handling/test_set_stepsteer_fx25_5deg.csv'
test_set_stepsteer_fx25_10deg = 'data/handling/test_set_stepsteer_fx25_10deg.csv'
test_set_stepsteer_fx25_15deg = 'data/handling/test_set_stepsteer_fx25_15deg.csv'
test_set_stepsteer_fx25_20deg = 'data/handling/test_set_stepsteer_fx25_20deg.csv'
test_set_stepsteer_fx25_25deg = 'data/handling/test_set_stepsteer_fx25_25deg.csv'
test_set_stepsteer_fx25_30deg = 'data/handling/test_set_stepsteer_fx25_30deg.csv'
test_set_stepsteer_fx25_45deg = 'data/handling/test_set_stepsteer_fx25_45deg.csv'
test_set_stepsteer_fx25_60deg = 'data/handling/test_set_stepsteer_fx25_60deg.csv'
test_set_stepsteer_fx25_75deg = 'data/handling/test_set_stepsteer_fx25_75deg.csv'
test_set_stepsteer_fx25_90deg = 'data/handling/test_set_stepsteer_fx25_90deg.csv'
test_set_stepsteer_fx25_100deg = 'data/handling/test_set_stepsteer_fx25_100deg.csv'

test_set_rampsteer_fx100_0 = 'data/handling/test_set_rampsteer_fx100_0.csv'
test_set_rampsteer_fx100_1 = 'data/handling/test_set_rampsteer_fx100_1.csv'
test_set_rampsteer_fx100_2 = 'data/handling/test_set_rampsteer_fx100_2.csv'
test_set_rampsteer_fx25_3 = 'data/handling/test_set_rampsteer_fx25_3.csv'
test_set_rampsteer_fx25_4 = 'data/handling/test_set_rampsteer_fx25_4.csv'
test_set_rampsteer_fx25_5 = 'data/handling/test_set_rampsteer_fx25_5.csv'

test_set_sinesteer_fx100_0 = 'data/handling/test_set_sinesteer_fx100_0.csv'
test_set_sinesteer_fx100_1 = 'data/handling/test_set_sinesteer_fx100_1.csv'
test_set_sinesteer_fx100_2 = 'data/handling/test_set_sinesteer_fx100_2.csv'
test_set_sinesteer_fx100_3 = 'data/handling/test_set_sinesteer_fx100_3.csv'
test_set_sinesteer_fx100_4 = 'data/handling/test_set_sinesteer_fx100_4.csv'
test_set_sinesteer_fx100_5 = 'data/handling/test_set_sinesteer_fx100_5.csv'
test_set_sinesteer_fx100_6 = 'data/handling/test_set_sinesteer_fx100_6.csv'
test_set_sinesteer_fx100_7 = 'data/handling/test_set_sinesteer_fx100_7.csv'
test_set_sinesteer_fx25_8 = 'data/handling/test_set_sinesteer_fx25_8.csv'
test_set_sinesteer_fx25_9 = 'data/handling/test_set_sinesteer_fx25_9.csv'
test_set_sinesteer_fx25_10 = 'data/handling/test_set_sinesteer_fx25_10.csv'
test_set_sinesteer_fx25_11 = 'data/handling/test_set_sinesteer_fx25_11.csv'
test_set_sinesteer_fx25_12 = 'data/handling/test_set_sinesteer_fx25_12.csv'
test_set_sinesteer_fx25_13 = 'data/handling/test_set_sinesteer_fx25_13.csv'
test_set_sinesteer_fx25_14 = 'data/handling/test_set_sinesteer_fx25_15.csv'
test_set_sinesteer_fx25_15 = 'data/handling/test_set_sinesteer_fx25_15.csv'

test_paramstudy_grip_1_perf_100 = 'data/new/test_set_paramstudy_grip_1_perf_100.csv'
test_paramstudy_grip_1_perf_75 = 'data/new/test_set_paramstudy_grip_1_perf_75.csv'
test_paramstudy_grip_1_perf_50 = 'data/new/test_set_paramstudy_grip_1_perf_50.csv'
test_paramstudy_grip_06_perf_100 = 'data/new/test_set_paramstudy_grip_06_perf_100.csv'
test_paramstudy_grip_06_perf_75 = 'data/new/test_set_paramstudy_grip_06_perf_75.csv'
test_paramstudy_grip_06_perf_50 = 'data/new/test_set_paramstudy_grip_06_perf_50.csv'
test_paramstudy_grip_08_perf_100 = 'data/new/test_set_paramstudy_grip_08_perf_100.csv'
test_paramstudy_grip_08_perf_75 = 'data/new/test_set_paramstudy_grip_08_perf_75.csv'
test_paramstudy_grip_08_perf_50 = 'data/new/test_set_paramstudy_grip_08_perf_50.csv'

test_set_from0_fx100_0deg = 'data/new/test_set_from0_fx100_0deg.csv'
test_set_from0_fx100_1deg = 'data/new/test_set_from0_fx100_1deg.csv'
test_set_from0_fx100_2deg = 'data/new/test_set_from0_fx100_2deg.csv'
test_set_from0_fx100_3deg = 'data/new/test_set_from0_fx100_3deg.csv'
test_set_from0_fx100_4deg = 'data/new/test_set_from0_fx100_4deg.csv'
test_set_from0_fx100_5deg = 'data/new/test_set_from0_fx100_5deg.csv'
test_set_from0_fx100_10deg = 'data/new/test_set_from0_fx100_10deg.csv'
test_set_from0_fx100_15deg = 'data/new/test_set_from0_fx100_15deg.csv'
test_set_from0_fx100_20deg = 'data/new/test_set_from0_fx100_20deg.csv'
test_set_from0_fx100_25deg = 'data/new/test_set_from0_fx100_25deg.csv'
test_set_from0_fx100_30deg = 'data/new/test_set_from0_fx100_30deg.csv'
test_set_from0_fx100_45deg = 'data/new/test_set_from0_fx100_45deg.csv'
test_set_from0_fx100_60deg = 'data/new/test_set_from0_fx100_60deg.csv'

latest_test_grip_1_perf_100 = 'data/latest_CRT/latest_sets/test_set_mu_1_perf_100.csv'
latest_test_grip_1_perf_75 = 'data/latest_CRT/latest_sets/test_set_mu_1_perf_75.csv'
latest_test_grip_1_perf_50 = 'data/latest_CRT/latest_sets/test_set_mu_1_perf_50.csv'
latest_test_grip_1_perf_25 = 'data/latest_CRT/latest_sets/test_set_mu_1_perf_25.csv'
latest_test_grip_08_perf_100 = 'data/latest_CRT/latest_sets/test_set_mu_08_perf_100.csv'
latest_test_grip_08_perf_75 = 'data/latest_CRT/latest_sets/test_set_mu_08_perf_75.csv'
latest_test_grip_08_perf_50 = 'data/latest_CRT/latest_sets/test_set_mu_08_perf_50.csv'
latest_test_grip_06_perf_100 = 'data/latest_CRT/latest_sets/test_set_mu_06_perf_100.csv'
latest_test_grip_06_perf_75 = 'data/latest_CRT/latest_sets/test_set_mu_06_perf_75.csv'
latest_test_grip_06_perf_50 = 'data/latest_CRT/latest_sets/test_set_mu_06_perf_50.csv'

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
# save_path = 'results/step_1/callbacks/2024_10_08/16_26_37/'
# save_path = 'results/step_1/callbacks/2024_10_11/22_06_11/'
"""save_path = 'results/step_1/callbacks/2024_06_12/13_00_14/'
model_path = 'saved_models/step_1/callbacks/2024_06_12/13_00_14/keras_model.h5'"""

"""save_path = 'results/step_1/callbacks/2024_10_14/19_30_02/'
model_path = 'saved_models/step_1/callbacks/2024_10_14/19_30_02/keras_model.h5'"""

save_path = 'results/step_1/callbacks/2024_10_20/16_09_55/'
model_path = 'saved_models/step_1/callbacks/2024_10_20/16_09_55/keras_model.h5'

"""save_path = 'results/step_1/callbacks/2024_09_17/17_41_36/'
model_path = 'saved_models/step_1/callbacks/2024_09_17/17_41_36/keras_model.h5'"""

p0 = create_thread('t0', test_set_mu_1, model_path, save_path + 'results_test_mu_1.csv', None)
p1 = create_thread('t1', test_set_mu_08, model_path, save_path + 'results_test_mu_08.csv', None)
p2 = create_thread('t2', test_set_mu_06, model_path, save_path + 'results_test_mu_06.csv', None)
p3 = create_thread('t3', test_set_1, model_path, save_path + 'results_test_perf_1.csv', None)
p4 = create_thread('t4', test_set_2, model_path, save_path + 'results_test_perf_2.csv', None)
p5 = create_thread('t5', test_set_3, model_path, save_path + 'results_test_perf_3.csv', None)
p6 = create_thread('t6', test_set_mu_0806, model_path, save_path + 'results_test_mu_0806.csv', None)
p7 = create_thread('t7', test_set_mu_0804, model_path, save_path + 'results_test_mu_0804.csv', None)
p8 = create_thread('t8', test_set_mu_06045, model_path, save_path + 'results_test_mu_06045.csv', None)
p9 = create_thread('t9', test_set_mu_0603, model_path, save_path + 'results_test_mu_0603.csv', None)

step_fx100_0deg = create_thread('step_fx100_0deg', test_set_stepsteer_fx100_0deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_0deg.csv', None)
step_fx100_1deg = create_thread('step_fx100_1deg', test_set_stepsteer_fx100_1deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_1deg.csv', None)
step_fx100_2deg = create_thread('step_fx100_2deg', test_set_stepsteer_fx100_2deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_2deg.csv', None)
step_fx100_3deg = create_thread('step_fx100_3deg', test_set_stepsteer_fx100_3deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_3deg.csv', None)
step_fx100_4deg = create_thread('step_fx100_4deg', test_set_stepsteer_fx100_4deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_4deg.csv', None)
step_fx100_5deg = create_thread('step_fx100_5deg', test_set_stepsteer_fx100_5deg, model_path,
                                save_path + 'results_test_stepsteer_fx100_5deg.csv', None)
step_fx100_10deg = create_thread('step_fx100_10deg', test_set_stepsteer_fx100_10deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_10deg.csv', None)
step_fx100_15deg = create_thread('step_fx100_15deg', test_set_stepsteer_fx100_15deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_15deg.csv', None)
step_fx100_20deg = create_thread('step_fx100_20deg', test_set_stepsteer_fx100_20deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_20deg.csv', None)
step_fx100_25deg = create_thread('step_fx100_25deg', test_set_stepsteer_fx100_25deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_25deg.csv', None)
step_fx100_30deg = create_thread('step_fx100_30deg', test_set_stepsteer_fx100_30deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_30deg.csv', None)
step_fx100_45deg = create_thread('step_fx100_45deg', test_set_stepsteer_fx100_45deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_45deg.csv', None)
step_fx100_60deg = create_thread('step_fx100_60deg', test_set_stepsteer_fx100_60deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_60deg.csv', None)
step_fx100_75deg = create_thread('step_fx100_75deg', test_set_stepsteer_fx100_75deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_75deg.csv', None)
step_fx100_90deg = create_thread('step_fx100_90deg', test_set_stepsteer_fx100_90deg, model_path,
                                 save_path + 'results_test_stepsteer_fx100_90deg.csv', None)
step_fx100_100deg = create_thread('step_fx100_100deg', test_set_stepsteer_fx100_100deg, model_path,
                                  save_path + 'results_test_stepsteer_fx100_100deg.csv', None)

step_fx25_0deg = create_thread('step_fx25_0deg', test_set_stepsteer_fx25_0deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_0deg.csv', None)
step_fx25_1deg = create_thread('step_fx25_1deg', test_set_stepsteer_fx25_1deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_1deg.csv', None)
step_fx25_2deg = create_thread('step_fx25_2deg', test_set_stepsteer_fx25_2deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_2deg.csv', None)
step_fx25_3deg = create_thread('step_fx25_3deg', test_set_stepsteer_fx25_3deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_3deg.csv', None)
step_fx25_4deg = create_thread('step_fx25_4deg', test_set_stepsteer_fx25_4deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_4deg.csv', None)
step_fx25_5deg = create_thread('step_fx25_5deg', test_set_stepsteer_fx25_5deg, model_path,
                               save_path + 'results_test_stepsteer_fx25_5deg.csv', None)
step_fx25_10deg = create_thread('step_fx25_10deg', test_set_stepsteer_fx25_10deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_10deg.csv', None)
step_fx25_15deg = create_thread('step_fx25_15deg', test_set_stepsteer_fx25_15deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_15deg.csv', None)
step_fx25_20deg = create_thread('step_fx25_20deg', test_set_stepsteer_fx25_20deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_20deg.csv', None)
step_fx25_25deg = create_thread('step_fx25_25deg', test_set_stepsteer_fx25_25deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_25deg.csv', None)
step_fx25_30deg = create_thread('step_fx25_30deg', test_set_stepsteer_fx25_30deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_30deg.csv', None)
step_fx25_45deg = create_thread('step_fx25_45deg', test_set_stepsteer_fx25_45deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_45deg.csv', None)
step_fx25_60deg = create_thread('step_fx25_60deg', test_set_stepsteer_fx25_60deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_60deg.csv', None)
step_fx25_75deg = create_thread('step_fx25_75deg', test_set_stepsteer_fx25_75deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_75deg.csv', None)
step_fx25_90deg = create_thread('step_fx25_90deg', test_set_stepsteer_fx25_90deg, model_path,
                                save_path + 'results_test_stepsteer_fx25_90deg.csv', None)
step_fx25_100deg = create_thread('step_fx25_100deg', test_set_stepsteer_fx25_100deg, model_path,
                                 save_path + 'results_test_stepsteer_fx25_100deg.csv', None)

ramp_fx100_0 = create_thread('ramp_fx100_0', test_set_rampsteer_fx100_0, model_path, save_path + 'results_test_rampsteer_fx100_0.csv', None)
ramp_fx100_1 = create_thread('ramp_fx100_1', test_set_rampsteer_fx100_1, model_path, save_path + 'results_test_rampsteer_fx100_1.csv', None)
ramp_fx100_2 = create_thread('ramp_fx100_2', test_set_rampsteer_fx100_2, model_path, save_path + 'results_test_rampsteer_fx100_2.csv', None)
ramp_fx25_3 = create_thread('ramp_fx25_3', test_set_rampsteer_fx25_3, model_path, save_path + 'results_test_rampsteer_fx25_3.csv', None)
ramp_fx25_4 = create_thread('ramp_fx25_4', test_set_rampsteer_fx25_4, model_path, save_path + 'results_test_rampsteer_fx25_4.csv', None)
ramp_fx25_5 = create_thread('ramp_fx25_5', test_set_rampsteer_fx25_5, model_path, save_path + 'results_test_rampsteer_fx25_5.csv', None)

sine_fx100_0 = create_thread('sine_fx100_0', test_set_sinesteer_fx100_0, model_path, save_path + 'results_test_sinesteer_fx100_0.csv', None)
sine_fx100_1 = create_thread('sine_fx100_1', test_set_sinesteer_fx100_1, model_path, save_path + 'results_test_sinesteer_fx100_1.csv', None)
sine_fx100_2 = create_thread('sine_fx100_2', test_set_sinesteer_fx100_2, model_path, save_path + 'results_test_sinesteer_fx100_2.csv', None)
sine_fx100_3 = create_thread('sine_fx100_3', test_set_sinesteer_fx100_3, model_path, save_path + 'results_test_sinesteer_fx100_3.csv', None)
sine_fx100_4 = create_thread('sine_fx100_4', test_set_sinesteer_fx100_4, model_path, save_path + 'results_test_sinesteer_fx100_4.csv', None)
sine_fx100_5 = create_thread('sine_fx100_5', test_set_sinesteer_fx100_5, model_path, save_path + 'results_test_sinesteer_fx100_5.csv', None)
sine_fx100_6 = create_thread('sine_fx100_6', test_set_sinesteer_fx100_6, model_path, save_path + 'results_test_sinesteer_fx100_6.csv', None)
sine_fx100_7 = create_thread('sine_fx100_7', test_set_sinesteer_fx100_7, model_path, save_path + 'results_test_sinesteer_fx100_7.csv', None)
sine_fx25_8 = create_thread('sine_fx25_8', test_set_sinesteer_fx25_8, model_path, save_path + 'results_test_sinesteer_fx25_8.csv', None)
sine_fx25_9 = create_thread('sine_fx25_9', test_set_sinesteer_fx25_9, model_path, save_path + 'results_test_sinesteer_fx25_9.csv', None)
sine_fx25_10 = create_thread('sine_fx25_10', test_set_sinesteer_fx25_10, model_path, save_path + 'results_test_sinesteer_fx25_10.csv', None)
sine_fx25_11 = create_thread('sine_fx25_11', test_set_sinesteer_fx25_11, model_path, save_path + 'results_test_sinesteer_fx25_11.csv', None)
sine_fx25_12 = create_thread('sine_fx25_12', test_set_sinesteer_fx25_12, model_path, save_path + 'results_test_sinesteer_fx25_12.csv', None)
sine_fx25_13 = create_thread('sine_fx25_13', test_set_sinesteer_fx25_13, model_path, save_path + 'results_test_sinesteer_fx25_13.csv', None)
sine_fx25_14 = create_thread('sine_fx25_14', test_set_sinesteer_fx25_14, model_path, save_path + 'results_test_sinesteer_fx25_14.csv', None)
sine_fx25_15 = create_thread('sine_fx25_15', test_set_sinesteer_fx25_15, model_path, save_path + 'results_test_sinesteer_fx25_15.csv', None)

p11 = create_thread('t11', test_paramstudy_grip_1_perf_100, model_path, save_path + 'results_test_grip_1_perf_100.csv', None)
p12 = create_thread('t12', test_paramstudy_grip_1_perf_75, model_path, save_path + 'results_test_grip_1_perf_75.csv', None)
p13 = create_thread('t13', test_paramstudy_grip_1_perf_50, model_path, save_path + 'results_test_grip_1_perf_50.csv', None)
p14 = create_thread('t14', test_paramstudy_grip_06_perf_100, model_path, save_path + 'results_test_grip_06_perf_100.csv', None)
p15 = create_thread('t15', test_paramstudy_grip_06_perf_75, model_path, save_path + 'results_test_grip_06_perf_75.csv', None)
p16 = create_thread('t16', test_paramstudy_grip_06_perf_50, model_path, save_path + 'results_test_grip_06_perf_50.csv', None)
p17 = create_thread('t17', test_paramstudy_grip_08_perf_100, model_path, save_path + 'results_test_grip_08_perf_100.csv', None)
p18 = create_thread('t18', test_paramstudy_grip_08_perf_75, model_path, save_path + 'results_test_grip_08_perf_75.csv', None)
p19 = create_thread('t19', test_paramstudy_grip_08_perf_50, model_path, save_path + 'results_test_grip_08_perf_50.csv', None)

mu1_perf100 = create_thread('mu1_perf100', latest_test_grip_1_perf_100, model_path,
                        save_path + 'results_latest_mu_1_perf_100.csv', None)
mu1_perf75 = create_thread('mu1_perf75', latest_test_grip_1_perf_75, model_path,
                        save_path + 'results_latest_mu_1_perf_75.csv', None)
mu1_perf50 = create_thread('mu1_perf50', latest_test_grip_1_perf_50, model_path,
                        save_path + 'results_latest_mu_1_perf_50.csv', None)
mu1_perf25 = create_thread('mu1_perf25', latest_test_grip_1_perf_25, model_path,
                        save_path + 'results_latest_mu_1_perf_25.csv', None)
mu08_perf100 = create_thread('mu08_perf100', latest_test_grip_08_perf_100, model_path,
                        save_path + 'results_latest_mu_08_perf_100.csv', None)
mu08_perf75 = create_thread('mu08_perf75', latest_test_grip_08_perf_75, model_path,
                        save_path + 'results_latest_mu_08_perf_75.csv', None)
mu08_perf50 = create_thread('mu08_perf50', latest_test_grip_08_perf_50, model_path,
                        save_path + 'results_latest_mu_08_perf_50.csv', None)
mu06_perf100 = create_thread('mu06_perf100', latest_test_grip_06_perf_100, model_path,
                        save_path + 'results_latest_mu_06_perf_100.csv', None)
mu06_perf75 = create_thread('mu06_perf75', latest_test_grip_06_perf_75, model_path,
                        save_path + 'results_latest_mu_06_perf_75.csv', None)
mu06_perf50 = create_thread('mu06_perf50', latest_test_grip_06_perf_50, model_path,
                        save_path + 'results_latest_mu_06_perf_50.csv', None)

# p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p11, p12, p13, p14, p15, p16, p17, p18, p19,
lista = [step_fx100_0deg, step_fx100_1deg, step_fx100_2deg, step_fx100_3deg,
         step_fx100_4deg, step_fx100_5deg, step_fx100_10deg, step_fx100_15deg, step_fx100_20deg, step_fx100_25deg,
         step_fx100_30deg, step_fx100_45deg, step_fx100_60deg, step_fx100_75deg, step_fx100_90deg, step_fx100_100deg,
         step_fx25_0deg, step_fx25_1deg, step_fx25_2deg, step_fx25_3deg, step_fx25_4deg, step_fx25_5deg,
         step_fx25_10deg, step_fx25_15deg, step_fx25_20deg, step_fx25_25deg, step_fx25_30deg, step_fx25_45deg,
         step_fx25_60deg, step_fx25_75deg, step_fx25_90deg, step_fx25_100deg, ramp_fx100_0, ramp_fx100_1, ramp_fx100_2,
         ramp_fx25_3, ramp_fx25_4, ramp_fx25_5, sine_fx100_0, sine_fx100_1, sine_fx100_2, sine_fx100_3, sine_fx100_4,
         sine_fx100_5, sine_fx100_6, sine_fx100_7, sine_fx25_8, sine_fx25_9, sine_fx25_10, sine_fx25_11, sine_fx25_12,
         sine_fx25_13, sine_fx25_14, sine_fx25_15]

new = [p11, p12, p13, p14, p15, p16, p17, p18, p19]

for el in new:
    el.start()
    el.join()

"""save_path = 'results/step_1/callbacks/2024_07_22/13_34_22/eps_0/'
model_path = 'saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_eps_0.keras'
t1 = create_thread('t1', test_set_mu_0806, model_path, save_path + 'results_test_mu_0806.csv', None)
save_path = 'results/step_1/callbacks/2024_07_22/13_34_22/eps_1/'
model_path = 'saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_eps_1.keras'
t2 = create_thread('t2', test_set_mu_0806, model_path, save_path + 'results_test_mu_0806.csv', None)
save_path = 'results/step_1/callbacks/2024_07_22/13_34_22/eps_2/'
model_path = 'saved_models/step_1/callbacks/2024_07_22/13_34_22/keras_scheduled_eps_2.keras'
t3 = create_thread('t3', test_set_mu_0806, model_path, save_path + 'results_test_mu_0806.csv', None)

start_parallel_threads([t1, t2, t3])"""
