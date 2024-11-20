import math

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------

def load_features(dataset):
    dataset_ = np.loadtxt(dataset, delimiter=',')

    perc_validation = 0.2

    train_feat, _, val_feat, _ = get_set(dataset_, perc_validation)

    return train_feat, val_feat


# ----------------------------------------------------------------------------------------------------------------------

def get_set(dataset, perc_valid):
    train_features, train_lab = get_training(dataset, perc_valid)
    val_features, val_lab = get_validation(dataset, perc_valid)

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

def fill_lists(source):
    dest1, dest2, dest3, dest4, dest5 = [], [], [], [], []

    for el in source:
        for count in [0, 5, 10, 15]:
            dest1.append(el[count])
            dest2.append(el[count + 1])
            dest3.append(el[count + 2])
            dest4.append(el[count + 3])
            dest5.append(el[count + 4])

    return dest1, dest2, dest3, dest4, dest5


# ----------------------------------------------------------------------------------------------------------------------

def pad(to_pad, final_length):
    return [np.pad(arr, (0, final_length - len(arr)), constant_values=np.nan) for arr in to_pad]


# ----------------------------------------------------------------------------------------------------------------------

def show_kde_car_perf(title, data, folder, save_format):
    """for iteration in range(4):
        plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=25)  # Titolo degli assi
        plt.rc('axes', labelsize=25)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda
        sns.set_style('whitegrid')
        sns.kdeplot(data=data, x='Train', fill=True, bw_adjust=.1, color='blue', label='Train')
        sns.kdeplot(data=data, x='Validation', fill=True, bw_adjust=.1, color='red', label='Val')

        x = 'Test ' + str(iteration)
        fill = True
        bw_adjust = .1
        color = 'green'

        if iteration == 0:
            label = 'Test high_perf'
            save_name = title.replace(' ', '_') + '_test_high_perf'
        if iteration == 1:
            label = 'Test mid_high_perf'
            save_name = title.replace(' ', '_') + '_test_mid_high_perf'
        if iteration == 2:
            label = 'Test mid_perf'
            save_name = title.replace(' ', '_') + '_test_mid_perf'
        if iteration == 3:
            label = 'Test mid_low_perf'
            save_name = title.replace(' ', '_') + '_test_mid_low_perf'

        sns.kdeplot(data=data, x=x, fill=fill, bw_adjust=bw_adjust, color=color, label=label)

        sns.despine()
        plt.xlabel(title)
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig(folder + 'test_' + str(iteration) + '/' + save_name + '.png', save_format='png')

        # plt.show()
        plt.close()"""

    # This following code is to be used in case we want to plot the kde only for training data, with no test data
    plt.figure(figsize=(19, 14))
    plt.rc('font', size=30)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=35)  # Titolo degli assi
    plt.rc('axes', labelsize=35)  # Etichette degli assi
    plt.rc('xtick', labelsize=30)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=30)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=25)  # Legenda
    sns.set_style('whitegrid')
    sns.kdeplot(data=data, x='Train', fill=True, bw_adjust=.1, color='blue', label='Train')
    sns.kdeplot(data=data, x='Val', fill=True, bw_adjust=.1, color='red', label='Validation')

    sns.despine()
    plt.xlabel(title, labelpad=20)
    plt.ylabel('Density', labelpad=20)
    plt.legend(loc='best')
    plt.title(title, pad=20)
    plt.savefig(folder + title + '.' + save_format, format=save_format)
    # plt.savefig(folder + 'test_' + str(iteration) + '/' + save_name + '.png', save_format='png')

    # plt.show()
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def show_kde_road_grip(title, data, folder):
    grip_levels = ['1', '0.8', '0.6']
    for i, grip_lev in enumerate(grip_levels):
        sns.set_style('whitegrid')
        sns.kdeplot(data=data, x='Train', fill=True, bw_adjust=.1, color='blue', label='Train')
        sns.kdeplot(data=data, x='Val', fill=True, bw_adjust=.1, color='red', label='Val')

        """x = 'Test ' + str(i)
        fill = True
        bw_adjust = .1
        color = 'green'

        label = 'Test road_μ = ' + grip_lev
        save_name = title.replace(' ', '_') + '_test_grip_' + grip_lev.replace('.', '')

        sns.kdeplot(data=data, x=x, fill=fill, bw_adjust=bw_adjust, color=color, label=label)"""

        sns.despine()
        plt.xlabel(title)
        plt.legend(loc='best')
        plt.title(title)
        plt.savefig(folder + title + '.' + save_format, format=save_format)
        # plt.savefig(folder + 'grip_' + grip_lev.replace('.', '') + '/' + savename + '.png', save_format='png')

        plt.show()
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------

def load_test_car_perf(path_):
    test_0_feat = np.loadtxt(path_ + 'test_set_0.csv', delimiter=',')
    test_1_feat = np.loadtxt(path_ + 'test_set_1.csv', delimiter=',')
    test_2_feat = np.loadtxt(path_ + 'test_set_2.csv', delimiter=',')
    test_3_feat = np.loadtxt(path_ + 'test_set_3.csv', delimiter=',')

    return test_0_feat, test_1_feat, test_2_feat, test_3_feat


# ----------------------------------------------------------------------------------------------------------------------

def load_test_road_grip(path_):
    test_0_feat = np.loadtxt(path_ + 'test_set_mu_1.csv', delimiter=',')
    test_1_feat = np.loadtxt(path_ + 'test_set_mu_08.csv', delimiter=',')
    test_2_feat = np.loadtxt(path_ + 'test_set_mu_06.csv', delimiter=',')

    return test_0_feat, test_1_feat, test_2_feat


# ----------------------------------------------------------------------------------------------------------------------

def prepare_list_test():
    vy_test, yaw_test, vx_test, delta_test, fx_test = [], [], [], [], []

    for idx in range(4):
        t1, t2, t3, t4, t5 = [], [], [], [], []
        vy_test.append(t1)
        vx_test.append(t2)
        yaw_test.append(t3)
        delta_test.append(t4)
        fx_test.append(t5)

    return vy_test, yaw_test, vx_test, delta_test, fx_test


# ----------------------------------------------------------------------------------------------------------------------

def create_df_car_perf(vy_train, yaw_train, vx_train, delta_train, fx_train,
                       vy_val, yaw_val, vx_val, delta_val, fx_val,
                       test_0_feat, test_1_feat, test_2_feat, test_3_feat,
                       vy_test, yaw_test, vx_test, delta_test, fx_test):
    yaw_test[0], yaw_test[1], yaw_test[2], yaw_test[3] = (
        test_0_feat[:, 0], test_1_feat[:, 0], test_2_feat[:, 0], test_3_feat[:, 0])
    vy_test[0], vy_test[1], vy_test[2], vy_test[3] = (
        test_0_feat[:, 1], test_1_feat[:, 1], test_2_feat[:, 1], test_3_feat[:, 1])
    vx_test[0], vx_test[1], vx_test[2], vx_test[3] = (
        test_0_feat[:, 2], test_1_feat[:, 2], test_2_feat[:, 2], test_3_feat[:, 2])
    delta_test[0], delta_test[1], delta_test[2], delta_test[3] = (
        test_0_feat[:, 3], test_1_feat[:, 3], test_2_feat[:, 3], test_3_feat[:, 3])
    fx_test[0], fx_test[1], fx_test[2], fx_test[3] = (
        test_0_feat[:, 4], test_1_feat[:, 4], test_2_feat[:, 4], test_3_feat[:, 4])

    # Normalize length
    vy_val.extend([np.nan] * (len(vy_train) - len(vy_val)))
    yaw_val.extend([np.nan] * (len(yaw_train) - len(yaw_val)))
    vx_val.extend([np.nan] * (len(vx_train) - len(vx_val)))
    delta_val.extend([np.nan] * (len(delta_train) - len(delta_val)))
    fx_val.extend([np.nan] * (len(fx_train) - len(fx_val)))

    vy_test = pad(vy_test, len(vy_train))
    vx_test = pad(vx_test, len(vx_train))
    yaw_test = pad(yaw_test, len(yaw_train))
    delta_test = pad(delta_test, len(delta_train))
    fx_test = pad(fx_test, len(fx_train))

    # Create Datasets for plotting
    df_vy = pd.DataFrame({'Train': vy_train, 'Val': vy_val, 'Test 0': vy_test[0],
                          'Test 1': vy_test[1], 'Test 2': vy_test[2], 'Test 3': vy_test[3]})
    df_vx = pd.DataFrame({'Train': vx_train, 'Val': vx_val, 'Test 0': vx_test[0],
                          'Test 1': vx_test[1], 'Test 2': vx_test[2], 'Test 3': vx_test[3]})
    df_yaw = pd.DataFrame({'Train': yaw_train, 'Val': yaw_val, 'Test 0': yaw_test[0],
                           'Test 1': yaw_test[1], 'Test 2': yaw_test[2], 'Test 3': yaw_test[3]})
    df_delta = pd.DataFrame({'Train': delta_train, 'Val': delta_val, 'Test 0': delta_test[0],
                             'Test 1': delta_test[1], 'Test 2': delta_test[2], 'Test 3': delta_test[3]})
    df_fx = pd.DataFrame({'Train': fx_train, 'Val': fx_val, 'Test 0': fx_test[0],
                          'Test 1': fx_test[1], 'Test 2': fx_test[2], 'Test 3': fx_test[3]})

    return df_vy, df_vx, df_yaw, df_delta, df_fx


# ----------------------------------------------------------------------------------------------------------------------

def create_df_road_grip(vy_train, yaw_train, vx_train, delta_train, fx_train,
                        vy_val, yaw_val, vx_val, delta_val, fx_val,
                        test_0_feat, test_1_feat, test_2_feat,
                        vy_test, yaw_test, vx_test, delta_test, fx_test):
    vy_test[0], vy_test[1], vy_test[2] = (
        test_0_feat[:, 1], test_1_feat[:, 1], test_2_feat[:, 1])
    yaw_test[0], yaw_test[1], yaw_test[2] = (
        test_0_feat[:, 0], test_1_feat[:, 0], test_2_feat[:, 0])
    vx_test[0], vx_test[1], vx_test[2] = (
        test_0_feat[:, 2], test_1_feat[:, 2], test_2_feat[:, 2])
    delta_test[0], delta_test[1], delta_test[2] = (
        test_0_feat[:, 3], test_1_feat[:, 3], test_2_feat[:, 3])
    fx_test[0], fx_test[1], fx_test[2] = (
        test_0_feat[:, 4], test_1_feat[:, 4], test_2_feat[:, 4])

    # Normalize length
    vy_val.extend([np.nan] * (len(vy_train) - len(vy_val)))
    yaw_val.extend([np.nan] * (len(yaw_train) - len(yaw_val)))
    vx_val.extend([np.nan] * (len(vx_train) - len(vx_val)))
    delta_val.extend([np.nan] * (len(delta_train) - len(delta_val)))
    fx_val.extend([np.nan] * (len(fx_train) - len(fx_val)))

    vy_test = pad(vy_test, len(vy_train))
    vx_test = pad(vx_test, len(vx_train))
    yaw_test = pad(yaw_test, len(yaw_train))
    delta_test = pad(delta_test, len(delta_train))
    fx_test = pad(fx_test, len(fx_train))

    # Create Datasets for plotting
    df_vy = pd.DataFrame({'Train': vy_train, 'Val': vy_val, 'Test 0': vy_test[0],
                          'Test 1': vy_test[1], 'Test 2': vy_test[2]})
    df_vx = pd.DataFrame({'Train': vx_train, 'Val': vx_val, 'Test 0': vx_test[0],
                          'Test 1': vx_test[1], 'Test 2': vx_test[2]})
    df_yaw = pd.DataFrame({'Train': yaw_train, 'Val': yaw_val, 'Test 0': yaw_test[0],
                           'Test 1': yaw_test[1], 'Test 2': yaw_test[2]})
    df_delta = pd.DataFrame({'Train': delta_train, 'Val': delta_val, 'Test 0': delta_test[0],
                             'Test 1': delta_test[1], 'Test 2': delta_test[2]})
    df_fx = pd.DataFrame({'Train': fx_train, 'Val': fx_val, 'Test 0': fx_test[0],
                          'Test 1': fx_test[1], 'Test 2': fx_test[2]})

    return df_vy, df_vx, df_yaw, df_delta, df_fx


# ----------------------------------------------------------------------------------------------------------------------

def kde_car_perf(selector: str, save_format: str):
    # Load data
    train_feat, val_feat = load_features('scirob_submission/Model_Learning/data/new/train_data_step_1_mu1.csv')
    # train_feat, val_feat = load_features('scirob_submission/Model_Learning/data/new/train_data_step_1_cplt.csv')
    print('Data length: ', str(len(train_feat) + len(val_feat)))
    folder = '../gg_plots/thesis_plots/density train_val/mu1/'

    if selector == 'car_perf':
        test_0_feat, test_1_feat, test_2_feat, test_3_feat = load_test_car_perf(
            'scirob_submission/Model_Learning/data/new/')

        # Prepare data
        vy_train, yaw_train, vx_train, delta_train, fx_train = fill_lists(train_feat)
        vy_val, yaw_val, vx_val, delta_val, fx_val = fill_lists(val_feat)
        vy_test, yaw_test, vx_test, delta_test, fx_test = prepare_list_test()

        df_vy, df_vx, df_yaw, df_delta, df_fx = create_df_car_perf(vy_train, yaw_train, vx_train, delta_train, fx_train,
                                                                   vy_val, yaw_val, vx_val, delta_val, fx_val,
                                                                   test_0_feat, test_1_feat, test_2_feat, test_3_feat,
                                                                   vy_test, yaw_test, vx_test, delta_test, fx_test)

        # Plotting and saving
        titles = ['Lateral_velocity', 'Longitudinal_velocity', 'Yaw_rate', 'Steering_angle', 'Longitudinal_force']
        sets = [df_vy, df_vx, df_yaw, df_delta, df_fx]

        for i, title in enumerate(titles):
            show_kde_car_perf(title, sets[i], folder, save_format)

    elif selector == 'road_grip':
        test_0_feat, test_1_feat, test_2_feat = load_test_road_grip('scirob_submission/Model_Learning/data/new/')

        # Prepare data
        vy_train, yaw_train, vx_train, delta_train, fx_train = fill_lists(train_feat)
        vy_val, yaw_val, vx_val, delta_val, fx_val = fill_lists(val_feat)
        vy_test, yaw_test, vx_test, delta_test, fx_test = prepare_list_test()

        df_vy, df_vx, df_yaw, df_delta, df_fx = create_df_road_grip(vy_train, yaw_train, vx_train, delta_train,
                                                                    fx_train,
                                                                    vy_val, yaw_val, vx_val, delta_val, fx_val,
                                                                    test_0_feat, test_1_feat, test_2_feat,
                                                                    vy_test, yaw_test, vx_test, delta_test, fx_test)

        # Plotting and saving
        titles = ['Lateral_velocity', 'Longitudinal_velocity', 'Yaw_rate', 'Steering_angle', 'Longitudinal_force']
        sets = [df_vy, df_vx, df_yaw, df_delta, df_fx]

        for i, title in enumerate(titles):
            show_kde_car_perf(title, sets[i], folder, save_format)


# ----------------------------------------------------------------------------------------------------------------------

def kde_road_grip(selector_test: str, save_format: str):
    # Load data
    train_feat, val_feat = load_features('scirob_submission/Model_Learning/data/new/train_data_step_1_mu06.csv')
    print('Data length: ', str(len(train_feat) + len(val_feat)))
    folder = '../gg_plots/thesis_plots/density train_val/road_grip/'

    if selector_test == 'car_perf':
        test_0_feat, test_1_feat, test_2_feat, test_3_feat = load_test_car_perf(
            'scirob_submission/Model_Learning/data/new/')

        # Prepare data
        vy_train, yaw_train, vx_train, delta_train, fx_train = fill_lists(train_feat)
        vy_val, yaw_val, vx_val, delta_val, fx_val = fill_lists(val_feat)
        vy_test, yaw_test, vx_test, delta_test, fx_test = prepare_list_test()

        df_vy, df_vx, df_yaw, df_delta, df_fx = create_df_car_perf(vy_train, yaw_train, vx_train, delta_train, fx_train,
                                                                   vy_val, yaw_val, vx_val, delta_val, fx_val,
                                                                   test_0_feat, test_1_feat, test_2_feat, test_3_feat,
                                                                   vy_test, yaw_test, vx_test, delta_test, fx_test)

        # Plotting and saving
        titles = ['Lateral_velocity', 'Longitudinal_velocity', 'Yaw_rate', 'Steering_angle', 'Longitudinal_force']
        sets = [df_vy, df_vx, df_yaw, df_delta, df_fx]

        for i, title in enumerate(titles):
            show_kde_car_perf(title, sets[i], folder, save_format)

    if selector_test == 'road_grip':
        test_0_feat, test_1_feat, test_2_feat = load_test_road_grip('scirob_submission/Model_Learning/data/new/')

        # Prepare data
        vy_train, yaw_train, vx_train, delta_train, fx_train = fill_lists(train_feat)
        vy_val, yaw_val, vx_val, delta_val, fx_val = fill_lists(val_feat)
        vy_test, yaw_test, vx_test, delta_test, fx_test = prepare_list_test()

        df_vy, df_vx, df_yaw, df_delta, df_fx = create_df_road_grip(vy_train, yaw_train, vx_train, delta_train,
                                                                    fx_train,
                                                                    vy_val, yaw_val, vx_val, delta_val, fx_val,
                                                                    test_0_feat, test_1_feat, test_2_feat,
                                                                    vy_test, yaw_test, vx_test, delta_test, fx_test)

        # Plotting and saving
        titles = ['Lateral_velocity', 'Longitudinal_velocity', 'Yaw_rate', 'Steering_angle', 'Longitudinal_force']
        sets = [df_vy, df_vx, df_yaw, df_delta, df_fx]

        for i, title in enumerate(titles):
            show_kde_road_grip(title, sets[i], folder, save_format)


# ----------------------------------------------------------------------------------------------------------------------

save_format = 'eps'
kde_car_perf('car_perf', save_format)
# kde_road_grip('car_perf')

"""
loaded_model = tf.keras.models.load_model('scirob_submission/Model_Learning/saved_models/step_1/callbacks/
                                           2024_05_27/08_38_04/keras_model.h5')

loaded_model.summary()
print(loaded_model.to_json())"""
