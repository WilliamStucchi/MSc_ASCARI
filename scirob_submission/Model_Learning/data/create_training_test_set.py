import math

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm

# ----------------------------------------------------------------------------------------------------------------------

def plot_delta_steps(path_to_data, number_of_sets):
    delta_fx = []
    delta_steer = []

    yaw_rate_max = 0
    max_vx = float('-inf')
    max_delta = float('-inf')
    min_fx = float('inf')
    max_fx = float('-inf')

    """directories = ['/train_car_perf/mid_perf/', '/train_car_perf/mid_high_perf/', '/train_car_perf/high_perf/',
                   '/train_car_perf/mid_low_perf/']"""
    directories = ['train_road_grip/grip_06/', 'train_road_grip/grip_06_perf_045/', 'train_road_grip/grip_06_perf_03/']
    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here
    filename = 'DemoSportsCar_mxp.csv'

    for k, dir_ in enumerate(directories):
        if 'const_speed' in dir_:
            number_of_sets = 9
            filename = 'DemoSportsCar_mnt.csv'

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            for idx in range(1, len(data)):
                if float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx]) > yaw_rate_max:
                    yaw_rate_max = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
                if float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx]) > max_vx:
                    max_vx = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
                if float(data['driver_demands.steering'][idx]) > max_delta:
                    max_delta = float(data['driver_demands.steering'][idx])

                frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
                frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
                fr = (frl + frr) / 2
                ffl = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
                ffr = float(data['Tire.Ground_Surface_Force_X.R1'][idx])
                ff = (ffl + ffr) / 2
                fx = (ff + fr) / 2

                if fx > max_fx:
                    max_fx = fx
                elif fx < min_fx:
                    min_fx = fx

                # Delta Fx
                prev_fr = (float(data['Tire.Ground_Surface_Force_X.L2'][idx - 1]) +
                           float(data['Tire.Ground_Surface_Force_X.R2'][idx - 1])) / 2
                prev_ff = (float(data['Tire.Ground_Surface_Force_X.L1'][idx - 1]) +
                           float(data['Tire.Ground_Surface_Force_X.R1'][idx - 1])) / 2
                prev_fx = (prev_ff + prev_fr) / 2

                curr_fr = (float(data['Tire.Ground_Surface_Force_X.L2'][idx]) +
                           float(data['Tire.Ground_Surface_Force_X.R2'][idx])) / 2
                curr_ff = (float(data['Tire.Ground_Surface_Force_X.L1'][idx]) +
                           float(data['Tire.Ground_Surface_Force_X.R1'][idx])) / 2
                curr_fx = (curr_ff + curr_fr) / 2
                
                delta_fx.append(curr_fx - prev_fx)

                # Delta Steer
                # 15.56 is for the conversion between steering wheel angle and tire steering angle
                prev_steer = float(data['driver_demands.steering'][idx - 1]) / 15.56
                curr_steer = float(data['driver_demands.steering'][idx]) / 15.56
                delta_steer.append(curr_steer - prev_steer)

    print('Yaw rate max = ', yaw_rate_max)
    print('Vx max = ', max_vx)
    print('Steer max = ', max_delta)
    print('Fx max = ', max_fx)
    print('Fx min = ', min_fx)

    media_fx = np.mean(delta_fx)
    varianza_fx = np.var(delta_fx)
    fx_df = pd.DataFrame({'Delta': delta_fx})

    media_steer = np.mean(delta_steer)
    varianza_steer = np.var(delta_steer)
    steer_df = pd.DataFrame({'Delta': delta_steer})

    # ----------------------------
    # Fx
    # ----------------------------
    mean, std_dev = norm.fit(delta_fx)

    # Istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(delta_fx, bins=100, density=True, color='#0000FF', edgecolor='#0000FF')
    plt.title('Histogram of deltaFx [Fx(t) - Fx(t-1)]', pad=20)
    plt.title('Histogram of deltaFx [Fx(t) - Fx(t-1)]', pad=20)
    plt.xlabel('[Fx(t) - Fx(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20),
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_distribution_mu06.png', format='png', dpi=300)
    plt.close()
    input('wait')

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(fx_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.yscale('log')
    plt.title('Histogram of deltaFx [Fx(t) - Fx(t-1)]', pad=20)
    plt.xlabel('[Fx(t) - Fx(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_distribution_logy_mu06.png', format='png', dpi=300)
    plt.close()

    # Create a box plot
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    sns.boxplot(data=fx_df, x='Delta')
    plt.title('Box Plot of DeltaFx')
    plt.xlabel('[Fx(t) - Fx(t-1)] values')
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_boxplot_mu06.png', format='png', dpi=300)
    plt.close()


    """for j in range(len(n)):
        print(f"Bin {j + 1} (da {bins[j]:.2f} a {bins[j+1]:.2f}): {n[j]} valori")"""

    dati_per_excel = {
        'Descrizione': [
            'Numero di elementi unici in deltaFx',
            'Media Fx',
            'Varianza Fx'
        ],
        'Valore': [
            len(np.unique(delta_fx)),
            media_fx,
            varianza_fx
        ]
    }

    df = pd.DataFrame(dati_per_excel)

    for j in range(len(n)):
        df = df._append({'Descrizione': f'Bin {j + 1} (da {bins[j]:.2f} a {bins[j + 1]:.2f})', 'Valore': n[j]},
                       ignore_index=True)

    # Esportare il DataFrame in un file Excel
    df.to_excel('../../../../test/deltafx_deltasteer/dati_output_fx_mu06.xlsx', index=False)

    z_scores_fx = stats.zscore(delta_fx)
    threshold = 3
    mask_zscore_fx = np.abs(z_scores_fx) < threshold
    delta_fx = np.array(delta_fx)
    delta_fx_cleaned = delta_fx[mask_zscore_fx]
    fx_df = pd.DataFrame({'Delta': delta_fx_cleaned})

    # Istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(fx_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.title('Histogram of deltaFx [Fx(t) - Fx(t-1)]', pad=20)
    plt.xlabel('[Fx(t) - Fx(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_distribution_zscore_mu06.png', format='png', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(fx_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.yscale('log')
    plt.title('Histogram of deltaFx [Fx(t) - Fx(t-1)]', pad=20)
    plt.xlabel('[Fx(t) - Fx(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_distribution_zscore_logy_mu06.png', format='png', dpi=300)
    plt.close()

    # Create a box plot
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    sns.boxplot(data=fx_df, x='Delta')
    plt.title('Box Plot of DeltaFx after zscore')
    plt.xlabel('[Fx(t) - Fx(t-1)] values')
    plt.savefig('../../../../test/deltafx_deltasteer/deltaFx_zscore_boxplot_mu06.png', format='png', dpi=300)
    plt.close()

    media_fx_zscore = np.mean(delta_fx_cleaned)
    varianza_fx_zscore = np.var(delta_fx_cleaned)
    mean, std_dev = norm.fit(delta_fx_cleaned)

    dati_per_excel = {
        'Descrizione': [
            'Numero di elementi unici in deltaFx',
            'Media Fx',
            'Varianza Fx'
        ],
        'Valore': [
            len(np.unique(delta_fx_cleaned)),
            media_fx_zscore,
            varianza_fx_zscore
        ]
    }

    df = pd.DataFrame(dati_per_excel)

    for j in range(len(n)):
        df = df._append({'Descrizione': f'Bin {j + 1} (da {bins[j]:.2f} a {bins[j + 1]:.2f})', 'Valore': n[j]},
                        ignore_index=True)

    # Esportare il DataFrame in un file Excel
    df.to_excel('../../../../test/deltafx_deltasteer/dati_output_fx_zscore_mu06.xlsx', index=False)

    # ----------------------------
    # Steer
    # ----------------------------

    # Istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(steer_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.title('Histogram of deltaSteer [δ(t) - δ(t-1)]', pad=20)
    plt.xlabel('[δ(t) - δ(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_distribution_mu06.png', format='png', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(steer_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.yscale('log')
    plt.title('Histogram of deltaSteer [δ(t) - δ(t-1)]', pad=20)
    plt.xlabel('[δ(t) - δ(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_distribution_logy_mu06.png', format='png', dpi=300)
    plt.close()

    # Create a box plot
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    sns.boxplot(data=steer_df, x='Delta')
    plt.title('Box Plot of DeltaSteer')
    plt.xlabel('[δ(t) - δ(t-1)] values')
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_boxplot_mu06.png', format='png', dpi=300)
    plt.close()

    """for j in range(len(n)):
        print(f"Bin {j + 1} (da {bins[j]:.2f} a {bins[j + 1]:.2f}): {n[j]} valori")"""

    dati_per_excel = {
        'Descrizione': [
            'Numero di elementi unici in deltaSteer',
            'Media steer',
            'Varianza steer'
        ],
        'Valore': [
            len(np.unique(delta_steer)),
            media_steer,
            varianza_steer
        ]
    }

    df = pd.DataFrame(dati_per_excel)

    for j in range(len(n)):
        df = df._append({'Descrizione': f'Bin {j + 1} (da {bins[j]:.2f} a {bins[j + 1]:.2f})', 'Valore': n[j]},
                       ignore_index=True)

    # Esportare il DataFrame in un file Excel
    df.to_excel('../../../../test/deltafx_deltasteer/dati_output_steer_mu06.xlsx', index=False)

    z_scores_steer = stats.zscore(delta_steer)
    threshold = 3
    mask_zscore_steer = np.abs(z_scores_steer) < threshold
    delta_steer = np.array(delta_steer)
    delta_steer_cleaned = delta_steer[mask_zscore_steer]
    steer_df = pd.DataFrame({'Delta': delta_steer_cleaned})

    # Istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(steer_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.title('Histogram of deltaSteer [δ(t) - δ(t-1)]', pad=20)
    plt.xlabel('[δ(t) - δ(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_distribution_zscore_mu06.png', format='png', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    n, bins, _ = plt.hist(steer_df['Delta'], bins=100, color='#0000FF', edgecolor='#0000FF')
    plt.yscale('log')
    plt.title('Histogram of deltaSteer [δ(t) - δ(t-1)]', pad=20)
    plt.xlabel('[δ(t) - δ(t-1)] values', labelpad=20)
    plt.ylabel('# elements per bin', labelpad=20)
    plt.grid()
    # plt.show()
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_distribution_zscore_logy_mu06.png', format='png', dpi=300)
    plt.close()

    # Create a box plot
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    sns.boxplot(data=steer_df, x='Delta')
    plt.title('Box Plot of DeltaSteer after zscore')
    plt.xlabel('[δ(t) - δ(t-1)] values')
    plt.savefig('../../../../test/deltafx_deltasteer/deltaSteer_zscore_boxplot_mu06.png', format='png', dpi=300)
    plt.close()

    media_steer_zscore = np.mean(delta_steer_cleaned)
    varianza_steer_zscore = np.var(delta_steer_cleaned)

    dati_per_excel = {
        'Descrizione': [
            'Numero di elementi unici in deltaSteer',
            'Media steer',
            'Varianza steer'
        ],
        'Valore': [
            len(np.unique(delta_steer_cleaned)),
            media_steer_zscore,
            varianza_steer_zscore
        ]
    }

    df = pd.DataFrame(dati_per_excel)

    for j in range(len(n)):
        df = df._append({'Descrizione': f'Bin {j + 1} (da {bins[j]:.2f} a {bins[j + 1]:.2f})', 'Valore': n[j]},
                        ignore_index=True)

    # Esportare il DataFrame in un file Excel
    df.to_excel('../../../../test/deltafx_deltasteer/dati_output_steer_zscore_mu06.xlsx', index=False)


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                 vy_estimation=False, include_grip=False):
    len_mu1_perf100 = 0
    len_mu1_perf75 = 0
    len_mu1_perf50 = 0
    len_mu06_perf100 = 0
    len_mu06_perf75 = 0
    len_mu06_perf50 = 0
    enter = False

    input_shape = 5
    output_shape = 3
    sum_ = 0

    name_file = 'balanced'
    directories = ['/train_road_grip/grip_06/', '/train_car_perf/high_perf/', '/train_car_perf/mid_high_perf/', 
                   '/train_car_perf/mid_perf/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/']

    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here
    filename = 'DemoSportsCar_mxp.csv'

    result = []
    for i in range(number_of_sets_ * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        if 'const_speed' in dir_:
            number_of_sets = 9
            filename = 'DemoSportsCar_mnt.csv'

        for i in tqdm(range(0, number_of_sets)):
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            if '/train_road_grip/grip_06/' in dir_:
                len_mu06_perf100 += len(data)
                enter = True

            elif '/train_car_perf/high_perf/' in dir_:
                if len_mu1_perf100 + len(data) < int(np.floor(len_mu06_perf100 / 2)):
                    enter = True
                    len_mu1_perf100 += len(data)
                elif len_mu1_perf100 >= int(np.floor(len_mu06_perf100 / 2)):
                    enter = False
                    print('Dati generati da /train_car_perf/high_perf/: ' + str(len_mu1_perf100) + ' su '
                          + str(len_mu06_perf100) + ' [' + str(len_mu1_perf100/len_mu06_perf100) + ']')
                else:
                    data = data[:(int(np.floor(len_mu06_perf100 / 2)) - len_mu1_perf100)]
                    len_mu1_perf100 += len(data)
                    enter = True

            elif '/train_car_perf/mid_high_perf/' in dir_:
                if len_mu1_perf75 + len(data) < int(np.floor(len_mu06_perf100 / 2)):
                    enter = True
                    len_mu1_perf75 += len(data)
                elif len_mu1_perf75 >= int(np.floor(len_mu06_perf100 / 2)):
                    enter = False
                    print('Dati generati da /train_car_perf/mid_high_perf/: ' + str(len_mu1_perf75) + ' su '
                          + str(len_mu06_perf100) + ' [' + str(len_mu1_perf75 / len_mu06_perf100) + ']')
                else:
                    data = data[:(int(np.floor(len_mu06_perf100 / 2)) - len_mu1_perf75)]
                    len_mu1_perf75 += len(data)
                    enter = True

            elif '/train_car_perf/mid_perf/' in dir_:
                if len_mu1_perf50 + len(data) < int(np.floor(len_mu06_perf100 / 2)):
                    enter = True
                    len_mu1_perf50 += len(data)
                elif len_mu1_perf50 >= int(np.floor(len_mu06_perf100 / 2)):
                    enter = False
                    print('Dati generati da /train_car_perf/mid_perf/: ' + str(len_mu1_perf50) + ' su '
                          + str(len_mu06_perf100) + ' [' + str(len_mu1_perf50 / len_mu06_perf100) + ']')
                else:
                    data = data[:(int(np.floor(len_mu06_perf100 / 2)) - len_mu1_perf50)]
                    len_mu1_perf50 += len(data)
                    enter = True

            elif '/train_road_grip/grip_06_perf_045/' in dir_:
                if len_mu06_perf75 + len(data) < int(np.floor(len_mu06_perf100 / 4)):
                    enter = True
                    len_mu06_perf75 += len(data)
                elif len_mu06_perf75 >= int(np.floor(len_mu06_perf100 / 4)):
                    enter = False
                    print('Dati generati da /train_road_grip/grip_06_perf_045/: ' + str(len_mu06_perf75) + ' su '
                          + str(len_mu06_perf100) + ' [' + str(len_mu06_perf75 / len_mu06_perf100) + ']')
                else:
                    data = data[:(int(np.floor(len_mu06_perf100 / 4)) - len_mu06_perf75)]
                    len_mu06_perf75 += len(data)
                    enter = True

            elif '/train_road_grip/grip_06_perf_03/' in dir_:
                if len_mu06_perf50 + len(data) < int(np.floor(len_mu06_perf100 / 4)):
                    enter = True
                    len_mu06_perf50 += len(data)
                elif len_mu06_perf50 >= int(np.floor(len_mu06_perf100 / 4)):
                    enter = False
                    print('Dati generati da /train_road_grip/grip_06_perf_03/: ' + str(len_mu06_perf50) + ' su '
                          + str(len_mu1_perf100) + ' [' + str(len_mu06_perf50 / len_mu06_perf100) + ']')
                else:
                    data = data[:(int(np.floor(len_mu06_perf100 / 4)) - len_mu06_perf50)]
                    len_mu06_perf50 += len(data)
                    enter = True

            if enter:
                # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
                sum_ += len(data)
                # input('Wait')

                if not include_grip:
                    result[k * number_of_sets_ + i] = np.zeros(
                        (len(data),
                         input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives
                else:
                    result[k * number_of_sets_ + i] = np.zeros(
                        (len(data),
                         input_shape * timesteps_back + output_shape + 1))  # 5 inputs, 4 steps in the past, 4 objectives

                vy_est_prev = 0.0  # in case of estimation of vy
                vy_est = 0.0
                count = 0
                for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                    # print(idx)
                    for j in range(0, timesteps_back):
                        yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                    yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                    ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])
                    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                    if vy_estimation:
                        for obj in range(0, timesteps_back):
                            result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]

                            # vy_dot = ay - yaw_rate * vx
                            ay_obj = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + obj])
                            vy_dot = ay_obj - yaw_rate[obj] * ux[obj]

                            if obj == 0:
                                vy_est = vy_dot * 0.01 + vy_est_prev
                                vy_est_prev = vy_est
                            else:
                                vy_est = vy_dot * 0.01 + vy_est

                            result[k * number_of_sets_ + i][count, obj * input_shape + 1] = vy_est
                            result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                        result[k * number_of_sets_ + i][count, -3] = yaw_acc
                        result[k * number_of_sets_ + i][count, -2] = ay
                        result[k * number_of_sets_ + i][count, -1] = ax

                    else:
                        for obj in range(0, timesteps_back):
                            result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                            result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                        result[k * number_of_sets_ + i][count, -3] = yaw_acc
                        result[k * number_of_sets_ + i][count, -2] = ay
                        result[k * number_of_sets_ + i][count, -1] = ax

                        if include_grip:
                            if 'grip_06' in dir_:
                                result[k * number_of_sets_ + i][count, -4] = 0.6
                            elif 'grip_08' in dir_:
                                result[k * number_of_sets_ + i][count, -4] = 0.8
                            elif 'grip_' not in dir_:
                                result[k * number_of_sets_ + i][count, -4] = 1.0

                    count += 1

                    if idx < (len(data) - (timesteps_back + 1)):
                        ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                    else:
                        for j in range(0, timesteps_back):
                            ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)


        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_' + name_file + '.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_' + name_file + '.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_' + name_file + '.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    """# Load bicycle data
    bicycle_data = np.loadtxt('../data/new/bicycle_model_2grip.csv', delimiter=',')
    result_flat = np.concatenate((result_flat, bicycle_data), axis=0)"""

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_' + name_file + '.csv', result_filtered,
               delimiter=',')

    # Shuffle input
    print('SHUFFLING')
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_' + name_file + '.csv', result_filtered, delimiter=',')
    print('Saving training at: ' + path_to_output + 'train_data_step_' + str(step) + '_' + name_file + '.csv')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))

    print('len_mu1_perf100 = ' + str(len_mu1_perf100))
    print('len_mu1_perf75 = ' + str(len_mu1_perf75))
    print('len_mu1_perf50 = ' + str(len_mu1_perf50))
    print('len_mu06_perf100 = ' + str(len_mu06_perf100))
    print('len_mu06_perf75 = ' + str(len_mu06_perf75))
    print('len_mu06_perf50 = ' + str(len_mu06_perf50))

def create_training_set_car_perf(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                 vy_estimation=False, include_grip=False):
    min_r = float('inf')
    max_r = float('-inf')
    min_vx = float('inf')
    max_vx = float('-inf')
    min_vy = float('inf')
    max_vy = float('-inf')
    min_delta = float('inf')
    max_delta = float('-inf')
    min_fx = float('inf')
    max_fx = float('-inf')
    last_delta = float('-inf')
    max_delta_difference = float('-inf')
    last_Fx = float('-inf')
    max_Fx_difference = float('-inf')
    file_max_delta = ''
    file_max_fx = ''
    pos_max_delta = ''
    pos_max_fx = ''
    couple_delta = ''
    couple_fx = ''

    input_shape = 5
    output_shape = 3

    sum_ = 0
    namefile = 'mu106'
    directories = ['/train_car_perf/high_perf/', '/train_car_perf/mid_high_perf/', '/train_car_perf/mid_perf/',
                   '/train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/']
    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here
    filename = 'DemoSportsCar_mxp.csv'

    result = []
    for i in range(number_of_sets_ * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        if 'const_speed' in dir_:
            number_of_sets = 9
            filename = 'DemoSportsCar_mnt.csv'

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            if not include_grip:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives
            else:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape + 1))  # 5 inputs, 4 steps in the past, 4 objectives

            vy_est_prev = 0.0  # in case of estimation of vy
            vy_est = 0.0
            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                # print(idx)
                for j in range(0, timesteps_back):
                    yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])
                ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                if min_r > yaw_rate[0]: min_r = yaw_rate[0]
                if max_r < yaw_rate[0]: max_r = yaw_rate[0]
                if min_vx > ux[0]: min_vx = ux[0]
                if max_vx < ux[0]: max_vx = ux[0]
                if min_vy > uy[0]: min_vy = uy[0]
                if max_vy < uy[0]: max_vy = uy[0]
                if min_delta > steer[0]: min_delta = steer[0]
                if max_delta < steer[0]: max_delta = steer[0]
                if min_fx > fx[0]: min_fx = fx[0]
                if max_fx < fx[0]: max_fx = fx[0]

                for y in range(0, timesteps_back):
                    if last_delta != float('-inf'):
                        if abs(steer[y] - last_delta) > max_delta_difference:
                            max_delta_difference = abs(steer[y] - last_delta)
                            file_max_delta = dir_ + 'test_' + str(i)
                            pos_max_delta = str(idx + (y-1)) + ' vs ' + str(idx + y)
                            couple_delta = str(steer[y]) + ' - ' + str(last_delta)
                    last_delta = steer[y]

                    if last_Fx != float('-inf'):
                        if abs(fx[y] - last_Fx) > max_Fx_difference:
                            max_Fx_difference = abs(fx[y] - last_Fx)
                            file_max_fx = dir_ + 'test_' + str(i)
                            pos_max_fx = str(idx + (y-1)) + ' vs ' + str(idx + y)
                            couple_fx = str(fx[y]) + ' - ' + str(last_Fx)
                    last_Fx = fx[y]

                last_delta = steer[1]
                last_Fx = fx[1]

                if vy_estimation:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]

                        # vy_dot = ay - yaw_rate * vx
                        ay_obj = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + obj])
                        vy_dot = ay_obj - yaw_rate[obj] * ux[obj]

                        if obj == 0:
                            vy_est = vy_dot * 0.01 + vy_est_prev
                            vy_est_prev = vy_est
                        else:
                            vy_est = vy_dot * 0.01 + vy_est

                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = vy_est
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                else:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                    if include_grip:
                        if 'grip_06' in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 0.6
                        elif 'grip_08' in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 0.8
                        elif 'grip_' not in dir_:
                            result[k * number_of_sets_ + i][count, -4] = 1.0

                count += 1

                if idx < (len(data) - (timesteps_back + 1)):
                    ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                else:
                    for j in range(0, timesteps_back):
                        ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

            last_delta = float('-inf')
            last_Fx = float('-inf')


        print('END CREATION TRAINING DATA ' + dir_)

    print('MIN_r: ', min_r)
    print('MAX_r: ', max_r)
    print('MIN_vx: ', min_vx)
    print('MAX_vx: ', max_vx)
    print('MIN_vy: ', min_vy)
    print('MAX_vy: ', max_vy)
    print('MIN_delta: ', min_delta)
    print('MAX_delta: ', max_delta)
    print('MIN_fx: ', min_fx)
    print('MAX_fx: ', max_fx)
    print('MAX_DIFF_DELTA: ', max_delta_difference)
    print('POS_MAX_DELTA: ' + file_max_delta + ' -> IDX: ' + str(pos_max_delta) + ' -> ' + couple_delta)
    print('MAX_DIFF_FX: ', max_Fx_difference)
    print('POS_MAX_FX: ' + file_max_fx + ' -> IDX: ' + str(pos_max_fx) + ' -> ' + couple_fx)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_' + namefile + '.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_' + namefile + '.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_' + namefile + '.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    """# Load bicycle data
    bicycle_data = np.loadtxt('../data/new/bicycle_model_2grip.csv', delimiter=',')
    result_flat = np.concatenate((result_flat, bicycle_data), axis=0)"""

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_' + namefile + '.csv', result_filtered,
               delimiter=',')

    # Shuffle input
    print('SHUFFLING')
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_' + namefile + '.csv', result_filtered, delimiter=',')
    print('Saving training at: ' + path_to_output + 'train_data_step_' + str(step) + '_' + namefile + '.csv')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_scheduled_sampling(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4):
    input_shape = 5
    output_shape = 3

    sum_ = 0
    directories = ['/train_car_perf/mid_low_perf/', '/train_car_perf/mid_perf/', '/train_car_perf/mid_high_perf/',
                   '/train_car_perf/high_perf/', '/train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/', '/train_road_grip/grip_08/',
                   '/train_road_grip/grip_08_perf_04/', '/train_road_grip/grip_08_perf_06/']
    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here

    result = []
    for i in range(number_of_sets_ * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # ay threshold for selection of samples
            data['Vehicle_States.lateral_acc_wrt_road'] = pd.to_numeric(data['Vehicle_States.lateral_acc_wrt_road'],
                                                                        errors='coerce')
            th_ay = (data['Vehicle_States.lateral_acc_wrt_road']).abs().max() * 0.85

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            result[k * number_of_sets_ + i] = np.zeros(
                (len(data), input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives

            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])

                if abs(ay) >= th_ay:
                    yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                    for j in range(0, timesteps_back):
                        yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                    count += 1

                    if idx < (len(data) - (timesteps_back + 1)):
                        ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                    else:
                        for j in range(0, timesteps_back):
                            ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_sched_cplt.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay_' + str(step) + '_sched_cplt.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_sched_cplt.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_sched_cplt.csv', result_filtered,
               delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set_road_grip(path_to_data, path_to_output, number_of_sets, step, timesteps_back=4,
                                  no_vy=False, vy_estimation=False):
    balancing_len = 0  # max len of data based on shortest directory
    balanced_len = 0  # len taken from each directory that must match balanced_len

    if no_vy:
        input_shape = 5 + 2
    else:
        input_shape = 5
    output_shape = 3

    number_of_sets_ = number_of_sets  # fixed number of sets for when I want to change the number of sets manually here

    sum_ = 0
    directories = ['train_car_perf/high_perf/', 'train_car_perf/mid_high_perf/', 'train_car_perf/mid_perf/',
                   'train_road_grip/grip_06/', '/train_road_grip/grip_06_perf_045/',
                   '/train_road_grip/grip_06_perf_03/',
                   'train_road_grip/grip_08/', 'train_road_grip/grip_08_perf_06/', 'train_road_grip/grip_08_perf_04/']

    result = []
    for i in range(number_of_sets * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(timesteps_back)
        uy = np.zeros(timesteps_back)
        ux = np.zeros(timesteps_back)
        steer = np.zeros(timesteps_back)
        fx = np.zeros(timesteps_back)

        balanced_len = 0
        if k == 1:
            print('BALANCING ', balancing_len)

        for i in tqdm(range(0, number_of_sets)):
            # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
            data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            # computation of len for balancing the sets
            if k == 0:
                balancing_len += len(data)
            elif balanced_len + len(data) <= balancing_len:
                balanced_len += len(data)
            elif balanced_len + len(data) > balancing_len and balanced_len < balancing_len:
                print('BEFORE ', len(data))
                data.drop(data.tail(balancing_len - balanced_len).index, inplace=True)
                print('AFTER ', len(data))
                balanced_len += len(data)
            else:
                print('BALANCED ', balanced_len)
                break

            ay_to_plot = np.zeros(len(data))
            ax_to_plot = np.zeros(len(data))

            # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
            sum_ += len(data)
            # input('Wait')

            if no_vy:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives
            else:
                result[k * number_of_sets_ + i] = np.zeros(
                    (len(data),
                     input_shape * timesteps_back + output_shape))  # 5 inputs, 4 steps in the past, 3 objectives

            vy_est_prev = 0.0
            vy_est = 0.0
            count = 0
            for idx in range(0, len(data) - (timesteps_back + 1) + 1, step):
                # print(idx)
                for j in range(0, timesteps_back):
                    yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

                yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + timesteps_back])
                ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + timesteps_back])
                ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + timesteps_back])

                if vy_estimation:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]

                        # vy_dot = ay - yaw_rate * vx
                        ay_obj = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + obj])
                        vy_dot = ay_obj - yaw_rate[obj] * ux[obj]

                        if obj == 0:
                            vy_est = vy_dot * 0.01 + vy_est_prev
                            vy_est_prev = vy_est
                        else:
                            vy_est = vy_dot * 0.01 + vy_est

                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = vy_est
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                else:
                    for obj in range(0, timesteps_back):
                        result[k * number_of_sets_ + i][count, obj * input_shape] = yaw_rate[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 1] = uy[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 2] = ux[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 3] = steer[obj]
                        result[k * number_of_sets_ + i][count, obj * input_shape + 4] = fx[obj]

                    result[k * number_of_sets_ + i][count, -3] = yaw_acc
                    result[k * number_of_sets_ + i][count, -2] = ay
                    result[k * number_of_sets_ + i][count, -1] = ax

                count += 1

                if idx < (len(data) - (timesteps_back + 1)):
                    ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
                else:
                    for j in range(0, timesteps_back):
                        ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_cplt.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay' + str(step) + '_cplt.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_cplt.csv', result_flat,
               delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step_no_shuffle_' + str(step) + '_cplt.csv',
               result_filtered,
               delimiter=',')

    print('SHUFFLING')
    # Shuffle input
    np.random.seed(1)
    np.random.shuffle(result_filtered)
    np.savetxt(path_to_output + 'train_data_step_' + str(step) + '_cplt.csv', result_filtered,
               delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------


def create_training_set_mass(path_to_data, path_to_output, number_of_sets, step):
    sum_ = 0
    directories = ['test_mass/']
    result = []
    for i in range(number_of_sets * len(directories)):
        temp = []
        result.append(temp)

    print('CREATION TRAINING DATA')
    for k, dir_ in enumerate(directories):
        print('CREATION TRAINING DATA ' + dir_)
        yaw_rate = np.zeros(4)
        uy = np.zeros(4)
        ux = np.zeros(4)
        steer = np.zeros(4)
        fx = np.zeros(4)

        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data = pd.read_csv(path_to_data + dir_ + 'test_2/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        ay_to_plot = np.zeros(len(data))
        ax_to_plot = np.zeros(len(data))

        # print(dir_ + 'test_' + str(i) + ' has len ' + str(len(data)))
        sum_ += len(data)
        # input('Wait')

        result[k * number_of_sets + i] = np.zeros((len(data), 5 * 4 + 3))  # 5 inputs, 4 steps in the past, 3 objectives

        count = 0
        for idx in range(0, len(data) - 5 + 1, step):
            # print(idx)
            for j in range(0, 4):
                yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data, idx + j)

            yaw_acc = float(data['Vehicle_States.yaw_angular_acc_wrt_road'][idx + 4])
            ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx + 4])
            ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx + 4])

            result[k * number_of_sets + i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                                                     yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                                                     yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                                                     yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                                                     yaw_acc, ay, ax]
            count += 1

            if idx < (len(data) - 5):
                ax_to_plot[idx], ay_to_plot[idx] = pick_acc_idx(data, idx)
            else:
                for j in range(0, 4):
                    ax_to_plot[j], ay_to_plot[j] = pick_acc_idx(data, idx + j)

        print('END CREATION TRAINING DATA ' + dir_)

    print('SAVING ACCELERATIONS')
    np.savetxt(path_to_output + 'train_ax_' + str(step) + '_mass.csv', ax_to_plot, delimiter=',')
    np.savetxt(path_to_output + 'train_ay' + str(step) + '_mass.csv', ay_to_plot, delimiter=',')

    print('FLATTENING')
    # Put together all data extracted
    result_flat = flatten_extend(result)
    np.savetxt(path_to_output + 'train_data_step_flat_' + str(step) + '_mass.csv', result_flat, delimiter=',')

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]
    np.savetxt(path_to_output + 'train_data_step' + str(step) + '_mass.csv', result_filtered, delimiter=',')

    print('END CREATION TRAINING DATA')
    print('sum = ' + str(sum_))


# ----------------------------------------------------------------------------------------------------------------------

def create_training_set_v1(path_to_data, path_to_output, number_of_sets, step):
    data = [None] * number_of_sets
    result = [None] * number_of_sets
    yaw_rate = np.zeros(4)
    uy = np.zeros(4)
    ux = np.zeros(4)
    steer = np.zeros(4)
    fx = np.zeros(4)

    for i in tqdm(range(0, number_of_sets)):
        # print('Opening: ' + path_to_data + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv')
        data[i] = pd.read_csv(path_to_data + '/flat/test_' + str(i + 1) + '/DemoSportsCar_mxp.csv', dtype=object)
        data[i] = data[i].drop(0, axis='rows')  # remove the row containing the measure units
        data[i].reset_index(drop=True, inplace=True)

        # print(len(data[i]))
        # input('Wait')

        result[i] = np.zeros((len(data[i]), 5 * 4 + 3))  # 5 inputs, 4 steps in the past, 3 objectives

        count = 0
        for idx in range(0, len(data[i]) - 5 + 1, step):
            # print(idx)
            for j in range(0, 4):
                yaw_rate[j], uy[j], ux[j], steer[j], fx[j] = pick_data_idx(data[i], idx + j)

            yaw_t = float(data[i]['Vehicle_States.yaw_angular_vel_wrt_road'][idx + 4])
            uy_t = float(data[i]['Vehicle_States.lateral_vel_wrt_road'][idx + 4])
            ux_t = float(data[i]['Vehicle_States.longitudinal_vel_wrt_road'][idx + 4])

            result[i][count] = [yaw_rate[0], uy[0], ux[0], steer[0], fx[0],
                                yaw_rate[1], uy[1], ux[1], steer[1], fx[1],
                                yaw_rate[2], uy[2], ux[2], steer[2], fx[2],
                                yaw_rate[3], uy[3], ux[3], steer[3], fx[3],
                                yaw_t, uy_t, ux_t]
            count += 1

    # Put together all data extracted
    result_flat = flatten_extend(result)

    # Flatten to obtain a list of lists (200000+, 23)
    result_flat = np.array(result_flat)

    # Remove all lists of all-zero elements
    result_filtered = result_flat[~np.all(result_flat == 0, axis=1)]

    # Shuffle and Divide into datasets
    # train, dev, test = shuffle_and_divide(data, Param)
    train, dev, test = shuffle_and_divide(result_filtered, 0.8, 0.15)

    # Write out the first generated dataset , dev=dev, test=test
    np.savez(path_to_output + 'train_data_step' + str(step),
             train_f=train[0],
             train_t=train[1],
             dev_f=dev[0],
             dev_t=dev[1],
             test_f=test[0],
             test_t=test[1])


# ----------------------------------------------------------------------------------------------------------------------

def shuffle_and_divide(gen_data, train_perc, dev_perc):
    # first shuffle the data generated
    # This is necessary for the two friction data.
    np.random.seed(1)
    np.random.shuffle(gen_data)

    # split into features and targets
    features = gen_data[:, 0:-3]
    targets = gen_data[:, -3:]

    # split into train, dev, and test sets

    # calculate train index
    train_ind = int(train_perc * features.shape[0])
    dev_ind = int((train_perc + dev_perc) * features.shape[0])

    # do the splitting
    train = (features[0:train_ind, :], targets[0:train_ind, :])

    dev = (features[train_ind:dev_ind, :], targets[train_ind:dev_ind, :])

    test = (features[dev_ind:, :], targets[dev_ind:, :])

    return train, dev, test


# ----------------------------------------------------------------------------------------------------------------------

def flatten_extend(matrix):
    flat_list = []
    for elem in matrix:
        flat_list.extend(elem)
    return flat_list


# ----------------------------------------------------------------------------------------------------------------------

def pick_data_idx(data, idx):
    yaw_rate = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
    uy = float(data['Vehicle_States.lateral_vel_wrt_road'][idx])
    ux = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
    steer = float(data['driver_demands.steering'][idx])

    # new version
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    fr = (frl + frr) / 2
    ffl = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    ffr = float(data['Tire.Ground_Surface_Force_X.R1'][idx])
    ff = (ffl + ffr) / 2
    fx = (ff + fr) / 2

    return yaw_rate, uy, ux, steer, fx


def pick_data_idx_rl(data, idx):
    yaw_rate = float(data['Vehicle_States.yaw_angular_vel_wrt_road'][idx])
    uy = float(data['Vehicle_States.lateral_vel_wrt_road'][idx])
    ux = float(data['Vehicle_States.longitudinal_vel_wrt_road'][idx])
    steer = float(data['driver_demands.steering'][idx])
    """# old version
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    fr = (frl + frr) / 2
    ff = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    fx = fr + ff"""

    # new version
    frl = float(data['Tire.Ground_Surface_Force_X.L2'][idx])
    frr = float(data['Tire.Ground_Surface_Force_X.R2'][idx])
    ffl = float(data['Tire.Ground_Surface_Force_X.L1'][idx])
    ffr = float(data['Tire.Ground_Surface_Force_X.R1'][idx])
    fl = (ffl + frl) / 2
    fr = (ffr + frr) / 2

    return yaw_rate, uy, ux, steer, ffr, ffl, frr, frl


# ----------------------------------------------------------------------------------------------------------------------

def pick_acc_idx(data, idx):
    ax = float(data['Vehicle_States.longitudinal_acc_wrt_road'][idx])
    ay = float(data['Vehicle_States.lateral_acc_wrt_road'][idx])

    return ax, ay


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_car_perf(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_car_perf/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        """fr = (ffr + frr) / 2
        fl = (ffl + frl) / 2"""

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        test_set_rl = np.transpose(np.array([yaw_rate, ux, steer, ffr, ffl, frr, frl, uy]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_' + str(i) + '.csv', index=False, header=False)
        dataframe_rl = pd.DataFrame(test_set_rl)
        dataframe_rl.to_csv(path_to_output_ + 'test_set_' + str(i) + '_rrff.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_road_grip(path_to_data, path_to_output_, number_of_sets):
    mu_set = ['1', '08', '06', '0806', '0804', '06045', '0603']

    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_road_grip/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_mu_' + mu_set[i] + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_static_equilibrium(path_to_output_, seconds_of_test):
    degs = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60]
    test_length = seconds_of_test * 100

    # Creating steering tests
    print('CREATING STEERING EQUILIBRA TEST')
    yaw_rate = np.full(test_length, 0.0)
    uy = np.full(test_length, 0.0)
    ux = np.full(test_length, 0.0)
    steer = np.full(test_length, 0.0)
    fx = np.full(test_length, 100)
    print('CREATING STEP STEERING TEST')
    for deg in degs:
        yaw_rate = np.full(test_length, 0.0)
        uy = np.full(test_length, 0.0)
        ux = np.full(test_length, 0.0)
        steer = np.zeros(test_length)
        steer[10000:] = deg * np.pi / 180
        fx = np.full(test_length, 100)
        # Save test set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # Save test set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_from0_fx100_' + str(deg).replace('-', '') + 'deg.csv', index=False, header=False)
    print('END CREATION STEP STEERING TEST')

def create_test_step_steer(path_to_data, path_to_output_, number_of_tests):
    dir_ = ['step_steering_highspeed/', 'step_steering_lowspeed/']
    filename = 'DemoSportsCar_step.csv'
    degs = ['0', '1', '2', '3', '4', '5', '10', '15', '20', '25', '30', '45', '60', '75', '90', '100']

    # Creating ramp steering tests
    print('CREATING STEP STEERING TEST')
    for el in dir_:
        for i in range(number_of_tests):
            data = pd.read_csv(path_to_data + el + 'test_' + str(i) + '/' + filename, dtype=object)
            data = data.drop(0, axis='rows')  # remove the row containing the measure units
            data.reset_index(drop=True, inplace=True)

            yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)
            uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)
            ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

            # Delta
            steer = data['driver_demands.steering'].to_numpy().astype(float)

            # Fx
            frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
            frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
            fr = (frl + frr) / 2
            ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
            ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
            ff = (ffl + ffr) / 2
            fx = (fr + ff) / 2

            # Save test set
            test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

            # Save test set
            dataframe = pd.DataFrame(test_set)
            if 'highspeed' in el:
                dataframe.to_csv(path_to_output_ + 'test_set_stepsteer_fx100_' + degs[i] + 'deg.csv', index=False,
                             header=False)
            elif 'lowspeed' in el:
                dataframe.to_csv(path_to_output_ + 'test_set_stepsteer_fx25_' + degs[i] + 'deg.csv', index=False,
                                 header=False)
    print('END CREATION STEP STEERING TEST')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_ramp_steer(path_to_data, path_to_output_, number_of_tests):
    dir_ = 'ramp_steering/'
    filename = 'DemoSportsCar_ramp.csv'

    # Creating ramp steering tests
    print('CREATING RAMP STEERING TEST')
    for i in range(number_of_tests):
        data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

        # Save test set
        dataframe = pd.DataFrame(test_set)
        if i < math.floor(number_of_tests / 2):
            dataframe.to_csv(path_to_output_ + 'test_set_rampsteer_fx100_' + str(i).replace('-', '') + '.csv',
                             index=False,
                             header=False)
        else:
            dataframe.to_csv(path_to_output_ + 'test_set_rampsteer_fx25_' + str(i).replace('-', '') + '.csv',
                             index=False,
                             header=False)
    print('END CREATION RAMP STEERING TEST')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_sine_steer(path_to_data, path_to_output_, number_of_tests):
    dir_ = 'sine_steering/'
    filename = 'DemoSportsCar_sin.csv'

    # Creating ramp steering tests
    print('CREATING SINE STEERING TEST')
    for i in range(number_of_tests):
        data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

        # Save test set
        dataframe = pd.DataFrame(test_set)
        if i < math.floor(number_of_tests / 2):
            dataframe.to_csv(path_to_output_ + 'test_set_sinesteer_fx100_' + str(i).replace('-', '') + '.csv',
                             index=False,
                             header=False)
        else:
            dataframe.to_csv(path_to_output_ + 'test_set_sinesteer_fx25_' + str(i).replace('-', '') + '.csv',
                             index=False,
                             header=False)
    print('END CREATION SINE STEERING TEST')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_impulse_steer(path_to_data, path_to_output_, number_of_tests, seconds_of_test):
    dir_ = 'impulse_steering/'
    filename = 'DemoSportsCar_imp.csv'

    test_length = seconds_of_test * 100

    # Creating ramp steering tests
    print('CREATING IMPULSE STEERING TEST')
    for i in range(number_of_tests):
        data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        yaw_rate = np.full(test_length, 0.0)
        uy = np.full(test_length, 0.0)
        ux = np.full(test_length, 29.0)
        steer = np.zeros(test_length)
        imp_steer = data['driver_demands.steering'].to_numpy().astype(float)
        steer[-len(imp_steer):] = imp_steer
        fx = np.full(test_length, 100)

        # Save test set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

        # Save test set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_impulsesteer_fx100_' + str(i).replace('-', '') + '.csv',
                         index=False, header=False)
    print('END CREATION IMPULSE STEERING TEST')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_sweep_steer(path_to_data, path_to_output_, number_of_tests, seconds_of_test):
    dir_ = 'swept_steering/'
    filename = 'DemoSportsCar_swe.csv'

    test_length = seconds_of_test * 100

    # Creating ramp steering tests
    print('CREATING SWEEP STEERING TEST')
    for i in range(number_of_tests):
        data = pd.read_csv(path_to_data + dir_ + 'test_' + str(i) + '/' + filename, dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        yaw_rate = np.full(test_length, 0.0)
        uy = np.full(test_length, 0.0)
        ux = np.full(test_length, 29.0)
        steer = np.zeros(test_length)
        sweep_steer = data['driver_demands.steering'].to_numpy().astype(float)
        steer[-len(sweep_steer):] = sweep_steer
        fx = np.full(test_length, 100)

        # Save test set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

        # Save test set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_sweepsteer_fx100_' + str(i).replace('-', '') + '.csv', index=False,
                         header=False)
    print('END CREATION SWEEP STEERING TEST')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_param_study(path_to_data, path_to_output_):
    directories = ['/grip_1_perf_100/', '/grip_1_perf_75/', '/grip_1_perf_50/',
                   '/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/',
                   '/grip_08_perf_100/', '/grip_08_perf_75/', '/grip_08_perf_50/']

    print('CREATION TEST DATA')
    for i, dir_ in tqdm(enumerate(directories)):
        data = pd.read_csv(path_to_data + 'parameters_study/' + dir_ + 'DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_paramstudy_' + dir_.replace('/', '') + '.csv', index=False,
                         header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_test_set_mass(path_to_data, path_to_output_, number_of_sets):
    print('CREATION TEST DATA')
    for i in tqdm(range(0, number_of_sets)):
        data = pd.read_csv(path_to_data + 'test_mass/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(path_to_output_ + 'test_set_mass_' + str(i) + '.csv', index=False, header=False)

    print('END CREATION TEST DATA')


# ----------------------------------------------------------------------------------------------------------------------

def create_piste_training_complete(data_, output_, number_of_sets):
    print('CREATION DATASET FROM TRAINING DATA')
    for i in tqdm(range(1, number_of_sets + 1)):
        data = pd.read_csv(data_ + '/flat/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Yaw rate
        yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

        # Vy
        uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

        # Vx
        ux = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

        # Delta
        steer = data['driver_demands.steering'].to_numpy().astype(float)

        # Fx
        frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
        frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
        fr = (frl + frr) / 2
        ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
        ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
        ff = (ffl + ffr) / 2
        fx = (fr + ff) / 2

        # Save test set
        # Create test_set
        test_set = np.transpose(np.array([yaw_rate, uy, ux, steer, fx]))
        # print(test_set.shape)

        # Save test_set
        dataframe = pd.DataFrame(test_set)
        dataframe.to_csv(output_ + 'test_set_' + str(i + 1) + '.csv', index=False, header=False)
    print('END CREATION DATASET FROM TRAINING DATA')

# ----------------------------------------------------------------------------------------------------------------------
