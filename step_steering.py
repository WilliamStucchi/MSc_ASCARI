import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ----------------------------------------------------------------------------------------------------------------------

# Funzione per calcolare tutte le metriche
def calculate_metrics(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae
    }


# ----------------------------------------------------------------------------------------------------------------------

speeds = ['highspeed', 'lowspeed']
step_steering = ['0deg', '1deg', '2deg', '3deg', '4deg', '5deg', '10deg', '15deg', '20deg', '25deg', '30deg', '45deg',
                 '60deg', '75deg', '90deg', '100deg']
final_yaw_NN = np.zeros(len(step_steering))
final_yaw_bicycle = np.zeros(len(step_steering))
final_yaw_bicycle_vx_comp = np.zeros(len(step_steering))
final_yaw_labels = np.zeros(len(step_steering))
final_vx_NN = np.zeros(len(step_steering))
final_vx_bicycle = np.zeros(len(step_steering))
final_vx_bicycle_vx_comp = np.zeros(len(step_steering))
final_vx_labels = np.zeros(len(step_steering))
difference = np.zeros(len(step_steering))

for speed in speeds:
    for idx, value in enumerate(step_steering):
        if 'highspeed' in speed:
            testname = 'results_test_stepsteer_fx100_' + value + '.csv'
        elif 'lowspeed' in speed:
            testname = 'results_test_stepsteer_fx25_' + value + '.csv'
        basepath_NN = 'scirob_submission/Model_Learning/results/step_1/callbacks/'
        #path2results_NN = basepath_NN + '2024_10_08/16_26_37/' + testname
        path2results_NN = basepath_NN + '2024_10_07/15_41_02/' + testname
        path2ax = path2results_NN[:path2results_NN.rfind('.')] + '_ax.csv'
        path2ay = path2results_NN[:path2results_NN.rfind('.')] + '_ay.csv'

        path2results_bicycle = '../test/steering_equilibrium/handling/results/' + testname
        path2results_bicycle_vx_computed = '../test/steering_equilibrium/handling_vx_computed/results/' + testname

        path2labels = 'scirob_submission/Model_Learning/data/handling/test_set_' + testname[testname.rfind('test_') + 5:]

        with open(path2results_NN, 'r') as fh:
            results_NN = np.loadtxt(fh)
        with open(path2ax, 'r') as fh:
            ax_res = np.loadtxt(fh)
        with open(path2ay, 'r') as fh:
            ay_res = np.loadtxt(fh)

        with open(path2results_bicycle, 'r') as fh:
            data = pd.read_csv(path2results_bicycle, dtype=object)

            results_bicycle = np.array(data, dtype=float)
            results_bicycle[np.isnan(results_bicycle)] = 0

        with open(path2results_bicycle_vx_computed, 'r') as fh:
            data = pd.read_csv(path2results_bicycle_vx_computed, dtype=object)

            results_bicycle_vx = np.array(data, dtype=float)
            results_bicycle_vx[np.isnan(results_bicycle_vx)] = 0

        with open(path2labels, 'r') as fh:
            labels = np.loadtxt(fh, delimiter=',')

        vx_result_NN = results_NN[:, 2][:, np.newaxis]
        vy_result_NN = results_NN[:, 1][:, np.newaxis]
        yaw_result_NN = results_NN[:, 0][:, np.newaxis]
        steer_quantity_NN = results_NN[:, 3][:, np.newaxis]

        vx_result_bicycle = results_bicycle[:, 5][:, np.newaxis]
        vy_result_bicycle = results_bicycle[:, 3][:, np.newaxis]
        yaw_result_bicycle = results_bicycle[:, 4][:, np.newaxis]
        steer_quantity = results_bicycle[:, 3][:, np.newaxis]

        vx_result_bicycle_vx_comp = results_bicycle_vx[:, 5][:, np.newaxis]
        vy_result_bicycle_vx_comp = results_bicycle_vx[:, 3][:, np.newaxis]
        yaw_result_bicycle_vx_comp = results_bicycle_vx[:, 4][:, np.newaxis]
        steer_quantity_vx_comp = results_bicycle_vx[:, 3][:, np.newaxis]

        ax_result_NN = ax_res[:, np.newaxis]
        ay_result_NN = ay_res[:, np.newaxis]
        ax_result_bicycle = results_bicycle[:, 1][:, np.newaxis]
        ay_result_bicycle = results_bicycle[:, 2][:, np.newaxis]
        ax_result_bicycle_vx_comp = results_bicycle_vx[:, 1][:, np.newaxis]
        ay_result_bicycle_vx_comp = results_bicycle_vx[:, 2][:, np.newaxis]

        vx_label = labels[:, 2][:, np.newaxis]
        vy_label = labels[:, 1][:, np.newaxis]
        yaw_label = labels[:, 0][:, np.newaxis]

        """# Longitudinal velocity
        plt.figure(figsize=(16, 8))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=25)  # Titolo degli assi
        plt.rc('axes', labelsize=25)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda

        time_values = np.linspace(0, len(vx_result_NN) / 100, len(vx_result_NN))
        plt.plot(time_values, vx_result_NN, label='Neural network', color='r', linewidth=1.5)
        plt.plot(time_values, vx_result_bicycle_vx_comp, label='Bicycle model with Fx as input', color='orange', linewidth=1.5)
        plt.plot(time_values, vx_result_bicycle, label='Ground Truth', color='b', linewidth=1.5)

        # Add labels and title
        plt.ylabel('Long. vel. vx [m/s]')
        plt.xlabel('Time [s]')
        plt.title('Longitudinal Speed')
        plt.legend(loc='best')

        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig('../test/steering_equilibrium/step_steering/fx100_' + value + '_vx.png', format='png', dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig('../test/steering_equilibrium/step_steering/fx25_' + value + '_vx.png', format='png', dpi=300)
        plt.close()

        # Yaw rate
        plt.figure(figsize=(16, 6))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=25)  # Titolo degli assi
        plt.rc('axes', labelsize=25)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda
        fig, ax1 = plt.subplots(figsize=(18, 10))

        ax1.xaxis.set_major_locator(MultipleLocator(1))
        # Primo asse y per lo yaw rate
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Yaw Rate [rad/s]', color='black')

        time_values = np.linspace(0, len(yaw_result_NN[:2000]) / 100, len(yaw_result_NN[:2000]))
        ax1.plot(time_values, yaw_result_NN[:2000], color='red', label='Neural Network', linewidth=2.5)
        ax1.plot(time_values, yaw_result_bicycle[:2000], color='green', label='Bicycle model', linewidth=2.5)
        ax1.plot(time_values, yaw_result_bicycle_vx_comp[:2000], color='orange', label='Bicycle model with Fx as input',
                 linewidth=2.5)
        ax1.plot(time_values, yaw_label[:2000], color='blue', label='Ground Truth', linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor='black')

        # Secondo asse y per lo sterzo
        ax2 = ax1.twinx()  # Condividi lo stesso asse x
        ax2.set_ylabel('Steering Input [rad]', color='purple')
        ax2.plot(time_values, steer_quantity_NN[:2000], color='purple', label='Steering Input', linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor='purple')

        fig.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax1.transAxes)
        # Display the plot
        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig('../test/steering_equilibrium/step_steering/fx100_' + value + '_yaw_rate.png', format='png', dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig('../test/steering_equilibrium/step_steering/fx25_' + value + '_yaw_rate.png', format='png',
                        dpi=300)
        plt.close()"""
    
        final_yaw_NN[idx] = results_NN[2000, 0]
        final_yaw_bicycle[idx] = results_bicycle[2000, 4]
        final_yaw_bicycle_vx_comp[idx] = results_bicycle_vx[2000, 4]
        final_yaw_labels[idx] = yaw_label[2000, 0]
    
        final_vx_NN[idx] = results_NN[2000, 2]
        final_vx_bicycle[idx] = results_bicycle[2000, 5]
        final_vx_bicycle_vx_comp[idx] = results_bicycle_vx[2000, 5]
        final_vx_labels[idx] = vx_label[2000, 0]
        """if (final_yaw_bicycle[idx] + final_yaw_bicycle[idx]) / 2 != 0:
            difference[idx] = 100 * abs(final_yaw_NN[idx] - final_yaw_bicycle[idx]) / ((final_yaw_bicycle[idx] + final_yaw_bicycle[idx]) / 2)
        else:
            difference[idx] = 0.0"""
    
    gradi = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 100]
    if 'highspeed' in speed:
        vel = '100 km/h'
    elif 'lowspeed' in speed:
        vel = '25 km/h'
    
    plt.figure(figsize=(23, 10))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.025))

    plt.plot(gradi, final_yaw_labels, label='Ground Truth', marker='o', linestyle='None', color='b', markersize=15, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle, label='Bicycle model', marker='*', linestyle='None', color='g', markersize=17, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle_vx_comp, label='Bicycle model with Fx as input', marker='+', markersize=17, markeredgewidth=2.5, linestyle='None',
             color='orange', linewidth=1.5)
    plt.plot(gradi, final_yaw_NN, label='NN model', marker='x', linestyle='None', color='r', markersize=17, markeredgewidth=2.5, linewidth=1.5)
    plt.xticks(gradi)
    
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Yaw rate [rad/s]')
    plt.xlabel('Steering [°]')
    plt.title('Relation between steering and Yaw Rate at ' + vel)
    plt.legend(loc='best')
    
    # Display the plot
    plt.grid(True)
    if 'highspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/steer_yaw_fx100_relation.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/steer_yaw_fx25_relation.png', format='png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(23, 10))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.0025))

    plt.plot(gradi, final_yaw_labels / final_vx_labels, label='Ground Truth', marker='o', linestyle='None', color='b',
             markersize=15, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle / final_vx_bicycle, label='Bicycle model', marker='*', linestyle='None', color='g',
             markersize=17, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle_vx_comp / final_vx_bicycle_vx_comp, label='Bicycle model with Fx as input',
             marker='+', linestyle='None', color='orange', markersize=17, markeredgewidth=2.5, linewidth=1.5)
    plt.plot(gradi, final_yaw_NN / final_vx_NN, label='NN model', marker='x', linestyle='None', color='r', markersize=17,
             markeredgewidth=2.5, linewidth=1.5)
    plt.xticks(gradi)
    
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Yaw rate [rad/s]')
    plt.xlabel('Steering [°]')
    plt.title('Relation between steering and Yaw Rate at ' + vel)
    plt.legend(loc='best')
    
    # Display the plot
    plt.grid(True)
    if 'highspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/steer_yaw_fx100_norm_relation.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/steer_yaw_fx25_norm_relation.png', format='png', dpi=300)
    plt.close()
    
    # Calcolo delle metriche per ogni modello
    metrics_nn = calculate_metrics(final_yaw_NN, final_yaw_labels)
    metrics_bicycle = calculate_metrics(final_yaw_bicycle, final_yaw_labels)
    metrics_bicycle_vx_comp = calculate_metrics(final_yaw_bicycle_vx_comp, final_yaw_labels)
    
    metrics_labels = ['RMSE', 'MAE']
    metrics_nn_values = list(metrics_nn.values())
    metrics_bicycle_values = list(metrics_bicycle.values())
    metrics_bicycle_vx_comp_values = list(metrics_bicycle_vx_comp.values())
    
    x = np.arange(len(metrics_labels))  # la posizione delle metriche sull'asse x
    width = 0.25  # larghezza delle barre
    
    # Creazione dell'istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=22)  # Titolo degli assi
    plt.rc('axes', labelsize=22)  # Etichette degli assi
    plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=17)  # Legenda
    rects1 = plt.bar(x - width, metrics_nn_values, width, label='Neural Network', color='red')
    rects2 = plt.bar(x, metrics_bicycle_values, width, label='Bicycle model', color='green')
    rects3 = plt.bar(x + width, metrics_bicycle_vx_comp_values, width, label='Bicycle model with Fx as input', color='orange')
    
    # Aggiunta delle etichette
    plt.ylabel('Values')
    plt.title('Comparison of the metrics at ' + vel)
    plt.xticks([0, 1], metrics_labels)
    plt.legend(loc='best')
    
    # Mostrare il grafico
    plt.tight_layout()
    if 'highspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/metrics_comparison_fx100.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig('../test/steering_equilibrium/step_steering (train bike mu 1)/metrics_comparison_fx25.png', format='png', dpi=300)
    
    """plt.figure(figsize=(16, 8))

    plt.plot(gradi, difference, marker='x', linestyle='None', color='r', linewidth=1.5)
    plt.xticks(gradi)

    plt.xlabel('Steering input [°]')
    plt.ylabel('Difference [%]')
    plt.title('Difference between NN prediction and Bicycle model prediction')

    # Display the plot
    plt.grid(True)
    plt.savefig('../test/steering_equilibrium/handling/step_steering/steer_yaw_difference.png', format='png', dpi=300)
    plt.close()"""
