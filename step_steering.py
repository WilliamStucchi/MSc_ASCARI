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

savepath = '../test/test_post_20241020/mu1mu06 che viene male'
speeds = ['highspeed', 'lowspeed']
step_steering = ['0deg', '1deg', '2deg', '3deg', '4deg', '5deg', '10deg', '15deg', '20deg', '25deg', '30deg', '45deg',
                 '60deg', '75deg', '90deg', '100deg']
final_yaw_NN = np.zeros(len(step_steering))
final_yaw_NN_2 = np.zeros(len(step_steering))
final_yaw_bicycle = np.zeros(len(step_steering))
final_yaw_bicycle_vx_comp = np.zeros(len(step_steering))
final_yaw_labels = np.zeros(len(step_steering))
final_vx_NN = np.zeros(len(step_steering))
final_vx_NN_2 = np.zeros(len(step_steering))
final_vx_bicycle = np.zeros(len(step_steering))
final_vx_bicycle_vx_comp = np.zeros(len(step_steering))
final_vx_labels = np.zeros(len(step_steering))
final_vy_NN = np.zeros(len(step_steering))
final_vy_NN_2 = np.zeros(len(step_steering))
final_vy_bicycle = np.zeros(len(step_steering))
final_vy_bicycle_vy_comp = np.zeros(len(step_steering))
final_vy_labels = np.zeros(len(step_steering))
difference = np.zeros(len(step_steering))

for speed in speeds:
    for idx, value in enumerate(step_steering):
        if 'highspeed' in speed:
            testname = 'results_test_stepsteer_fx100_' + value + '.csv'
        elif 'lowspeed' in speed:
            testname = 'results_test_stepsteer_fx25_' + value + '.csv'
        basepath_NN = 'scirob_submission/Model_Learning/results/step_1/callbacks/'
        # path2results_NN = basepath_NN + '2024_05_17/12_29_37/' + testname # modello allenato su mu1, che sbaglia a bassi gradi di sterzo
        # path2results_NN = basepath_NN + '2024_10_08/16_26_37/' + testname
        path2results_NN = basepath_NN + '2024_10_17/16_05_26/' + testname
        # path2results_NN = basepath_NN + '2024_10_07/15_41_02/' + testname
        path2ax = path2results_NN[:path2results_NN.rfind('.')] + '_ax.csv'
        path2ay = path2results_NN[:path2results_NN.rfind('.')] + '_ay.csv'
        name_NN = 'NN: Bike (μ=1) + CRT (μ=1)'

        path2results_NN_2 = basepath_NN + '2024_10_20/16_09_55/' + testname
        path2ax_2 = path2results_NN_2[:path2results_NN_2.rfind('.')] + '_ax.csv'
        path2ay_2 = path2results_NN_2[:path2results_NN_2.rfind('.')] + '_ay.csv'
        name_NN_2 = 'NN: Bike (μ=1, μ=0.6) + CRT (μ=1, μ=0.6)'

        path2results_bicycle = '../test/steering_equilibrium/handling/results/' + testname
        path2results_bicycle_vx_computed = '../test/steering_equilibrium/handling_vx_computed/results/' + testname

        path2labels = 'scirob_submission/Model_Learning/data/handling/test_set_' + testname[testname.rfind('test_') + 5:]

        with open(path2results_NN, 'r') as fh:
            results_NN = np.loadtxt(fh)
        with open(path2ax, 'r') as fh:
            ax_res = np.loadtxt(fh)
        with open(path2ay, 'r') as fh:
            ay_res = np.loadtxt(fh)

        with open(path2results_NN_2, 'r') as fh:
            results_NN_2 = np.loadtxt(fh)
        with open(path2ax_2, 'r') as fh:
            ax_res_2 = np.loadtxt(fh)
        with open(path2ay_2, 'r') as fh:
            ay_res_2 = np.loadtxt(fh)

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

        vx_result_NN_2 = results_NN_2[:, 2][:, np.newaxis]
        vy_result_NN_2 = results_NN_2[:, 1][:, np.newaxis]
        yaw_result_NN_2 = results_NN_2[:, 0][:, np.newaxis]
        steer_quantity_NN_2 = results_NN_2[:, 3][:, np.newaxis]

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
        
        ax_result_NN_2 = ax_res_2[:, np.newaxis]
        ay_result_NN_2 = ay_res_2[:, np.newaxis]

        ax_result_bicycle = results_bicycle[:, 1][:, np.newaxis]
        ay_result_bicycle = results_bicycle[:, 2][:, np.newaxis]
        ax_result_bicycle_vx_comp = results_bicycle_vx[:, 1][:, np.newaxis]
        ay_result_bicycle_vx_comp = results_bicycle_vx[:, 2][:, np.newaxis]

        vx_label = labels[:, 2][:, np.newaxis]
        vy_label = labels[:, 1][:, np.newaxis]
        yaw_label = labels[:, 0][:, np.newaxis]

        # Longitudinal velocity
        plt.figure(figsize=(16, 8))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=20)  # Titolo degli assi
        plt.rc('axes', labelsize=20)  # Etichette degli assi
        plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=12)  # Legenda

        time_values = np.linspace(0, len(vx_result_NN) / 100, len(vx_result_NN))
        plt.plot(time_values, vx_result_NN, label=name_NN, color='r', linewidth=1.5)
        plt.plot(time_values, vx_result_NN_2, label=name_NN_2, color='orange', linewidth=1.5)
        plt.plot(time_values, vx_result_bicycle_vx_comp, label='Bicycle model with Fx as input', color='green', linewidth=1.5)
        plt.plot(time_values, vx_label, label='Ground Truth', color='b', linewidth=1.5)

        # Add labels and title
        plt.ylabel('Long. vel. vx [m/s]')
        plt.xlabel('Time [s]')
        plt.title('Longitudinal Speed')
        plt.legend(loc='best')

        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig(savepath + '/step_steering'
                        '/fx100_' + value + '_vx.png', format='png', dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/step_steering'
                        '/fx25_' + value + '_vx.png', format='png', dpi=300)
        plt.close()

        # Lateral velocity
        plt.figure(figsize=(16, 8))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=20)  # Titolo degli assi
        plt.rc('axes', labelsize=20)  # Etichette degli assi
        plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=12)  # Legenda

        time_values = np.linspace(0, len(vx_result_NN) / 100, len(vx_result_NN))
        plt.plot(time_values, vy_result_NN, label=name_NN, color='r', linewidth=1.5)
        plt.plot(time_values, vy_result_NN_2, label=name_NN_2, color='orange', linewidth=1.5)
        plt.plot(time_values, vy_result_bicycle, label='Bicycle model', color='green', linewidth=1.5)
        plt.plot(time_values, vy_result_bicycle_vx_comp, label='Bicycle model with Fx as input', color='green',
                 linestyle='dashed', linewidth=1.5)
        plt.plot(time_values, vy_label, label='Ground Truth', color='b', linewidth=1.5)

        # Add labels and title
        plt.ylabel('Lat. vel. vy [m/s]')
        plt.xlabel('Time [s]')
        plt.title('Lateral Speed')
        plt.legend(loc='best')

        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig(savepath + '/step_steering'
                        '/fx100_' + value + '_vy.png', format='png', dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/step_steering'
                        '/fx25_' + value + '_vy.png', format='png', dpi=300)
        plt.close()

        # Yaw rate
        plt.figure(figsize=(16, 6))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=20)  # Titolo degli assi
        plt.rc('axes', labelsize=20)  # Etichette degli assi
        plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=12)  # Legenda
        fig, ax1 = plt.subplots(figsize=(18, 10))

        ax1.xaxis.set_major_locator(MultipleLocator(1))
        # Primo asse y per lo yaw rate
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Yaw Rate [rad/s]', color='black')

        time_values = np.linspace(0, len(yaw_result_NN[:2000]) / 100, len(yaw_result_NN[:2000]))
        ax1.plot(time_values, yaw_result_NN[:2000], color='red', label=name_NN, linewidth=2.5)
        ax1.plot(time_values, yaw_result_NN_2[:2000], color='orange', label=name_NN_2, linewidth=2.5)
        ax1.plot(time_values, yaw_result_bicycle[:2000], color='green', label='Bicycle model', linewidth=2.5)
        ax1.plot(time_values, yaw_result_bicycle_vx_comp[:2000], color='green', label='Bicycle model with Fx as input',
                 linestyle='dashed', linewidth=2.5)
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
            plt.savefig(savepath + '/step_steering'
                        '/fx100_' + value + '_yaw_rate.png', format='png', dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/step_steering'
                        '/fx25_' + value + '_yaw_rate.png', format='png',
                        dpi=300)
        plt.close(fig)
        plt.close()
    
        final_yaw_NN[idx] = results_NN[-1, 0]
        final_yaw_NN_2[idx] = results_NN_2[-1, 0]
        final_yaw_bicycle[idx] = results_bicycle[-1, 4]
        final_yaw_bicycle_vx_comp[idx] = results_bicycle_vx[-1, 4]
        final_yaw_labels[idx] = yaw_label[-1, 0]
    
        final_vx_NN[idx] = results_NN[2000, 2]
        final_vx_NN_2[idx] = results_NN_2[2000, 2]
        final_vx_bicycle[idx] = results_bicycle[2000, 5]
        final_vx_bicycle_vx_comp[idx] = results_bicycle_vx[2000, 5]
        final_vx_labels[idx] = vx_label[2000, 0]

        final_vy_NN[idx] = results_NN[2000, 2]
        final_vy_NN_2[idx] = results_NN_2[2000, 2]
        final_vy_bicycle[idx] = results_bicycle[2000, 5]
        final_vy_bicycle_vy_comp[idx] = results_bicycle_vx[2000, 5]
        final_vy_labels[idx] = vy_label[2000, 0]
    
    gradi = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90, 100]
    if 'highspeed' in speed:
        vel = '100 km/h'
    elif 'lowspeed' in speed:
        vel = '25 km/h'
    
    plt.figure(figsize=(25, 10))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=20)  # Titolo degli assi
    plt.rc('axes', labelsize=20)  # Etichette degli assi
    plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=12)  # Legenda
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.025))

    plt.plot(gradi, final_yaw_labels, label='Ground Truth', marker='o', linestyle='None', color='b', markersize=15, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle, label='Bicycle model', marker='*', linestyle='None', color='#3A3042', markersize=15, markeredgewidth=2.0, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle_vx_comp, label='Bicycle model with Fx as input', marker='v', markersize=15, markeredgewidth=2.0, linestyle='None',
             color='green', linewidth=1.5)
    plt.plot(gradi, final_yaw_NN, label=name_NN, marker='+', linestyle='None', color='r', markersize=15, markeredgewidth=2.0, linewidth=1.5)
    plt.plot(gradi, final_yaw_NN_2, label=name_NN_2, marker='x', linestyle='None', color='orange', markersize=15, markeredgewidth=2.0, linewidth=1.5)
    # per far vedere che mu=1 è errato, usa riga sotto
    # plt.plot(gradi, final_yaw_NN, label='NN trained on μ=1', marker='+', linestyle='None', color='r', markersize=15, markeredgewidth=2.0, linewidth=1.5)
    
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Yaw rate [rad/s]')
    plt.xlabel('Steering [°]')
    plt.title('Relation between steering and Yaw Rate at ' + vel)
    plt.legend(loc='best')
    
    # Display the plot
    plt.grid(True)
    if 'highspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/steer_yaw_fx100_relation.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/steer_yaw_fx25_relation.png', format='png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(25, 10))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=20)  # Titolo degli assi
    plt.rc('axes', labelsize=20)  # Etichette degli assi
    plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=15)  # Legenda
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.0025))

    plt.plot(gradi, final_yaw_labels / final_vx_labels, label='Ground Truth', marker='o', linestyle='None', color='b',
             markersize=15, markeredgewidth=2.0, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle / final_vx_bicycle, label='Bicycle model', marker='*', linestyle='None', color='#3A3042',
             markersize=15, markeredgewidth=2.0, linewidth=1.5)
    plt.plot(gradi, final_yaw_bicycle_vx_comp / final_vx_bicycle_vx_comp, label='Bicycle model with Fx as input',
             marker='v', linestyle='None', color='green', markersize=10, markeredgewidth=2.5, linewidth=1.5)
    plt.plot(gradi, final_yaw_NN / final_vx_NN, label=name_NN, marker='+', linestyle='None',
             color='r',
             markersize=15, markeredgewidth=2.0, linewidth=1.5)
    plt.plot(gradi, final_yaw_NN_2 / final_vx_NN_2, label=name_NN_2, marker='x', linestyle='None', color='orange',
             markersize=15, markeredgewidth=2.0, linewidth=1.5)
    # per far vedere che mu=1 è errato, usa riga sotto
    """plt.plot(gradi, final_yaw_NN / final_vx_NN, label='NN trained on μ=1', marker='+', linestyle='None', color='r',
             markersize=15, markeredgewidth=2.0, linewidth=1.5)"""

    plt.xticks(gradi)
    
    # plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Yaw rate [rad/s]')
    plt.xlabel('Steering [°]')
    plt.title('Normalized (by Vx) relation between Steering angle and Yaw Rate at ' + vel)
    plt.legend(loc='best')
    
    # Display the plot
    plt.grid(True)
    if 'highspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/steer_yaw_fx100_norm_relation.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/steer_yaw_fx25_norm_relation.png', format='png', dpi=300)
    plt.close()
    
    # Calcolo delle metriche per ogni modello
    metrics_nn = calculate_metrics(final_yaw_NN, final_yaw_labels)
    metrics_nn_2 = calculate_metrics(final_yaw_NN_2, final_yaw_labels)
    metrics_bicycle = calculate_metrics(final_yaw_bicycle, final_yaw_labels)
    metrics_bicycle_vx_comp = calculate_metrics(final_yaw_bicycle_vx_comp, final_yaw_labels)
    
    metrics_labels = ['RMSE', 'MAE']
    metrics_nn_values = list(metrics_nn.values())
    metrics_nn_values_2 = list(metrics_nn_2.values())
    metrics_bicycle_values = list(metrics_bicycle.values())
    metrics_bicycle_vx_comp_values = list(metrics_bicycle_vx_comp.values())
    
    x = np.arange(len(metrics_labels))  # la posizione delle metriche sull'asse x
    width = 0.125  # larghezza delle barre
    
    # Creazione dell'istogramma
    plt.figure(figsize=(20, 12))
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=20)  # Titolo degli assi
    plt.rc('axes', labelsize=20)  # Etichette degli assi
    plt.rc('xtick', labelsize=20)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=20)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=15)  # Legenda
    plt.bar(x - 3/2*width, metrics_nn_values, width, label=name_NN, color='red')
    plt.bar(x - width / 2, metrics_nn_values_2, width, label=name_NN_2, color='orange')
    plt.bar(x + width / 2, metrics_bicycle_values, width, label='Bicycle model', color='green')
    plt.bar(x + 3/2*width, metrics_bicycle_vx_comp_values, width, label='Bicycle model with Fx as input',
            color='#3A3042')
    # per far vedere che mu=1 è errato, usa riga sotto
    # plt.bar(x - width, metrics_nn_values_2, width, label='NN trained on μ=1', color='green')
    
    # Aggiunta delle etichette
    plt.ylabel('Values')
    plt.title('Comparison of the metrics at ' + vel)
    plt.xticks([0, 1], metrics_labels)
    plt.legend(loc='best')
    plt.grid()
    
    # Mostrare il grafico
    plt.tight_layout()
    if 'highspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/metrics_comparison_fx100.png', format='png', dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/step_steering'
                    '/metrics_comparison_fx25.png', format='png', dpi=300)
