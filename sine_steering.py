import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import iqr


# ----------------------------------------------------------------------------------------------------------------------

# Funzione per calcolare tutte le metriche
def calculate_metrics(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'MSE': mse,
        'MAE': mae
    }


# ----------------------------------------------------------------------------------------------------------------------

color_dict = {'Red_pastel': '#960018',
              'Blue_pastel': '#2A52BE',
              'Green_pastel': '#3CB371',
              'Saffron_pastel': '#FF9933',
              'Purple_pastel': '#9F00FF'}


save_format = 'png'
savepath = '../test/thesis_plots/confronto_NN06_bike/' + save_format + '/'

speeds = ['highspeed', 'lowspeed']
sine_steering = [['0', '1', '2', '3', '4', '5', '6', '7'],
                 ['8', '9', '10', '11', '12', '13', '14', '15']]
time_shown = [700, 400, 300, 200, 600, 400, 300, 250]

yaws_NN = np.empty((len(sine_steering[0]),), dtype=object)
yaws_NN_2 = np.empty((len(sine_steering[0]),), dtype=object)
yaws_NN_3 = np.empty((len(sine_steering[0]),), dtype=object)
yaws_bicycle = np.empty((len(sine_steering[0]),), dtype=object)
yaws_bicycle_vx_comp = np.empty((len(sine_steering[0]),), dtype=object)
yaws_labels = np.empty((len(sine_steering[0]),), dtype=object)

for u, speed in enumerate(speeds):
    for idx, value in enumerate(sine_steering[u]):
        if 'highspeed' in speed:
            testname = 'results_test_sinesteer_fx100_' + value + '.csv'
        elif 'lowspeed' in speed:
            testname = 'results_test_sinesteer_fx25_' + value + '.csv'

        basepath_NN = 'scirob_submission/Model_Learning/results/step_1/callbacks/'
        # path2results_NN = basepath_NN + '2024_10_17/10_56_37/' + testname
        # path2results_NN = basepath_NN + '2024_10_16/19_19_49/' + testname
        # path2results_NN = basepath_NN + '2024_10_08/16_26_37/' + testname
        # path2results_NN = basepath_NN + '2024_10_07/15_41_02/' + testname
        path2results_NN = basepath_NN + '2024_10_20/16_09_55/' + testname
        # path2results_NN = basepath_NN + '2024_10_07/15_41_02/' + testname
        path2ax = path2results_NN[:path2results_NN.rfind('.')] + '_ax.csv'
        path2ay = path2results_NN[:path2results_NN.rfind('.')] + '_ay.csv'
        name_NN = 'NN: Bike (μ=1, μ=0.6) + CRT (μ=1, μ=0.6)'

        path2results_NN_2 = basepath_NN + '2024_10_17/10_56_37/' + testname
        path2ax_2 = path2results_NN_2[:path2results_NN_2.rfind('.')] + '_ax.csv'
        path2ay_2 = path2results_NN_2[:path2results_NN_2.rfind('.')] + '_ay.csv'
        name_NN_2 = 'NN: Bike (μ=1, μ=0.6) + CRT (μ=1)'

        path2results_NN_3 = basepath_NN + '2024_10_20/16_09_55/' + testname
        path2ax_3 = path2results_NN_3[:path2results_NN_2.rfind('.')] + '_ax.csv'
        path2ay_3 = path2results_NN_3[:path2results_NN_2.rfind('.')] + '_ay.csv'
        name_NN_3 = 'NN: Bike (μ=1, μ=0.6) + CRT (μ=1, μ=0.6)'

        path2results_bicycle = '../matlab/handling/' + testname
        name_bike = 'Bicycle model'
        path2results_bicycle_vx_computed = '../matlab/handling_vx_computed/' + testname
        name_bike_vx_comp = 'Bicycle model with FX as input'

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

        with open(path2results_NN_3, 'r') as fh:
            results_NN_3 = np.loadtxt(fh)
        with open(path2ax_3, 'r') as fh:
            ax_res_3 = np.loadtxt(fh)
        with open(path2ay_3, 'r') as fh:
            ay_res_3 = np.loadtxt(fh)

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

        vx_result_NN_3 = results_NN_3[:, 2][:, np.newaxis]
        vy_result_NN_3 = results_NN_3[:, 1][:, np.newaxis]
        yaw_result_NN_3 = results_NN_3[:, 0][:, np.newaxis]
        steer_quantity_NN_3 = results_NN_3[:, 3][:, np.newaxis]

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

        ax_result_NN_3 = ax_res_3[:, np.newaxis]
        ay_result_NN_3 = ay_res_3[:, np.newaxis]

        ax_result_bicycle = results_bicycle[:, 1][:, np.newaxis]
        ay_result_bicycle = results_bicycle[:, 2][:, np.newaxis]
        ax_result_bicycle_vx_comp = results_bicycle_vx[:, 1][:, np.newaxis]
        ay_result_bicycle_vx_comp = results_bicycle_vx[:, 2][:, np.newaxis]

        vx_label = labels[:, 2][:, np.newaxis]
        vy_label = labels[:, 1][:, np.newaxis]
        yaw_label = labels[:, 0][:, np.newaxis]

        # Longitudinal velocity
        plt.figure(figsize=(15, 9))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=30)  # Titolo degli assi
        plt.rc('axes', labelsize=30)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda

        time_values = np.linspace(0, len(vx_result_NN) / 100, len(vx_result_NN))
        plt.plot(time_values, vx_result_NN, label=name_NN, color=color_dict['Red_pastel'], linewidth=2.5)
        """plt.plot(time_values, vx_result_NN_2, label=name_NN_2, color=color_dict['Saffron_pastel'], linewidth=2.5)
        plt.plot(time_values, vx_result_NN_3, label=name_NN_3, color=color_dict['Green_pastel'], linewidth=2.5)"""
        plt.plot(time_values, vx_result_bicycle_vx_comp, label=name_bike_vx_comp, color=color_dict['Green_pastel'], linestyle='dashed', linewidth=2.5)
        plt.plot(time_values, vx_label, label='Ground Truth', color=color_dict['Blue_pastel'], linestyle='dashed', linewidth=2.5)

        # Add labels and title
        plt.ylabel('Long. vel. vx [m/s]')
        plt.xlabel('Time [s]')
        plt.title('Longitudinal Speed')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                        '/fx100_' + value + '_vx.' + save_format, format=save_format, dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                        '/fx25_' + value + '_vx.' + save_format, format=save_format, dpi=300)
        plt.close()

        # Lateral velocity
        plt.figure(figsize=(15, 9))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=30)  # Titolo degli assi
        plt.rc('axes', labelsize=30)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda

        time_values = np.linspace(0, len(vy_result_NN[75:time_shown[idx]]) / 100, len(vy_result_NN[75:time_shown[idx]]))

        plt.plot(time_values, vy_result_NN[75:time_shown[idx]], label=name_NN, color=color_dict['Red_pastel'], linewidth=2.5)
        """plt.plot(time_values, vy_result_NN_2[75:time_shown[idx]], label=name_NN_2, color=color_dict['Saffron_pastel'], linewidth=2.5)
        plt.plot(time_values, vy_result_NN_3[75:time_shown[idx]], label=name_NN_3, color=color_dict['Green_pastel'], linewidth=2.5)"""
        plt.plot(time_values, vy_result_bicycle[75:time_shown[idx]], label=name_bike, color=color_dict['Saffron_pastel'],
                 linewidth=2.5)
        plt.plot(time_values, vy_result_bicycle_vx_comp[75:time_shown[idx]], label=name_bike_vx_comp, color=color_dict['Green_pastel'],
                 linestyle='dashed', linewidth=2.5)
        plt.plot(time_values, vy_label[75:time_shown[idx]], label='Ground Truth', color=color_dict['Blue_pastel'],
                 linestyle='dashed', linewidth=1.5)

        # Add labels and title
        plt.ylabel('Lat. vel. vy [m/s]')
        plt.xlabel('Time [s]')
        plt.title('Lateral Speed')
        plt.legend(loc='best')
        plt.tight_layout()

        plt.grid(True)
        if 'highspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                        '/fx100_' + value + '_vy.' + save_format, format=save_format, dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                        '/fx25_' + value + '_vy.' + save_format, format=save_format, dpi=300)
        plt.close()

        # Yaw rate
        plt.figure(figsize=(15, 9))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=30)  # Titolo degli assi
        plt.rc('axes', labelsize=30)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda

        fig, ax1 = plt.subplots(figsize=(18, 10))
        ax1.xaxis.set_major_locator(MultipleLocator(1))

        # Primo asse y per lo yaw rate
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Yaw Rate [rad/s]', color='black')

        time_values = np.linspace(0, len(yaw_result_NN[75:time_shown[idx]]) / 100, len(yaw_result_NN[75:time_shown[idx]]))

        ax1.plot(time_values, yaw_result_NN[75:time_shown[idx]], color=color_dict['Red_pastel'], label=name_NN, linewidth=2.5)
        """ax1.plot(time_values, yaw_result_NN_2[75:time_shown[idx]], color=color_dict['Saffron_pastel'], label=name_NN_2, linewidth=2.5)
        ax1.plot(time_values, yaw_result_NN_3[75:time_shown[idx]], color=color_dict['Green_pastel'], label=name_NN_3, linewidth=2.5)"""
        ax1.plot(time_values, yaw_result_bicycle[75:time_shown[idx]], color=color_dict['Saffron_pastel'], label=name_bike, linewidth=2.5)
        ax1.plot(time_values, yaw_result_bicycle_vx_comp[75:time_shown[idx]], color=color_dict['Green_pastel'], label=name_bike_vx_comp,
                 linestyle='dashed', linewidth=2.5)
        ax1.plot(time_values, yaw_label[75:time_shown[idx]], color=color_dict['Blue_pastel'], label='Ground Truth', linestyle='dashed', linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor='black')

        # Secondo asse y per lo sterzo
        ax2 = ax1.twinx()  # Condividi lo stesso asse x
        ax2.set_ylabel('Steering Input [rad]', color=color_dict['Purple_pastel'])
        ax2.plot(time_values, steer_quantity_NN[75:time_shown[idx]], color=color_dict['Purple_pastel'], label='Steering Input', linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color_dict['Purple_pastel'])

        fig.legend(loc="lower left", bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes)
        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        if 'highspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                                   '/fx100_' + value + '_yaw_rate.' + save_format, format=save_format, dpi=300)
        elif 'lowspeed' in speed:
            plt.savefig(savepath + '/sine_steering'
                                   '/fx25_' + value + '_yaw_rate.' + save_format, format=save_format, dpi=300)
        plt.close()

        yaws_NN[idx] = results_NN[100:, 0]
        yaws_NN_2[idx] = results_NN_2[100:, 0]
        yaws_NN_3[idx] = results_NN_3[100:, 0]
        yaws_bicycle[idx] = results_bicycle[100:, 4]
        yaws_bicycle_vx_comp[idx] = results_bicycle_vx[100:, 4]
        yaws_labels[idx] = labels[100:, 0]

    rmse_values_NN = []
    mae_values_NN = []
    rmse_values_NN_2 = []
    mae_values_NN_2 = []
    rmse_values_NN_3 = []
    mae_values_NN_3 = []
    rmse_values_bike = []
    mae_values_bike = []
    rmse_values_bike_vx_comp = []
    mae_values_bike_vx_comp = []

    for i in range(len(yaws_labels)):
        rmse_values_NN.append(np.sqrt(mean_squared_error(yaws_labels[i], yaws_NN[i])))
        mae_values_NN.append(mean_absolute_error(yaws_labels[i], yaws_NN[i]))

        rmse_values_NN_2.append(np.sqrt(mean_squared_error(yaws_labels[i], yaws_NN_2[i])))
        mae_values_NN_2.append(mean_absolute_error(yaws_labels[i], yaws_NN_2[i]))

        rmse_values_NN_3.append(np.sqrt(mean_squared_error(yaws_labels[i], yaws_NN_3[i])))
        mae_values_NN_3.append(mean_absolute_error(yaws_labels[i], yaws_NN_3[i]))

        rmse_values_bike.append(np.sqrt(mean_squared_error(yaws_labels[i], yaws_bicycle[i])))
        mae_values_bike.append(mean_absolute_error(yaws_labels[i], yaws_bicycle[i]))

        rmse_values_bike_vx_comp.append(np.sqrt(mean_squared_error(yaws_labels[i], yaws_bicycle_vx_comp[i])))
        mae_values_bike_vx_comp.append(mean_absolute_error(yaws_labels[i], yaws_bicycle_vx_comp[i]))

    # NN 1
    median_rmse_NN = np.median(rmse_values_NN)
    iqr_rmse_NN = iqr(rmse_values_NN)
    median_mae_NN = np.median(mae_values_NN)
    iqr_mae_NN = iqr(mae_values_NN)

    # NN 2
    median_rmse_NN_2 = np.median(rmse_values_NN_2)
    iqr_rmse_NN_2 = iqr(rmse_values_NN_2)
    median_mae_NN_2 = np.median(mae_values_NN_2)
    iqr_mae_NN_2 = iqr(mae_values_NN_2)

    # NN 2
    median_rmse_NN_3 = np.median(rmse_values_NN_3)
    iqr_rmse_NN_3 = iqr(rmse_values_NN_3)
    median_mae_NN_3 = np.median(mae_values_NN_3)
    iqr_mae_NN_3 = iqr(mae_values_NN_3)
    
    # Bike
    median_rmse_bike = np.median(rmse_values_bike)
    iqr_rmse_bike = iqr(rmse_values_bike)
    median_mae_bike = np.median(mae_values_bike)
    iqr_mae_bike = iqr(mae_values_bike)

    # bike_vx_comp
    median_rmse_bike_vx_comp = np.median(rmse_values_bike_vx_comp)
    iqr_rmse_bike_vx_comp = iqr(rmse_values_bike_vx_comp)
    median_mae_bike_vx_comp = np.median(mae_values_bike_vx_comp)
    iqr_mae_bike_vx_comp = iqr(mae_values_bike_vx_comp)
    

    metrics_nn_values = list({'Median RMSE': median_rmse_NN, 'Median MAE': median_mae_NN}.values())
    metrics_nn_values_2 = list({'Median RMSE': median_rmse_NN_2, 'Median MAE': median_mae_NN_2}.values())
    metrics_nn_values_3 = list({'Median RMSE': median_rmse_NN_3, 'Median MAE': median_mae_NN_3}.values())
    metrics_bike_values = list({'Median RMSE': median_rmse_bike, 'Median MAE': median_mae_bike}.values())
    metrics_bike_vx_comp_values = list({'Median RMSE': median_rmse_bike_vx_comp, 'Median MAE': median_mae_bike_vx_comp}.values())

    metrics_labels = ['Median RMSE', 'Median MAE']

    x = np.arange(len(metrics_labels))  # la posizione delle metriche sull'asse x
    width = 0.125  # larghezza delle barre

    # Creazione dell'istogramma
    plt.figure(figsize=(15, 10))
    plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=30)  # Titolo degli assi
    plt.rc('axes', labelsize=30)  # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda
    plt.bar(x - width, metrics_nn_values, width, label=name_NN, color=color_dict['Red_pastel'])
    """plt.bar(x, metrics_nn_values_2, width, label=name_NN_2, color=color_dict['Saffron_pastel'])
    plt.bar(x + width, metrics_nn_values_3, width, label=name_NN_3, color=color_dict['Green_pastel'])"""
    plt.bar(x, metrics_bike_values, width, label=name_bike, color=color_dict['Saffron_pastel'])
    plt.bar(x + width, metrics_bike_vx_comp_values, width, label=name_bike_vx_comp, color=color_dict['Green_pastel'])

    # Aggiunta delle etichette
    plt.ylabel('Values')
    plt.title('Comparison of the metrics for the yaw rate')
    plt.xticks([0, 1], metrics_labels)
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    # Mostrare il grafico
    if 'highspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_median_fx100.' + save_format, format=save_format, dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_median_fx25.' + save_format, format=save_format, dpi=300)
    plt.close()

    # Data: RMSE and MAE values for each model
    """rmse_data = [rmse_values_NN, rmse_values_NN_2, rmse_values_NN_3]
    mae_data = [mae_values_NN, mae_values_NN_2, mae_values_NN_3]"""

    rmse_data = [rmse_values_NN, rmse_values_bike, rmse_values_bike_vx_comp]
    mae_data = [mae_values_NN, mae_values_bike, mae_values_bike_vx_comp]

    # Group data by metric
    data = [rmse_data, mae_data]

    # Positioning for each model within RMSE and MAE
    positions = [[1, 2, 3], [5, 6, 7]]  # Positions for RMSE and MAE for each model

    # Colors for each model
    colors = [color_dict['Red_pastel'], color_dict['Saffron_pastel'], color_dict['Green_pastel']]

    plt.figure(figsize=(15, 10))
    plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=30)  # Titolo degli assi
    plt.rc('axes', labelsize=30)  # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda
    # Create the box plots
    for i, metric_data in enumerate(data):
        for j, model_data in enumerate(metric_data):
            plt.boxplot(model_data, positions=[positions[i][j]], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=colors[j]), medianprops=dict(color="black"), showfliers=False)

    # Set the x-axis labels and legend
    plt.xticks([2, 6], ['RMSE', 'MAE'])
    plt.xlabel("Metrics")
    plt.title("RMSE and MAE Distribution for Different Models")

    # Add a legend for the models
    # Create custom legend handles
    legend_handles = [
        mpatches.Patch(color=colors[0], label=name_NN),
        mpatches.Patch(color=colors[1], label=name_bike),
        mpatches.Patch(color=colors[2], label=name_bike_vx_comp)
    ]
    plt.legend(handles=legend_handles, loc="best")
    # plt.show()
    plt.tight_layout()
    # Mostrare il grafico
    if 'highspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_whiskers_fx100.' + save_format, format=save_format, dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_whiskers_fx25.' + save_format, format=save_format, dpi=300)
    plt.close()

    """# Calcolo delle metriche per ogni modello
    metrics_nn_individual = []
    metrics_nn_individual_2 = []
    metrics_nn_individual_3 = []
    metrics_bicycle_individual = []
    metrics_bicycle_vx_comp_individual = []

    for i in range(len(sine_steering)):
        metrics_nn_individual.append(calculate_metrics(yaws_NN[i], yaws_labels[i]))
        metrics_nn_individual_2.append(calculate_metrics(yaws_NN_2[i], yaws_labels[i]))
        metrics_nn_individual_3.append(calculate_metrics(yaws_NN_3[i], yaws_labels[i]))
        metrics_bicycle_individual.append(calculate_metrics(yaws_bicycle[i], yaws_labels[i]))
        metrics_bicycle_vx_comp_individual.append(calculate_metrics(yaws_bicycle_vx_comp[i], yaws_labels[i]))

    mse_values_NN = [item['MSE'] for item in metrics_nn_individual]
    mse_values_NN_2 = [item['MSE'] for item in metrics_nn_individual_2]
    mse_values_NN_3 = [item['MSE'] for item in metrics_nn_individual_3]
    mse_values_bicycle = [item['MSE'] for item in metrics_bicycle_individual]
    mse_values_bicycle_vx_comp = [item['MSE'] for item in metrics_bicycle_vx_comp_individual]

    mae_values_NN = [item['MAE'] for item in metrics_nn_individual]
    mae_values_NN_2 = [item['MAE'] for item in metrics_nn_individual_2]
    mae_values_NN_3 = [item['MAE'] for item in metrics_nn_individual_3]
    mae_values_bicycle = [item['MAE'] for item in metrics_bicycle_individual]
    mae_values_bicycle_vx_comp = [item['MAE'] for item in metrics_bicycle_vx_comp_individual]

    mean_mse_NN = np.mean(mse_values_NN)
    mean_mse_NN_2 = np.mean(mse_values_NN_2)
    mean_mse_NN_3 = np.mean(mse_values_NN_3)
    mean_mse_bicycle = np.mean(mse_values_bicycle)
    mean_mse_bicycle_vx_comp = np.mean(mse_values_bicycle_vx_comp)

    rmse_NN = np.sqrt(mean_mse_NN)
    rmse_NN_2 = np.sqrt(mean_mse_NN_2)
    rmse_NN_3 = np.sqrt(mean_mse_NN_3)
    rmse_bicycle = np.sqrt(mean_mse_bicycle)
    rmse_bicycle_vx_comp = np.sqrt(mean_mse_bicycle_vx_comp)

    mean_mae_NN = np.mean(mae_values_NN)
    mean_mae_NN_2 = np.mean(mae_values_NN_2)
    mean_mae_NN_3 = np.mean(mae_values_NN_3)
    mean_mae_bicycle = np.mean(mae_values_bicycle)
    mean_mae_bicycle_vx_comp = np.mean(mae_values_bicycle_vx_comp)

    metrics_nn_values = list({'RMSE': rmse_NN, 'MAE': mean_mae_NN}.values())
    metrics_nn_values_2 = list({'RMSE': rmse_NN_2, 'MAE': mean_mae_NN_2}.values())
    metrics_nn_values_3 = list({'RMSE': rmse_NN_3, 'MAE': mean_mae_NN_3}.values())
    metrics_bicycle_values = list({'RMSE': rmse_bicycle, 'MAE': mean_mae_bicycle}.values())
    metrics_bicycle_vx_comp_values = list({'RMSE': rmse_bicycle_vx_comp, 'MAE': mean_mae_bicycle_vx_comp}.values())

    metrics_labels = ['RMSE', 'MAE']

    x = np.arange(len(metrics_labels))  # la posizione delle metriche sull'asse x
    width = 0.125  # larghezza delle barre

    # Creazione dell'istogramma
    plt.figure(figsize=(15, 10))
    plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=30)  # Titolo degli assi
    plt.rc('axes', labelsize=30)  # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda
    plt.bar(x - width, metrics_nn_values, width, label=name_NN, color=color_dict['Red_pastel'])
    plt.bar(x, metrics_nn_values_2, width, label=name_NN_2, color=color_dict['Saffron_pastel'])
    plt.bar(x + width, metrics_nn_values_3, width, label=name_NN_3, color=color_dict['Green_pastel'])"""
    """plt.bar(x, metrics_bicycle_values, width, label='Bicycle model', color='green')
    plt.bar(x + width, metrics_bicycle_vx_comp_values, width, label='Bicycle model with Fx as input',
                     color='#3A3042')"""

    """# Aggiunta delle etichette
    plt.ylabel('Values')
    plt.title('Comparison of the metrics for the yaw rate')
    plt.xticks([0, 1], metrics_labels)
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()

    # Mostrare il grafico
    plt.tight_layout()
    if 'highspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_fx100.' + save_format, format=save_format, dpi=300)
    elif 'lowspeed' in speed:
        plt.savefig(savepath + '/sine_steering'
                               '/metrics_comparison_fx25.' + save_format, format=save_format, dpi=300)"""




