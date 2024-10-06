import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

step_steering = ['0deg', '1deg', '2deg', '3deg', '4deg', '5deg', '10deg', '15deg', '20deg', '25deg', '30deg', '45deg', '60deg']
final_yaw_NN = np.zeros(len(step_steering))
final_yaw_bicycle = np.zeros(len(step_steering))
difference = np.zeros(len(step_steering))

ramp_steering = ['0', '1', '2', '3']
impulse_steering = ['0', '1', '2', '3', '4', '5']
sine_steering = ['0', '1']
sweep_steering = ['0']


for idx, value in enumerate(sine_steering):
    testname = 'results_test_sinesteer_fx100_' + value + '.csv'
    basepath_NN = 'scirob_submission/Model_Learning/results/step_1/callbacks/'
    path2results_NN = basepath_NN + '2024_10_02/16_30_34/' + testname
    path2ax = path2results_NN[:path2results_NN.rfind('.')] + '_ax.csv'
    path2ay = path2results_NN[:path2results_NN.rfind('.')] + '_ay.csv'

    path2results_bicycle = '../test/steering_equilibrium/bicycle_2/' + testname

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

    vx_result_NN = results_NN[:, 2][:, np.newaxis]
    vy_result_NN = results_NN[:, 1][:, np.newaxis]
    yaw_result_NN = results_NN[:, 0][:, np.newaxis]
    steer_quantity_NN = results_NN[:, 3][:, np.newaxis]

    vx_result_bicycle = results_bicycle[:, 5][:, np.newaxis]
    vy_result_bicycle = results_bicycle[:, 3][:, np.newaxis]
    yaw_result_bicycle = results_bicycle[:, 4][:, np.newaxis]
    steer_quantity = results_bicycle[:, 3][:, np.newaxis]

    ax_result_NN = ax_res[:, np.newaxis]
    ay_result_NN = ay_res[:, np.newaxis]
    ax_result_bicycle = results_bicycle[:, 1][:, np.newaxis]
    ay_result_bicycle = results_bicycle[:, 2][:, np.newaxis]

    # Longitudinal velocity
    plt.figure(figsize=(16, 8))
    # Aumentare il font size per tutto il grafico
    plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
    plt.rc('axes', titlesize=25)  # Titolo degli assi
    plt.rc('axes', labelsize=25)  # Etichette degli assi
    plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
    plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
    plt.rc('legend', fontsize=20)  # Legenda

    plt.plot(vx_result_NN, label='Neural network', color='r', linewidth=1.5)
    plt.plot(vx_result_bicycle, label='Bicycle model', color='b', linewidth=1.6)

    # Add labels and title
    plt.ylabel('Long. vel. vx [m/s]')
    plt.xlabel('Time [steps]')
    plt.title('Longitudinal Speed')
    plt.legend(loc='best')

    plt.grid(True)
    plt.savefig('../test/steering_equilibrium/sine_steering/fx100_' + value + '_vx.png', format='png', dpi=300)
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
    fig, ax1 = plt.subplots(figsize=(25, 15))

    """plt.plot(yaw_result_NN, label='Neural network', color='r', linewidth=0.75)
    plt.plot(yaw_result_bicycle, label='Bicycle model', color='b', linewidth=0.750)"""

    ax1.xaxis.set_major_locator(MultipleLocator(0.1))
    # Primo asse y per lo yaw rate
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Yaw Rate [rad/s]', color='black')

    time_values = np.linspace(0, len(yaw_result_NN[10050:10500]) / 100, len( yaw_result_NN[10050:10500]))
    ax1.plot(time_values, yaw_result_NN[10050:10500], color='red', label='Yaw rate Neural Network', linewidth=1.5)
    ax1.plot(time_values, yaw_result_bicycle[10050:10500], color='blue', label='Yaw rate Bicycle model', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=90)

    """# Secondo asse y per lo sterzo
    ax2 = ax1.twinx()  # Condividi lo stesso asse x
    ax2.set_ylabel('Steering Input [rad]', color='green')
    ax2.plot(time_values, steer_quantity_NN[10000:], color='green', label='Steering Input', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor='green')"""

    """# Add labels and title
    plt.ylabel('Yaw rate [rad/s]')
    plt.xlabel('Time [steps]')
    plt.title('Yaw rate')
    plt.legend(loc='best')"""
    fig.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax1.transAxes)
    # Display the plot
    plt.grid(True)
    plt.savefig('../test/steering_equilibrium/sine_steering/fx100_' + value + '_yaw_rate.png', format='png', dpi=300)
    plt.close()

    final_yaw_NN[idx] = results_NN[-1, 0]
    final_yaw_bicycle[idx] = results_bicycle[-1, 4]
    if (final_yaw_bicycle[idx] + final_yaw_bicycle[idx]) / 2 != 0:
        difference[idx] = 100 * abs(final_yaw_NN[idx] - final_yaw_bicycle[idx]) / ((final_yaw_bicycle[idx] + final_yaw_bicycle[idx]) / 2)
    else:
        difference[idx] = 0.0

"""gradi = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60]

plt.figure(figsize=(16, 8))
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_locator(MultipleLocator(1))

plt.plot(gradi, final_yaw_NN, label='NN model', marker='x', linestyle='None', color='r', linewidth=1.5)
plt.plot(gradi, final_yaw_bicycle, label='Bicycle model', marker='o', linestyle='None', color='b', linewidth=1.5)
plt.xticks(gradi)

plt.ylabel('Yaw rate [rad/s]')
plt.xlabel('Steering [°]')
plt.title('Relation between steering and Yaw Rate')
plt.legend(loc='best')

# Display the plot
plt.grid(True)
plt.savefig('../test/steering_equilibrium/step_steering/steer_yaw_relation.png', format='png', dpi=300)
plt.close()

plt.figure(figsize=(16, 8))

plt.plot(gradi, difference, marker='x', linestyle='None', color='r', linewidth=1.5)
plt.xticks(gradi)

plt.xlabel('Steering input [°]')
plt.ylabel('Difference between NN prediction and Bicycle model prediction [%]')
plt.title('Relation between steering and Yaw Rate')

# Display the plot
plt.grid(True)
plt.savefig('../test/steering_equilibrium/step_steering/steer_yaw_difference.png', format='png', dpi=300)
plt.close()"""



