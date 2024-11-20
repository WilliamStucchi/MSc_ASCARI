# Run all required imports
from new_data_generation_functions import *
# from just_for_plot_data_generation_functions import *
from parameters.learning_params import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Random Seed for Repeatability
np.random.seed(1)

print('Generating experiment 1 data!')
# Generate Data
data, yaw_rates, steering_angles, long_vels = gen_data_mod(Param, Veh)
yaw_rates = np.array(yaw_rates[:20000])
steering_angles = np.array(steering_angles[:20000])  * 180 / np.pi
long_vels = np.array(long_vels[:20000])

print(data.shape)

np.random.shuffle(data)

np.savetxt('../data/new/just_for_plots.csv', data, delimiter=',')
input('wait')
"""
Ranges for the speeds:
    - range 1: 0 to 40 km/h
    - range 2: 41 to 80 km/h
    - range 3: 81 to 120 km/h
    - range 4: 121 to 160 km/h
    - range 5: 161 to max km/h
"""


print('PLOTTING')
save_format = 'eps'
title = 'old_mu1'
# --------------------------------------------------------------------
# Complete plot
# --------------------------------------------------------------------
cmap = sns.color_palette("magma", as_cmap=True)

# Plot gg-plot
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=25)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=30)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda

points = plt.scatter(steering_angles, yaw_rates, c=long_vels, s=20, cmap=cmap)
cbar = plt.colorbar(points)
cbar.set_label('Long. Vel. [m/s]', rotation=90, labelpad=30)

plt.title('(Steering angle, Yaw_rate) relation at different velocities', pad=20)
plt.ylabel('Yaw rate [rad/s]', labelpad=20)
plt.xlabel('Steering [deg]', labelpad=20)

plt.grid(True)
plt.tight_layout()
fig1 = plt.gcf()
plt.show()
fig1.savefig('../../../../test/deltafx_deltasteer/steer_yaw_rel/steering_yaw_relation_total_vincolifisici_' + title + '.' + save_format, format=save_format)


# --------------------------------------------------------------------
# Positive steering angles
# --------------------------------------------------------------------
mask_positive_steering = (steering_angles >= 0)

# Plot gg-plot
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=25)  # Titolo degli assi
plt.rc('axes', labelsize=30)  # Etichette degli assi
plt.rc('xtick', labelsize=30)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda

points = plt.scatter(steering_angles[mask_positive_steering], yaw_rates[mask_positive_steering],
                     c=long_vels[mask_positive_steering], s=20, cmap=cmap)
cbar = plt.colorbar(points)
cbar.set_label('Long. Vel. [m/s]', rotation=90, labelpad=30)

plt.title('(Steering angle, Yaw_rate) relation at different velocities', pad=20)
plt.ylabel('Yaw rate [rad/s]', labelpad=20)
plt.xlabel('Steering [deg]', labelpad=20)

plt.grid(True)
plt.tight_layout()
fig1 = plt.gcf()
plt.show()
fig1.savefig('../../../../test/deltafx_deltasteer/steer_yaw_rel/steering_yaw_relation_positive_vincolifisici_' + title + '.' + save_format, format=save_format)


# --------------------------------------------------------------------
# Positive steering angles and yaw rate
# --------------------------------------------------------------------

mask_small_yaw_rate = (yaw_rates <= 1) & (yaw_rates >= 0)

# Plot gg-plot
plt.figure(figsize=(15, 10))
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=25)  # Titolo degli assi
plt.rc('axes', labelsize=25)  # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=20)  # Legenda

points = plt.scatter(steering_angles[mask_positive_steering & mask_small_yaw_rate],
                     yaw_rates[mask_positive_steering & mask_small_yaw_rate],
                     c=long_vels[mask_positive_steering & mask_small_yaw_rate], s=25, cmap=cmap)
cbar = plt.colorbar(points)
cbar.set_label('Long. Vel. [m/s]', rotation=90, labelpad=30)


plt.title('(Steering angle, Yaw_rate) relation at different velocities', pad=20)
plt.ylabel('Yaw rate [rad/s]', labelpad=20)
plt.xlabel('Steering [deg]', labelpad=20)

plt.grid(True)
plt.tight_layout()
fig1 = plt.gcf()
plt.show()
fig1.savefig('../../../../test/deltafx_deltasteer/steer_yaw_rel/steering_yaw_relation_smaller_vincolifisici_' + title + '.' + save_format, format=save_format)

