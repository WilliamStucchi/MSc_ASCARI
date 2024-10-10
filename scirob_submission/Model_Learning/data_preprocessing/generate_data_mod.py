# Run all required imports
from new_data_generation_functions import *
from parameters.learning_params import *
import numpy as np
import matplotlib.pyplot as plt

# Set the Random Seed for Repeatability
np.random.seed(1)

print('Generating experiment 1 data!')
# Generate Data
data, yaw_rates, steering_angles, long_vels = gen_data_mod(Param, Veh)
yaw_rates = np.array(yaw_rates[:10000])
steering_angles = np.array(steering_angles[:10000])
long_vels = np.array(long_vels[:10000])

print(data.shape)
np.savetxt('../data/new/bicycle_model_for_yawsteergraph.csv', data, delimiter=',')

"""
Ranges for the speeds:
    - range 1: 0 to 40 km/h
    - range 2: 41 to 80 km/h
    - range 3: 81 to 120 km/h
    - range 4: 121 to 160 km/h
    - range 5: 161 to max km/h
"""
mask_range_1 = (long_vels >= 0) & (long_vels <= 40)
mask_range_2 = (long_vels >= 41) & (long_vels <= 80)
mask_range_3 = (long_vels >= 81) & (long_vels <= 120)
mask_range_4 = (long_vels >= 121) & (long_vels <= 160)
mask_range_5 = (long_vels >= 161)
geater_than_zero_mask = (steering_angles >= 0)

yaw_1 = yaw_rates[mask_range_1 & geater_than_zero_mask]
yaw_2 = yaw_rates[mask_range_2 & geater_than_zero_mask]
yaw_3 = yaw_rates[mask_range_3 & geater_than_zero_mask]
yaw_4 = yaw_rates[mask_range_4 & geater_than_zero_mask]
yaw_5 = yaw_rates[mask_range_5 & geater_than_zero_mask]

steering_1 = (steering_angles[mask_range_1 & geater_than_zero_mask]) * 180 / np.pi
steering_2 = (steering_angles[mask_range_2 & geater_than_zero_mask]) * 180 / np.pi
steering_3 = (steering_angles[mask_range_3 & geater_than_zero_mask]) * 180 / np.pi
steering_4 = (steering_angles[mask_range_4 & geater_than_zero_mask]) * 180 / np.pi
steering_5 = (steering_angles[mask_range_5 & geater_than_zero_mask]) * 180 / np.pi

print('PLOTTING')
# Plot gg-plot
plt.figure(figsize=(20, 12))
plt.rc('font', size=15)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=22)  # Titolo degli assi
plt.rc('axes', labelsize=22)  # Etichette degli assi
plt.rc('xtick', labelsize=22)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=22)  # Etichette dei ticks su y
plt.rc('legend', fontsize=17)  # Legenda

# Road grip
plt.scatter(steering_1, yaw_1, alpha=0.9, s=5, label='0 km/h <= Vx <= 40 km/h', color='red')
plt.scatter(steering_2, yaw_2, alpha=0.9, s=5, label='41 km/h <= Vx <= 80 km/h', color='black')
plt.scatter(steering_3, yaw_3, alpha=0.9, s=5, label='81 km/h <= Vx <= 120 km/h', color='green')
plt.scatter(steering_4, yaw_4, alpha=0.9, s=5, label='121 km/h <= Vx <= 160 km/h', color='yellow')
plt.scatter(steering_5, yaw_5, alpha=0.9, s=5, label='161 km/h <= Vx', color='orange')

legend = plt.legend(loc='best')
for lh in legend.legendHandles:
    lh.set_alpha(0.8)  # Increase opacity of legend items
plt.title('Steering-Yaw_rate relation at different velocities', pad=20)
plt.ylabel('Yaw rate [rad/s]', labelpad=20)
plt.xlabel('Steering [°]', labelpad=20)

plt.grid(True)
fig1 = plt.gcf()
plt.show()
fig1.savefig('../../../../test/deltafx_deltasteer/steering_yaw_relation.png', format='png')

