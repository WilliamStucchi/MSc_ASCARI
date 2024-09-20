import numpy as np
import matplotlib.pyplot as plt

basepath = 'scirob_submission/Model_Learning/results/step_1/callbacks/'
path2results = basepath + '2024_09_17/17_41_36/results_test_steereq_fx10.csv'
path2ax = path2results[:path2results.rfind('.')] + '_ax.csv'
path2ay = path2results[:path2results.rfind('.')] + '_ay.csv'

with open(path2results, 'r') as fh:
    results = np.loadtxt(fh)
with open(path2ax, 'r') as fh:
    ax_res = np.loadtxt(fh)
with open(path2ay, 'r') as fh:
    ay_res = np.loadtxt(fh)

vx_result = results[:, 2][:, np.newaxis]
vy_result = results[:, 1][:, np.newaxis]
yaw_result = results[:, 0][:, np.newaxis]
steer_quantity = results[:, 3][:, np.newaxis]

ax_result = ax_res[:, np.newaxis]
ay_result = ay_res[:, np.newaxis]

plt.figure(figsize=(16, 8))
plt.plot(vx_result, color='r', linewidth=1.0)

# Add labels and title
plt.ylabel('Long. vel. vx [m/s]')
plt.xlabel('Time [steps]')
plt.title('Longitudinal Speed')

plt.grid(True)
plt.savefig('../test/steering_equilibrium/fx10_steer0_vx.png', format='png', dpi=300)
plt.close()

plt.figure(figsize=(16, 8))
fig, ax1 = plt.subplots(figsize=(16, 8))

ax2 = ax1.twinx()
ax1.plot(yaw_result, label='yaw rate', color='r', linewidth=1.0)
ax2.plot(steer_quantity, label='steering', color='b', linewidth=1.0)

ax1.set_xlabel('Time [steps]')
ax1.set_ylabel('Yaw rate [rad/s]')
ax2.set_ylabel('Steering input [rad]')


# Display the plot
plt.grid(True)
plt.savefig('../test/steering_equilibrium/fx10_steer0_yaw_rate.png', format='png', dpi=300)
plt.close()

