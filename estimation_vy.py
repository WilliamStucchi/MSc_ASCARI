import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../CRT_data/test_car_perf/test_0/DemoSportsCar_mxp.csv', dtype=object)
data = data.drop(0, axis='rows')  # remove the row containing the measure units
data.reset_index(drop=True, inplace=True)

# Yaw rate
yaw_rate = data['Vehicle_States.yaw_angular_vel_wrt_road'].to_numpy().astype(float)

# Vy
uy = data['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float)

# Vx
vx = data['Vehicle_States.longitudinal_vel_wrt_road'].to_numpy().astype(float)

ax = data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float)
ay = data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float)

vy = vy

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