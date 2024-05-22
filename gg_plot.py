import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# gg-plot of the dataset created in VI-Grade CarRealTime
# ----------------------------------------------------------------------------------------------------------------------

train_x = []
train_y = []
test_ax = []
test_ay = []

# Acquire training data
directories = ['mid_low_perf/', 'mid_perf/', 'mid_high_perf/', 'high_perf/']

print('ACQUIRING TRAINING DATA')
for k, dir_ in enumerate(directories):
    print('ACQUIRING TRAINING DATA from ' + dir_)
    for i in tqdm(range(0, 17, 1)):
        # Load set
        data = pd.read_csv('../CRT_data/' + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Accelerations
        train_x.append(data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float))
        train_y.append(data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float))

    print('END ACQUISITION TRAINING DATA from ' + dir_)

# Acquire test data
for i in range(0, 4):
    temp = []
    test_ax.append(temp)

for i in range(0, 4):
    temp = []
    test_ay.append(temp)

print('ACQUIRING TEST DATA')
for i in tqdm(range(0, 4, 1)):
    # Load set
    data = np.loadtxt('NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/test_set_' + str(i) + '.csv',
                      delimiter=',',
                      dtype=object)

    # Accelerations
    test_ax[i].append(data[:, 3].astype(float))
    test_ay[i].append(data[:, 4].astype(float))

print('END ACQUISITION TEST DATA')

# Manipulate data
flat_train_ax = []
for row in train_x:
    flat_train_ax.extend(row)

flat_train_ay = []
for row in train_y:
    flat_train_ay.extend(row)

# g-scaled data
train_ax_g = np.array(flat_train_ax) / 9.81
train_ay_g = np.array(flat_train_ay) / 9.81

for i in range(0, len(test_ax)):
    test_ax[i] = np.array(test_ax[i]) / 9.81
    test_ay[i] = np.array(test_ay[i]) / 9.81

# Ellipses
center_x, center_y = 0, 0
a, b = 0.7, 0.7

assert len(train_ax_g) == len(train_ay_g)

df = pd.DataFrame({'x': train_ax_g, 'y': train_ay_g})
count = 0
for index, row in df.iterrows():
    if ((row['y'] - center_y)**2 / b**2) + ((row['x'] - center_x)**2 / a**2) <= 1:
        count += 1

print('Computing number of points inside ellipse...')
print('The number of points inside the ellipse is ', count)
print('The total number of elements is ', len(train_ax_g))
print('The elements outside the ellipse are ', len(train_ax_g) - count)

# Plot gg-plot
plt.figure(figsize=(10, 8))
plt.scatter(train_ay_g, train_ax_g, alpha=0.1, s=5, label='Training Data', color='blue')
plt.scatter(test_ay[0], test_ax[0], alpha=0.1, s=30, label='high_perf Data', color='red')
plt.scatter(test_ay[1], test_ax[1], alpha=0.1, s=30, label='mid_high_perf Data', color='black')
plt.scatter(test_ay[2], test_ax[2], alpha=0.1, s=30, label='mid_perf Data', color='green')
plt.scatter(test_ay[3], test_ax[3], alpha=0.1, s=30, label='mid_low_perf Data', color='yellow')
ellipse = plt.matplotlib.patches.Ellipse((center_x, center_y), 2*a, 2*b, edgecolor='lime', facecolor='none', linestyle='dashed')
plt.gca().add_patch(ellipse)
legend = plt.legend(loc='best')
for lh in legend.legendHandles:
    lh.set_alpha(0.8)  # Increase opacity of legend items
plt.title('gg-plot')
plt.ylabel('Longitudinal acceleration [g]')
plt.xlabel('Lateral acceleration [g]')

plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
plt.axvline(0, color='k', linewidth=0.5, linestyle='--')

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('../gg_plot.png', format='png')
