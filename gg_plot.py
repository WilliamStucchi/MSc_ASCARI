import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
from tqdm import tqdm
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# gg-plot of the dataset created in VI-Grade CarRealTime
# ----------------------------------------------------------------------------------------------------------------------

train_ax = []
train_ax_g = []
train_ay = []
train_ay_g = []
test_ax = []
test_ay = []

filename = 'gg_test_mu1'
# Acquire training data
directories = ['parameters_study/grip_1_perf_100/', 'parameters_study/grip_1_perf_75/', 'parameters_study/grip_1_perf_50/']
titles = ['100% performance', '75% performance', '50% performance', '25% performance']
colors = {'100% performance': 'red', '75% performance': 'blue', '50% performance': 'green', '25% performance': 'orange'}
color_values = [colors[perf] for perf in titles]

for i in range(len(directories)):
    temp = []
    train_ax.append(temp)

for i in range(len(directories)):
    temp = []
    train_ay.append(temp)

"""print('ACQUIRING TRAINING DATA')
for k, dir_ in enumerate(directories):
    print('ACQUIRING TRAINING DATA from ' + dir_)
    for i in tqdm(range(0, 17, 1)):
        # Load set
        data = pd.read_csv('../CRT_data/' + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Accelerations
        train_ax[k].append(data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float))
        train_ay[k].append(data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float))

    print('END ACQUISITION TRAINING DATA from ' + dir_)"""

print('ACQUIRING TEST DATA')
for k, dir_ in enumerate(directories):
    print('ACQUIRING TRAINING DATA from ' + dir_)
    # Load set
    data = pd.read_csv('../CRT_data/' + dir_ + '/DemoSportsCar_mxp.csv', dtype=object)
    data = data.drop(0, axis='rows')  # remove the row containing the measure units
    data.reset_index(drop=True, inplace=True)

    # Accelerations
    train_ax[k].append(data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float))
    train_ay[k].append(data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float))

    print('END ACQUISITION TRAINING DATA from ' + dir_)

flattened_ax = [np.concatenate(sublist).tolist() for sublist in train_ax]
flattened_ay = [np.concatenate(sublist).tolist() for sublist in train_ay]

# g-scaled data
for i in range(len(directories)):
    temp = []
    train_ax_g.append(temp)

for i in range(len(directories)):
    temp = []
    train_ay_g.append(temp)

for i in range(len(flattened_ax)):
    train_ax_g[i] = np.array(flattened_ax[i]) / 9.81
    train_ay_g[i] = np.array(flattened_ay[i]) / 9.81

"""print('ACQUIRING TEST DATA')

# Acquire test data
for i in range(len(directories)):
    temp = []
    test_ax.append(temp)

for i in range(len(directories)):
    temp = []
    test_ay.append(temp)
    
for i in tqdm(range(5)):
    # Load set
    data = pd.read_csv('../CRT_data/test_road_grip/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
    data = data.drop(0, axis='rows')  # remove the row containing the measure units
    data.reset_index(drop=True, inplace=True)

    # Accelerations
    test_ax[i].append(data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float))
    test_ay[i].append(data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float))

print('END ACQUISITION TEST DATA')


for i in range(0, len(test_ax)):
    test_ax[i] = np.array(test_ax[i]) / 9.81
    test_ay[i] = np.array(test_ay[i]) / 9.81"""

"""print('CREATION OF THE ELLIPSES')
# Ellipses
center_x, center_y = -1.155, 0.02
a, b = 0.04, 0.1

assert len(train_ax_g) == len(train_ay_g)

df = pd.DataFrame({'x': test_ax[0].reshape(test_ax[0].shape[1]), 'y': test_ay[0].reshape(test_ay[0].shape[1])})
count_test = 0
for index, row in df.iterrows():
    if ((row['y'] - center_y) ** 2 / b ** 2) + ((row['x'] - center_x) ** 2 / a ** 2) <= 1:
        count_test += 1

df = pd.DataFrame({'x': train_ax_g, 'y': train_ay_g})
count_train = 0
for index, row in df.iterrows():
    if ((row['y'] - center_y)**2 / b**2) + ((row['x'] - center_x)**2 / a**2) <= 1:
        count_train += 1

print('Computing number of points inside ellipse...')
print('The number of training points inside the ellipse is ', count_train)
print('The number of test points inside the ellipse is ', count_test)
print('The total number of elements is ', len(train_ax_g))
print('The elements outside the ellipse are ', len(train_ax_g) - count_train)"""

print('PLOTTING')
# Plot gg-plot
plt.figure(figsize=(15, 13))
# Aumentare il font size per tutto il grafico
plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
plt.rc('axes', titlesize=25)   # Titolo degli assi
plt.rc('axes', labelsize=25)   # Etichette degli assi
plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
plt.rc('legend', fontsize=18)  # Legenda

scatters = []
for i in range(len(directories)):
    scatter = plt.scatter(train_ay_g[i], train_ax_g[i], alpha=.9, s=10, label=titles[i], c=color_values[i], cmap='Set1')
    scatters.append(scatter)

"""ellipse = plt.matplotlib.patches.Ellipse((center_x, center_y), 2*a, 2*b, edgecolor='lime', facecolor='none', linestyle='dashed')
plt.gca().add_patch(ellipse)"""

# Customizing the legend marker size without affecting the scatter plot
plt.legend(handler_map={scatter: HandlerPathCollection(marker_pad=0.3, numpoints=1) for scatter in scatters},
           scatterpoints=1, markerscale=5)
plt.title('gg-plot')
plt.ylabel('Longitudinal acceleration [g]')
plt.xlabel('Lateral acceleration [g]')

plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
plt.axvline(0, color='k', linewidth=0.5, linestyle='--')

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('../gg_plots/thesis_plots/' + filename + '.png', format='png')

