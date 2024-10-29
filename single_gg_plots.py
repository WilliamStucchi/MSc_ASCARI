import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# gg-plot of the dataset created in VI-Grade CarRealTime piccolo
# ----------------------------------------------------------------------------------------------------------------------

train_ax = []
train_ax_g = []
train_ay = []
train_ay_g = []
test_ax = []
test_ay = []

# Acquire training data
directories = ['train_car_perf/perf_25/']

for i in range(17):
    temp = []
    train_ax.append(temp)

for i in range(17):
    temp = []
    train_ay.append(temp)

print('ACQUIRING TRAINING DATA')
for k, dir_ in enumerate(directories):
    print('ACQUIRING TRAINING DATA from ' + dir_)
    for i in tqdm(range(0, 17, 1)):
        # Load set
        data = pd.read_csv('../new_CRT/' + dir_ + 'test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        data = data.drop(0, axis='rows')  # remove the row containing the measure units
        data.reset_index(drop=True, inplace=True)

        # Accelerations
        train_ax[i].append(data['Vehicle_States.longitudinal_acc_wrt_road'].to_numpy().astype(float))
        train_ay[i].append(data['Vehicle_States.lateral_acc_wrt_road'].to_numpy().astype(float))

    print('END ACQUISITION TRAINING DATA from ' + dir_)

print('PLOTTING')
# Plot gg-plot
for i in range(17):
    plt.figure(figsize=(10, 8))

    # Car perf
    plt.scatter(train_ay[i], train_ax[i], alpha=.9, s=5, color='blue')

    legend = plt.legend(loc='best')
    for lh in legend.legendHandles:
        lh.set_alpha(0.8)  # Increase opacity of legend items
    plt.title('gg-plot test ' + str(i))
    plt.ylabel('Longitudinal acceleration [g]')
    plt.xlabel('Lateral acceleration [g]')

    plt.grid(True)
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='k', linewidth=0.5, linestyle='--')

    fig1 = plt.gcf()
    plt.show()
    # fig.savefig('../gg_plots/thesis_plots/dataset_mu1.png', format='png')