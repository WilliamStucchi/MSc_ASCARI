import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------

def extend_with_nan(array, desired_length):
    # Convert input to a numpy array if it is not already one
    array = np.asarray(array)
    current_length = len(array)

    if current_length < desired_length:
        # Create a new array filled with np.nan
        new_array = np.full(desired_length, np.nan)
        # Copy the original array into the beginning of the new array
        new_array[:current_length] = array
    else:
        # Truncate the array if it's longer than the desired length
        new_array = array[:desired_length]

    return new_array


# ----------------------------------------------------------------------------------------------------------------------

order = ['mid_high_perf', 'mid_perf', 'low_perf', 'high_grip (perf)', 'mid_grip', 'low_grip']
steer = []
forcex = []

print('ACQUIRING TEST DATA')
for i in tqdm(range(1, 4)):
    # Load set
    data = pd.read_csv('../CRT_data/test_car_perf/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
    data = data.drop(0, axis='rows')  # remove the row containing the measure units
    data.reset_index(drop=True, inplace=True)

    # Accelerations
    steer.append(data['driver_demands.steering'].to_numpy().astype(float))

    # Fx
    frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
    frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
    fr = (frl + frr) / 2
    ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
    ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
    ff = (ffl + ffr) / 2
    forcex.append((fr + ff) / 2)

for i in tqdm([0, 2, 4]):
    # Load set
    data = pd.read_csv('../CRT_data/test_road_grip/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
    data = data.drop(0, axis='rows')  # remove the row containing the measure units
    data.reset_index(drop=True, inplace=True)

    # Accelerations
    steer.append(data['driver_demands.steering'].to_numpy().astype(float))

    # Fx
    frl = data['Tire.Ground_Surface_Force_X.L2'].to_numpy().astype(float)
    frr = data['Tire.Ground_Surface_Force_X.R2'].to_numpy().astype(float)
    fr = (frl + frr) / 2
    ffl = data['Tire.Ground_Surface_Force_X.L1'].to_numpy().astype(float)
    ffr = data['Tire.Ground_Surface_Force_X.R1'].to_numpy().astype(float)
    ff = (ffl + ffr) / 2
    forcex.append((fr + ff) / 2)

print('END ACQUISITION TEST DATA')

max_len = 0
for el in steer:
    if len(el) > max_len:
        max_len = len(el)

steer[0] = extend_with_nan(steer[0], max_len)
steer[1] = extend_with_nan(steer[1], max_len)
steer[2] = extend_with_nan(steer[2], max_len)
steer[3] = extend_with_nan(steer[3], max_len)
steer[4] = extend_with_nan(steer[4], max_len)
steer[5] = extend_with_nan(steer[5], max_len)

forcex[0] = extend_with_nan(forcex[0], max_len)
forcex[1] = extend_with_nan(forcex[1], max_len)
forcex[2] = extend_with_nan(forcex[2], max_len)
forcex[3] = extend_with_nan(forcex[3], max_len)
forcex[4] = extend_with_nan(forcex[4], max_len)
forcex[5] = extend_with_nan(forcex[5], max_len)

colors = ['xkcd:blue', 'xkcd:green', 'xkcd:purple', 'xkcd:orange', 'xkcd:yellow', 'xkcd:black']

for i, el in enumerate(steer):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    plt.plot(el, label=order[i], color=colors[0])

    plt.title('Steer input test ' + order[i])
    plt.ylabel('steer')
    plt.xlabel('Time steps (10 ms)')
    plt.legend(loc='best')
    plt.grid()
    # plt.show()

    plt.savefig('../gg_plots/Steer_input_' + order[i], format='png')
    plt.close()

for i, el in enumerate(forcex):
    plt.figure(figsize=(25, 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    plt.plot(el, label=order[i], color=colors[0])

    plt.title('Longitudinal force input test ' + order[i])
    plt.ylabel('steer')
    plt.xlabel('Time steps (10 ms)')
    plt.legend(loc='best')
    plt.grid()
    # plt.show()

    plt.savefig('../gg_plots/Fx_input_' + order[i], format='png')
    plt.close()
