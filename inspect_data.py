import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------

def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))


# ----------------------------------------------------------------------------------------------------------------------

directories = ['train_car_perf/mid_low_perf/', 'train_car_perf/mid_perf/', 'train_car_perf/mid_high_perf/',
                   'train_car_perf/high_perf/']

training_set = np.loadtxt('scirob_submission/Model_Learning/data/new/train_data_step_no_shuffle_1_est.csv',
                          delimiter=',')

previous_idx = 0
for el in directories:
    for i in tqdm(range(0, 17)):
        original = pd.read_csv('../CRT_data/' + el + '/test_' + str(i) + '/DemoSportsCar_mxp.csv', dtype=object)
        original = original.drop(0, axis='rows')  # remove the row containing the measure units
        original.reset_index(drop=True, inplace=True)

        estimated = training_set[previous_idx:previous_idx + original.shape[0]]

        previous_idx = previous_idx + original.shape[0]

        vy_est = []
        for idx in range(0, original.shape[0]):
            if idx == 0:
                vy_est = np.append(vy_est, estimated[idx, 1])
                vy_est = np.append(vy_est, estimated[idx, 6])
                vy_est = np.append(vy_est, estimated[idx, 11])
                vy_est = np.append(vy_est, estimated[idx, 16])
            else:
                vy_est = np.append(vy_est, estimated[idx, 16])

        plt.figure(figsize=(25, 10))
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

        plt.plot(vy_est, label='Estimate', color='tab:red')
        plt.plot(original['Vehicle_States.lateral_vel_wrt_road'].to_numpy().astype(float), label='Original',
                 color='tab:blue')

        plt.ylabel('Lateral velocity [m/s]')
        plt.xlabel('Time steps (10 ms)')
        plt.legend()
        plt.grid()

        # plt.show()
        plt.savefig('../test/vy_estimation/car_perf/' + el[find_second_last(el, '/') + 1:-1] + '_test_' + str(i), format='png')

        plt.close()
