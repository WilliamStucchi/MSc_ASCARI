import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm


spindle = False
forces = True

save_format = 'eps'
directories = ['/grip_1_perf_100/', '/grip_1_perf_75/', '/grip_1_perf_50/',
               '/grip_08_perf_100/', '/grip_08_perf_75/', '/grip_08_perf_50/',
               '/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/']
# directories = ['/grip_06_perf_100/', '/grip_06_perf_75/', '/grip_06_perf_50/']
for k, dir_ in tqdm(enumerate(directories)):
    data = pd.read_csv('../CRT_data/parameters_study/' + dir_ + '/DemoSportsCar_mxp.csv', dtype=object)
    data = data.drop(0, axis='rows')  # remove the row containing the measure units
    data.reset_index(drop=True, inplace=True)

    if spindle:
        time = np.array(data['time.TIME'], dtype=float)

        driver_demand_steering = np.array(data['driver_demands.steering'], dtype=float)
        spindle_steering_L1 = np.array(data['Wheel.Spindle_Steer.L1'], dtype=float)
        spindle_steering_R1 = np.array(data['Wheel.Spindle_Steer.R1'], dtype=float)
        spindle_steering = (spindle_steering_L1 + spindle_steering_R1) / 2

        # PLOT
        plt.figure(figsize=(16, 8))
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_locator(MultipleLocator(0.025))
        plt.plot(spindle_steering, driver_demand_steering, color='r', linewidth=1.0)

        # Add labels and title
        plt.ylabel('Driver demand')
        plt.xlabel('Spindle steering')
        plt.title(dir_.replace('/', '').replace('_', ' ') + '%')

        # Display the plot
        plt.grid(True)
        # plt.show()
        plt.close()

    if forces:
        Fyf_L_real = np.array(data['Tire.Ground_Surface_Force_Y.L1'], dtype=float)
        Fyf_R_real = np.array(data['Tire.Ground_Surface_Force_Y.R1'], dtype=float)
        Fyf_real = (Fyf_L_real + Fyf_R_real) / 2
        Fyf_real[Fyf_real > 0] = 0

        Fyr_L_real = np.array(data['Tire.Ground_Surface_Force_Y.L2'], dtype=float)
        Fyr_R_real = np.array(data['Tire.Ground_Surface_Force_Y.R2'], dtype=float)
        Fyr_real = (Fyr_L_real + Fyr_R_real) / 2
        Fyr_real[Fyr_real > 0] = 0

        Fzf_L_real = np.array(data['Tire.Ground_Surface_Force_Z.L1'], dtype=float)
        Fzf_R_real = np.array(data['Tire.Ground_Surface_Force_Z.R1'], dtype=float)
        Fzf_real = (Fzf_L_real + Fzf_R_real) / 2
        Fzf_real[Fzf_real < 0] = 0

        Fzr_L_real = np.array(data['Tire.Ground_Surface_Force_Z.L2'], dtype=float)
        Fzr_R_real = np.array(data['Tire.Ground_Surface_Force_Z.R2'], dtype=float)
        Fzr_real = (Fzr_L_real + Fzr_R_real) / 2
        Fzr_real[Fzr_real < 0] = 0

        alphaf_L_real = np.array(data['Tire.Lateral_Slip_Without_Lag.L1'], dtype=float)
        alphaf_R_real = np.array(data['Tire.Lateral_Slip_Without_Lag.R1'], dtype=float)
        alphaf_real = (alphaf_L_real + alphaf_R_real) / 2
        alphaf_real[alphaf_real < 0] = 0
        alphaf_real = alphaf_real * 180 / np.pi

        alphar_L_real = np.array(data['Tire.Lateral_Slip_Without_Lag.L2'], dtype=float)
        alphar_R_real = np.array(data['Tire.Lateral_Slip_Without_Lag.R2'], dtype=float)
        alphar_real = (alphar_L_real + alphar_R_real) / 2
        alphar_real[alphar_real < 0] = 0
        alphar_real = alphar_real * 180 / np.pi

        #sim = pd.read_csv('../matlab/test_' + dir_.replace('/', '') + '.csv', dtype=object)
        sim = pd.read_csv('../matlab/paths_with_pacejka_1/bike/sim_test_' + dir_.replace('/', '') + '.csv', dtype=object)
        sim.reset_index(drop=True, inplace=True)

        Fyf_sim = np.array(sim['Fyf'], dtype=float)
        Fyf_sim[Fyf_sim > 0] = 0
        Fyr_sim = np.array(sim['Fyr'], dtype=float)
        Fyr_sim[Fyr_sim > 0] = 0

        Fzf_sim = np.array(sim['Fzf'], dtype=float)
        Fzf_sim[Fzf_sim < 0] = 0
        Fzr_sim = np.array(sim['Fzr'], dtype=float)
        Fzr_sim[Fzr_sim < 0] = 0

        alphaf_sim = np.array(sim['alphaf'], dtype=float)
        alphaf_sim[alphaf_sim < 0] = 0
        alphaf_sim = alphaf_sim * 180 / np.pi
        alphar_sim = np.array(sim['alphar'], dtype=float)
        alphar_sim[alphar_sim < 0] = 0
        alphar_sim = alphar_sim * 180 / np.pi

        # PLOT
        plt.figure(figsize=(12, 10))
        # Aumentare il font size per tutto il grafico
        plt.rc('font', size=20)  # Modifica la grandezza del font globalmente
        plt.rc('axes', titlesize=25)  # Titolo degli assi
        plt.rc('axes', labelsize=25)  # Etichette degli assi
        plt.rc('xtick', labelsize=25)  # Etichette dei ticks su x
        plt.rc('ytick', labelsize=25)  # Etichette dei ticks su y
        plt.rc('legend', fontsize=20)  # Legenda
        plt.plot(alphaf_real, -Fyf_real / Fzf_real, label='CRT data', color='r', linewidth=2.0)
        plt.plot(alphaf_sim, -Fyf_sim / Fzf_sim, label='Bicycle model', color='b', linewidth=2.0)

        # Add labels and title
        plt.ylabel('Fy / Fz', labelpad=20)
        plt.xlabel('Slip angle [deg]', labelpad=15)
        plt.title('Test: ' + dir_.replace('/', '').replace('_', ' ')
                  .replace('grip', 'μ =') + '%' + ' front', pad=20)
        plt.legend(loc='best')

        # Display the plot
        plt.grid(True)
        # plt.show()
        plt.savefig('../matlab/tires_with_pacejka_1/tires/front_tire_' + dir_.replace('/', '') +
                    '.' + save_format, format=save_format, dpi=300)
        plt.close()

        # PLOT
        plt.figure(figsize=(12, 10))
        plt.plot(alphar_real, -Fyr_real / Fzr_real, label='CRT data', color='r', linewidth=2.0)
        plt.plot(alphar_sim, -Fyr_sim / Fzr_sim, label='Bicycle model', color='b', linewidth=2.0)

        # Add labels and title
        plt.ylabel('Fy / Fz', labelpad=20)
        plt.xlabel('Slip angle [deg]', labelpad=15)
        plt.title('Test: ' + dir_.replace('/', '').replace('_', ' ')
                  .replace('grip', 'μ =') + '%' + ' rear', pad=20)
        plt.legend(loc='best')

        # Display the plot
        plt.grid(True)
        # plt.show()
        plt.savefig('../matlab/tires_with_pacejka_1/tires/rear_tire_' + dir_.replace('/', '') +
                    '.' + save_format, format=save_format, dpi=300)
        plt.close()


