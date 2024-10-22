# Nathan Spielberg
# DDL 10.17.2018

# Data Generation Functions for use in data generation
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz
import os
import math


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def euler_int(x, xdot, dt):
    return x + dt * xdot


# tire model
def fiala(alpha, Ca, mu, fz):
    alpha_slide = np.abs(np.arctan(3 * mu * fz / Ca))
    if np.abs(alpha) < alpha_slide:
        fy = (-Ca * np.arctan(alpha) + ((Ca ** 2) / (3 * mu * fz)) * (np.abs(np.arctan(alpha))) * np.arctan(alpha) -
              ((Ca ** 3) / (9 * (mu ** 2) * (fz ** 2))) * (np.arctan(alpha) ** 3) * (1 - 2 * mu / (3 * mu)))
    else:
        fy = -mu * fz * np.sign(alpha)
    return fy


def pacejka_magic_formula(a, b, c, e, alpha, Fz):
    return - c / b / a * Fz * np.sin(b * np.arctan(a * (1 - e) * np.tan(alpha) + e * np.arctan(a * np.tan(alpha))))


def sample(Param, Veh, mu):
    Ux_lim = Param["UX_LIM"]
    g = Veh['g']
    a = Veh["a"]
    b = Veh["b"]
    m = Veh["m"]
    mu1 = Veh['mu']
    mu06 = Veh['mu_2']
    Cr = Veh["Cr"] * Veh['rear_normal_load']
    mul_margin = Veh['mul_margin']
    crt_yawrate_max_mu1 = Veh['CRT_yawrate_max_mu1']
    crt_yawrate_max_mu06 = Veh['CRT_yawrate_max_mu06']
    max_physical_steer_mu1 = Veh['max_physical_steer_mu1']
    max_physical_steer_mu06 = Veh['max_physical_steer_mu06']

    # First sample vx
    vx = np.random.uniform(low=3.0, high=Ux_lim)

    # Yaw rate
    if mu == mu1:
        r_lim = min((mu * g) / vx, crt_yawrate_max_mu1) * mul_margin
    else:
        r_lim = min((mu * g) / vx, crt_yawrate_max_mu06) * mul_margin

    r = np.random.uniform(low=-r_lim, high=r_lim)

    # Lateral velocity
    vy_lim = (3 * vx * m * 9.81 * mu) / Cr * (a / (a + b)) + b * r_lim
    vy = np.random.uniform(low=-vy_lim, high=vy_lim)

    # Steering angle (the max steering angle is the same at different mu, but I wanted to separate them to make a distinction
    if mu == mu1:
        delta_lim = min(abs(((r * (a+b)) / vx) * mul_margin), max_physical_steer_mu1)
    else:
        delta_lim = min(abs(((r * (a+b)) / vx) * mul_margin), max_physical_steer_mu06)

    delta = np.random.uniform(low=-delta_lim, high=delta_lim)

    # Longitudinal force
    ay = r * vx
    ax_lim = (np.sqrt(np.power(mu * g * mul_margin, 2) - np.power(ay, 2)))
    ax = np.random.uniform(low=-ax_lim, high=ax_lim)
    Fx = (m * ax) / 4

    return r, vy, vx, delta, Fx


def sample_contd(Param, Veh, mu, last_delta, last_Fx):
    mu1 = Veh['mu']
    mu06 = Veh['mu_2']
    delta_lim_mu1 = Veh['delta_lim']
    delta_lim_mu06 = Veh['delta_lim']
    max_pos_fx_mu1 = Veh['max_pos_fx_mu1']
    max_neg_fx_mu1 = Veh['max_neg_fx_mu1']
    max_pos_fx_mu06 = Veh['max_pos_fx_mu06']
    max_neg_fx_mu06 = Veh['max_neg_fx_mu06']

    mean_deltaFx_mu1 = Veh['mean_deltaFx_mu1']
    var_deltaFx_mu1 = Veh['var_deltaFx_mu1']
    mean_deltaSteer_mu1 = Veh['mean_deltaSteer_mu1']
    var_deltaSteer_mu1 = Veh['var_deltaSteer_mu1']

    mean_deltaFx_mu06 = Veh['mean_deltaFx_mu06']
    var_deltaFx_mu06 = Veh['var_deltaFx_mu06']
    mean_deltaSteer_mu06 = Veh['mean_deltaSteer_mu06']
    var_deltaSteer_mu06 = Veh['var_deltaSteer_mu06']

    # Steering angle
    if mu == mu1:
        delta = last_delta + np.random.normal(loc=mean_deltaSteer_mu1, scale=np.sqrt(var_deltaSteer_mu1))
        if delta > 0:
            delta = min(delta, delta_lim_mu1)
        else:
            delta = max(delta, -delta_lim_mu1)
    elif mu == mu06:
        delta = last_delta + np.random.normal(loc=mean_deltaSteer_mu06, scale=np.sqrt(var_deltaSteer_mu06))
        if delta > 0:
            delta = min(delta, delta_lim_mu06)
        else:
            delta = max(delta, -delta_lim_mu06)

    assert abs(delta) < 0.44

    # Longitudinal force
    if mu == mu1:
        Fx = last_Fx + np.random.normal(loc=mean_deltaFx_mu1, scale=np.sqrt(var_deltaFx_mu1))
        if Fx > 0:
            Fx = min(Fx, max_pos_fx_mu1)
        else:
            Fx = max(Fx, max_neg_fx_mu1)

        assert Fx >= max_neg_fx_mu1 and Fx <= max_pos_fx_mu1
    elif mu == mu06:
        Fx = last_Fx + np.random.normal(loc=mean_deltaFx_mu06, scale=np.sqrt(var_deltaFx_mu06))
        if Fx > 0:
            Fx = min(Fx, max_pos_fx_mu06)
        else:
            Fx = max(Fx, max_neg_fx_mu06)

        assert Fx >= max_neg_fx_mu06 and Fx <= max_pos_fx_mu06

    return delta, Fx


def step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx, Param, Veh, beta_t, a_f_in, a_r_in, af, ar, bf, br, ef, er, lf, lr, tf, tr, DT):
    a = Veh["a"]
    b = Veh["b"]
    cr = Veh["Cr"]
    cf = Veh["Cf"]
    m = Veh["m"]
    Izz = Veh["Izz"]
    sig_f = Veh["sig_f"]
    sig_r = Veh["sig_r"]
    Fzf0 = Veh['Fzf0']
    Fzr0 = Veh['Fzr0']
    CxA = Veh['CxA']
    CzfA = Veh['CzfA']
    CzrA = Veh['CzrA']
    air_density = Veh['rho_air']

    # Calculate slip angles:
    if Param["RELAX_LENGTH"]:
        a_f = a_f_in
        a_r = a_r_in

        # Then run updates to get next slips
        V = np.sqrt(Ux_t ** 2 + Uy_t ** 2)
        a_f_t1 = a_f + DT * ((V / sig_f) * (np.arctan((Uy_t + a * r_t) / Ux_t) - del_t - a_f))
        a_r_t1 = a_r + DT * ((V / sig_r) * (np.arctan((Uy_t - b * r_t) / Ux_t) - a_r))
    else:
        """a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)"""

        # We don't use next slips in this case.
        a_f_t1 = a_f_in + DT * ((beta_t - del_t + a * r_t / Ux_t - a_f_in) / (lf / Ux_t))
        a_r_t1 = a_r_in + DT * ((beta_t - 0.0 - b * r_t / Ux_t - a_r_in) / (lr / Ux_t))

    if Param["WEIGHT_TRANSFER"]:
        # Need another component in here for Fy sin del!
        print('weight transfer')
        # First with WT Calc ax:
        # ax = (Fxf * np.cos(del_t) + Fxr) / m

        # Now Calc Normal loads which are now dep on ax
        # Fzf = (b / l) * m * 9.81 - (h / l) * m * ax
        # Fzr = (a / l) * m * 9.81 + (h / l) * m * ax
    else:
        # Fzf = (m * 9.81 * b) / (l)
        # Fzr = (m * 9.81 * a) / (l)
        # new formula for the computation of Fzf,r
        Fzf = Fzf0 + 0.5 * air_density * CzfA * (Ux_t ** 2)
        Fzr = Fzr0 + 0.5 * air_density * CzrA * (Ux_t ** 2)

    # Calculate Forces
    # Fyf = fiala(a_f, Cf, mu, Fzf)
    # Fyr = fiala(a_r, Cr, mu, Fzr)
    Fyf = pacejka_magic_formula(af, bf, cf, ef, a_f_t1, Fzf)
    Fyr = pacejka_magic_formula(ar, br, cr, er, a_r_t1, Fzr)

    """# Calculate Derivatives
    Uy_dot_t = (Fyf * np.cos(del_t) + Fx * np.sin(del_t) + Fyr) / m
    Uy_dot = Uy_dot_t - r_t * Ux_t

    r_dot = (a * Fyf * np.cos(del_t) + a * Fx * np.sin(del_t) - b * Fyr) / Izz

    Ux_dot_t = (Fx * np.cos(del_t) + Fx - Fyf * np.sin(del_t)) / m
    Ux_dot = Ux_dot_t + r_t * Uy_t

    Uy_t_1 = Uy_t + DT * Uy_dot
    r_t_1 = r_t + DT * r_dot
    Ux_t_1 = Ux_t + DT * Ux_dot"""

    # New model
    r_dot = ((a - tf) * Fyf - (b + tr) * Fyr) / Izz
    r_t_1 = r_t + DT * r_dot

    ax       = ((4 * Fx - 0.5 * air_density * CxA * (Ux_t ** 2)) / m)
    Ux_t_1   = Ux_t + DT * (ax + r_t_1 * Uy_t)

    beta_t_1 = beta_t + DT * ((Fyf + Fyr) / m / Ux_t - r_t)

    # ay       = Ux_t * (r_t_1 + (beta_t_1 - beta_t) / DT)
    ay = (Fyf + Fyr) / m
    Uy_t_1   = Uy_t + DT * (ay - r_t_1 * Ux_t_1)


    return r_t_1, Uy_t_1, Ux_t_1, a_f_t1, a_r_t1, r_dot, ay, ax, beta_t_1


def gen_data_mod(Param, Veh):
    yaw_rates = []
    steering_angles = []
    longitudinal_vels = []

    # Unpack dicts
    T = Param["T"]
    N_SAMPLES = Param["N_SAMPLES"]
    N_STATE_INPUT = Param["N_STATE_INPUT"]
    TWO_FRIC = Param["TWO_FRIC"]
    Fx_norm = Param["FX_NORM"]

    mu1 = Veh["mu"]
    mu06 = Veh["mu_2"]
    a = Veh["a"]
    b = Veh["b"]

    gen_data = np.zeros((N_SAMPLES, (N_STATE_INPUT * T + 3)))  # 2 targets r, Uy and Ux

    SW_rate = Veh['SW_rate']
    cf = Veh['Cf']
    cr = Veh['Cr']
    tf = 0
    tr = 0
    hf = Veh['Hf']
    hr = Veh['Hr']
    pf = Veh['Pf']
    pr = Veh['Pr']
    muf_mu1 = Veh['muyf_mu1']
    muf_mu06 = Veh['muyf_mu06']
    mur_mu1 = Veh['muyr_mu1']
    mur_mu06 = Veh['muyr_mu06']
    peakf_mu1 = Veh['Peakyf_mu1']
    peakf_mu06 = Veh['Peakyf_mu06']
    peakr_mu1 = Veh['Peakyr_mu1']
    peakr_mu06 = Veh['Peakyr_mu06']

    bf = 1 + (1 - 2 / np.pi * np.arcsin(pf))
    br = 1 + (1 - 2 / np.pi * np.arcsin(pr))
    af_mu1 = cf / bf / muf_mu1
    ar_mu1 = cr / br / mur_mu1
    ef_mu1 = (af_mu1 * peakf_mu1 - np.tan(np.pi / 2 / bf)) / (af_mu1 * peakf_mu1 - np.arctan(af_mu1 * peakf_mu1))
    er_mu1 = (ar_mu1 * peakr_mu1 - np.tan(np.pi / 2 / br)) / (ar_mu1 * peakr_mu1 - np.arctan(ar_mu1 * peakr_mu1))
    lf = hf
    lr = hr

    af_mu06 = cf / bf / muf_mu06
    ar_mu06 = cr / br / mur_mu06
    ef_mu06 = (af_mu06 * peakf_mu06 - np.tan(np.pi / 2 / bf)) / (af_mu06 * peakf_mu06 - np.arctan(af_mu06 * peakf_mu06))
    er_mu06 = (ar_mu06 * peakr_mu06 - np.tan(np.pi / 2 / br)) / (ar_mu06 * peakr_mu06 - np.arctan(ar_mu06 * peakr_mu06))

    print("Starting Data Generation: ")
    # print("else")
    for i in tqdm(range(N_SAMPLES)):

        if TWO_FRIC and i < N_SAMPLES / 2:
            for j in range(T):
                # FIll out the delay states!
                if j == 0:
                    r_t, Uy_t, Ux_t, del_t, Fx_t = sample(Param, Veh, mu1)
                    gen_data[i, 0] = r_t
                    gen_data[i, 1] = Uy_t
                    gen_data[i, 2] = Ux_t
                    gen_data[i, 3] = del_t * SW_rate  # steering angle from wheels to steering wheel
                    gen_data[i, 4] = Fx_t

                    # Calculate the initial slip from SS assumptions
                    a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                    a_r = np.arctan((Uy_t - b * r_t) / Ux_t)
                    beta_t = 0  # initialization

                    yaw_rates.append(r_t)
                    steering_angles.append(gen_data[i, 3])
                    longitudinal_vels.append(Ux_t)
                else:
                    # Propagate the next state w/ dynamics model!
                    r_t, Uy_t, Ux_t, a_f, a_r, _, _, _, beta_t = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t, Param, Veh,
                                                                               beta_t, a_f, a_r, af_mu1, ar_mu1, bf,
                                                                               br, ef_mu1, er_mu1, lf,
                                                                               lr, tf, tr, Param["DT"])

                    # And cache the next state in our data matrix
                    gen_data[i, (N_STATE_INPUT * j)] = r_t
                    gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                    gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                    # Sample the next set of controls!
                    last_delta = gen_data[i, (N_STATE_INPUT * (j - 1)) + 3] / SW_rate
                    last_fx = gen_data[i, (N_STATE_INPUT * (j - 1)) + 4]
                    del_t, Fx_t = sample_contd(Param, Veh, mu1, last_delta, last_fx)

                    # Then Cache them in the data!
                    gen_data[i, (N_STATE_INPUT * j) + 3] = del_t * SW_rate
                    gen_data[i, (N_STATE_INPUT * j) + 4] = Fx_t

                    yaw_rates.append(r_t)
                    steering_angles.append(gen_data[i, (N_STATE_INPUT * j) + 3])
                    longitudinal_vels.append(Ux_t)

            # For the Last T step we will get the vel targets we want to predict!
            _, _, _, _, _, r_dot_t, Uy_dot_t, Ux_dot_t, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t,
                                                                          Param, Veh, beta_t, a_f, a_r,
                                                                          af_mu1, ar_mu1, bf, br, ef_mu1, er_mu1,
                                                                          lf, lr, tf, tr, Param["DT"])

            # Cache the targets!
            gen_data[i, (N_STATE_INPUT * T)] = r_dot_t
            gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_dot_t
            gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_dot_t

        else:
            # run a single simulation for one trajectory of length T
            for j in range(T):
                # FIll out the delay states!
                if j == 0:
                    r_t, Uy_t, Ux_t, del_t, Fx_t = sample(Param, Veh, mu06)
                    gen_data[i, 0] = r_t
                    gen_data[i, 1] = Uy_t
                    gen_data[i, 2] = Ux_t
                    gen_data[i, 3] = del_t * SW_rate  # steering angle from wheels to steering wheel
                    gen_data[i, 4] = Fx_t

                    # Calculate the initial slip from SS assumptions
                    a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                    a_r = np.arctan((Uy_t - b * r_t) / Ux_t)
                    beta_t = 0  # initialization

                    yaw_rates.append(r_t)
                    steering_angles.append(gen_data[i, 3])
                    longitudinal_vels.append(Ux_t)
                else:
                    # Propagate the next state w/ dynamics model!
                    r_t, Uy_t, Ux_t, a_f, a_r, _, _, _, beta_t = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t, Param, Veh,
                                                                               beta_t, a_f, a_r, af_mu06, ar_mu06, bf,
                                                                               br, ef_mu06, er_mu06, lf, lr, tf, tr,
                                                                               Param["DT"])

                    # And cache the next state in our data matrix
                    gen_data[i, (N_STATE_INPUT * j)] = r_t
                    gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                    gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                    # Sample the next set of controls!
                    last_delta = gen_data[i, (N_STATE_INPUT * (j - 1)) + 3] / SW_rate
                    last_fx = gen_data[i, (N_STATE_INPUT * (j - 1)) + 4]
                    del_t, Fx_t = sample_contd(Param, Veh, mu06, last_delta, last_fx)

                    # Then Cache them in the data!
                    gen_data[i, (N_STATE_INPUT * j) + 3] = del_t * SW_rate
                    gen_data[i, (N_STATE_INPUT * j) + 4] = Fx_t

                    yaw_rates.append(r_t)
                    steering_angles.append(gen_data[i, (N_STATE_INPUT * j) + 3])
                    longitudinal_vels.append(Ux_t)

            # For the Last T step we will get the vel targets we want to predict!
            _, _, _, _, _, r_dot_t, Uy_dot_t, Ux_dot_t, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t,
                                                                          Param, Veh, beta_t, a_f, a_r,
                                                                          af_mu06, ar_mu06, bf, br, ef_mu06, er_mu06,
                                                                          lf, lr, tf, tr, Param["DT"])

            # Cache the targets!
            gen_data[i, (N_STATE_INPUT * T)] = r_dot_t
            gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_dot_t
            gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_dot_t


    return gen_data, yaw_rates, steering_angles, longitudinal_vels


def add_noise(gen_data, Param):
    N_STATE_INPUT = Param["N_STATE_INPUT"]
    T = Param["T"]

    data_noise = np.zeros(shape=(gen_data.shape))
    N_Samples = gen_data.shape[0]

    for i in range(T):
        data_noise[:, N_STATE_INPUT * i] = np.random.normal(0.0, .01, size=(N_Samples))
        data_noise[:, N_STATE_INPUT * i + 1] = np.random.normal(0.0, .01, size=(N_Samples))
        data_noise[:, N_STATE_INPUT * i + 2] = np.random.normal(0.0, .001, size=(N_Samples))
        data_noise[:, N_STATE_INPUT * i + 3] = np.random.normal(0.0, .001, size=(N_Samples))
        data_noise[:, N_STATE_INPUT * i + 4] = np.random.normal(0.0, .001, size=(N_Samples)) / Param["Fx_norm"]

    data_noise[:, N_STATE_INPUT * T] = np.random.normal(0.0, .01, size=(N_Samples))
    data_noise[:, N_STATE_INPUT * T + 1] = np.random.normal(0.0, .01, size=(N_Samples))

    return (data_noise + gen_data)


def shuffle_and_divide(gen_data, Param):
    # first shuffle the data generated
    # This is necesary for the two friction data.
    np.random.shuffle(gen_data)

    # split into features and targets
    features = gen_data[:, 0:-2]
    targets = gen_data[:, -2:]

    # split into train, dev, and test sets

    # calculate train index
    train_ind = int(Param["TRAIN_PERCENT"] * features.shape[0])
    dev_ind = int((Param["TRAIN_PERCENT"] + Param["DEV_PERCENT"]) * features.shape[0])

    # do the splitting
    train = (features[0:train_ind, :], targets[0:train_ind, :])

    dev = (features[train_ind:dev_ind, :], targets[train_ind:dev_ind, :])

    test = (features[dev_ind:, :], targets[dev_ind:, :])

    return train, dev, test


def shuffle_and_divide_modified(gen_data, Param):
    # first shuffle the data generated
    # This is necessary for the two friction data.
    np.random.shuffle(gen_data)

    # split into features and targets
    features = gen_data[:, 0:-3]
    targets = gen_data[:, -3:]

    # split into train, dev, and test sets

    # calculate train index
    train_ind = int(Param["TRAIN_PERCENT"] * features.shape[0])
    dev_ind = int((Param["TRAIN_PERCENT"] + Param["DEV_PERCENT"]) * features.shape[0])

    # do the splitting
    train = (features[0:train_ind, :], targets[0:train_ind, :])

    dev = (features[train_ind:dev_ind, :], targets[train_ind:dev_ind, :])

    test = (features[dev_ind:, :], targets[dev_ind:, :])

    return train, dev, test


def divide(gen_data, Param):
    # first shuffle the data generated
    # This is necesary for the two friction data.

    # split into features and targets
    features = gen_data[:, 0:-3]
    targets = gen_data[:, -3:]

    # split into train, dev, and test sets

    # calculate train index
    train_ind = int(Param["TRAIN_PERCENT"] * features.shape[0])
    dev_ind = int((Param["TRAIN_PERCENT"] + Param["DEV_PERCENT"]) * features.shape[0])

    # do the splitting
    train = (features[0:train_ind, :], targets[0:train_ind, :])

    dev = (features[train_ind:dev_ind, :], targets[train_ind:dev_ind, :])

    test = (features[dev_ind:, :], targets[dev_ind:, :])

    return train, dev, test


# Loads experimental trajectories from recorded data
def load_exp_traj(File_Direct, Param):
    Train_Files = os.listdir(File_Direct)
    Iters = np.arange(len(Train_Files))

    N_STATE_INPUT = Param["N_STATE_INPUT"]
    T = Param["T"]
    DT = Param["DT"]

    for Filename, It in zip(Train_Files, Iters):

        data = np.load(File_Direct + Filename)
        ux = data["ux"]
        uy = data["uy"]
        r = data["r"]
        delta = data["delta"]
        fxf = data["fxf"]

        # Number of Trajectories from a single data file
        N_TRAJ = len(ux) - T

        gen_data = np.zeros(shape=(N_TRAJ, (N_STATE_INPUT * T + 2)))

        for i in range(N_TRAJ):
            # Order of state etc
            for j in range(T):
                gen_data[i, N_STATE_INPUT * j + 0] = r[i + j]
                gen_data[i, N_STATE_INPUT * j + 1] = uy[i + j]
                gen_data[i, N_STATE_INPUT * j + 2] = ux[i + j]
                gen_data[i, N_STATE_INPUT * j + 3] = delta[i + j]
                gen_data[i, N_STATE_INPUT * j + 4] = fxf[i + j] / Param["FX_NORM"]

            gen_data[i, N_STATE_INPUT * T] = r[i + T]
            gen_data[i, N_STATE_INPUT * T + 1] = uy[i + T]

        # this actually how we get data from the multiple files
        if It == 0:
            exp_data = np.copy(gen_data)
        else:
            exp_data = np.concatenate((exp_data, gen_data), axis=0)

    return exp_data
