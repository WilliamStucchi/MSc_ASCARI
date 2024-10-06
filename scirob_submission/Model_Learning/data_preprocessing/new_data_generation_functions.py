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


def sample(Param, Veh, mu, last_delta, last_fx):
    Ux_lim = Param["UX_LIM"]
    a = Veh["a"]
    b = Veh["b"]
    m = Veh["m"]
    Cr = Veh["Cr"] * Veh['rear_normal_load']
    del_lim = Veh["del_lim"]

    max_delta_diff = Veh["max_delta_diff"]
    max_fx_diff = Veh["max_fx_diff"]
    p_lim = Veh["p_lim"]
    b_bias = Veh["b_bias"]

    if last_delta is None or last_fx is None:
        delta = np.random.uniform(low=-del_lim, high=del_lim)

        # Could use Power and friction limits
        # Fx = np.random.uniform(low = -(mu*m*9.81) , high = p_lim/ux) #8640
        # empirical force limits
        Fx = np.random.uniform(low=-4700, high=3700)

    else:
        if last_delta - max_delta_diff >= -del_lim:
            lower_lim_delta = last_delta - max_delta_diff
        else:
            lower_lim_delta = -del_lim
        if last_fx + max_delta_diff <= del_lim:
            upper_lim_delta = last_fx + max_delta_diff
        else:
            upper_lim_delta = del_lim
        delta = np.random.uniform(low=lower_lim_delta, high=upper_lim_delta)

        if last_fx - max_fx_diff >= -4700:
            lower_lim_fx = last_fx - max_fx_diff
        else:
            lower_lim_fx = -4700
        if last_fx + max_delta_diff <= 3700:
            upper_lim_fx = last_fx + max_fx_diff
        else:
            upper_lim_fx = 3700
        Fx = np.random.uniform(low=lower_lim_fx, high=upper_lim_fx)

    # First sample Ux to ensure dynamically feasible uy and r! This was a mistake before.
    Ux = np.random.uniform(low=1.0, high=Ux_lim)

    r_lim = (mu * 9.81) / Ux
    Uy_lim = (3 * Ux * m * 9.81 * mu) / Cr * (a / (a + b)) + b * r_lim

    r = np.random.uniform(low=-r_lim, high=r_lim)
    Uy = np.random.uniform(low=-Uy_lim, high=Uy_lim)



    return r, Uy, Ux, delta, Fx


def step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx, Param, Veh, mu, beta_t, a_f_in, a_r_in, af, ar, bf, br, ef, er, lf, lr, tf, tr, DT):
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
    # Unpack dicts
    T = Param["T"]
    N_SAMPLES = Param["N_SAMPLES"]
    N_STATE_INPUT = Param["N_STATE_INPUT"]
    TWO_FRIC = Param["TWO_FRIC"]
    Fx_norm = Param["FX_NORM"]

    mu = Veh["mu"]
    mu_2 = Veh["mu_2"]
    a = Veh["a"]
    b = Veh["b"]

    # gen_data = np.zeros( (N_SAMPLES, (N_STATE_INPUT*T+2) ) ) #2 targets r and Uy
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
    muf = Veh['muyf']
    mur = Veh['muyr']
    peakf = Veh['Peakyf']
    peakr = Veh['Peakyr']

    bf = 1 + (1 - 2 / np.pi * np.arcsin(pf))
    br = 1 + (1 - 2 / np.pi * np.arcsin(pr))
    af = cf / bf / muf
    ar = cr / br / mur
    ef = (af * peakf - np.tan(np.pi / 2 / bf)) / (af * peakf - np.arctan(af * peakf))
    er = (ar * peakr - np.tan(np.pi / 2 / br)) / (ar * peakr - np.arctan(ar * peakr))
    lf = hf
    lr = hr

    print("Starting Data Generation: ")
    # print("else")
    for i in tqdm(range(N_SAMPLES)):
        last_delta = None
        last_fx = None

        # run a single simulation for one trajectory of length T
        for j in range(T):
            # FIll out the delay states!
            if j == 0:
                r_t, Uy_t, Ux_t, del_t, Fx_t = sample(Param, Veh, mu, last_delta, last_fx)
                gen_data[i, 0] = r_t
                gen_data[i, 1] = Uy_t
                gen_data[i, 2] = Ux_t
                gen_data[i, 3] = del_t * SW_rate
                gen_data[i, 4] = Fx_t

                # Calculate the initial slip from SS assumptions
                a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                a_r = np.arctan((Uy_t - b * r_t) / Ux_t)
                beta_t = 0  # initialization
            else:
                # Propagate the next state w/ dynamics model!
                r_t, Uy_t, Ux_t, a_f, a_r, _, _, _, beta_t = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t, Param, Veh, mu,
                                                                           beta_t, a_f, a_r, af, ar, bf, br, ef, er, lf,
                                                                           lr, tf, tr, Param["DT"])

                # And cache the next state in our data matrix
                gen_data[i, (N_STATE_INPUT * j)] = r_t
                gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                # Sample the next set of controls!
                last_delta = gen_data[i, (N_STATE_INPUT * (j - 1)) + 3] / SW_rate
                last_fx = gen_data[i, (N_STATE_INPUT * (j - 1)) + 4]
                _, _, _, del_t, Fx_t = sample(Param, Veh, mu, last_delta, last_fx)

                # Then Cache them in the data!
                gen_data[i, (N_STATE_INPUT * j) + 3] = del_t * SW_rate
                gen_data[i, (N_STATE_INPUT * j) + 4] = Fx_t

        # For the Last T step we will get the vel targets we want to predict!
        _, _, _, _, _, r_dot_t, Uy_dot_t, Ux_dot_t, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fx_t,
                                                                      Param, Veh, mu, beta_t, a_f, a_r,
                                                                      af, ar, bf, br, ef, er, lf, lr,
                                                                      tf, tr, Param["DT"])

        # Cache the targets!
        gen_data[i, (N_STATE_INPUT * T)] = r_dot_t
        gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_dot_t
        gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_dot_t

    return gen_data


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
