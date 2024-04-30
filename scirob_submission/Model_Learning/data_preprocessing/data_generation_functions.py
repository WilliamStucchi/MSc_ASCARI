# Nathan Spielberg
# DDL 10.17.2018

# Data Generation Functions for use in data generation
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter, freqz
import os


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


def sample(Param, Veh, mu):
    Ux_lim = Param["UX_LIM"]
    a = Veh["a"]
    b = Veh["b"]
    m = Veh["m"]
    Cr = Veh["Cr"]
    del_lim = Veh["del_lim"]

    p_lim = Veh["p_lim"]
    b_bias = Veh["b_bias"]

    # Could use Power and friction limits
    # Fx = np.random.uniform(low = -(mu*m*9.81) , high = p_lim/ux) #8640
    # empirical force limits
    Fx = np.random.uniform(low=-(19000), high=15000)

    if Fx > 0:
        Fxf = Fx
        Fxr = 0
    else:
        Fxf = b_bias * Fx
        Fxr = (1 - b_bias) * Fx

    # First sample Ux to ensure dynamically feasible uy and r! This was a mistake before.
    Ux = np.random.uniform(low=1.0, high=Ux_lim)

    r_lim = (mu * 9.81) / Ux
    Uy_lim = (3 * Ux * m * 9.81 * mu) / Cr * (a / (a + b)) + b * r_lim

    r = np.random.uniform(low=-r_lim, high=r_lim)
    Uy = np.random.uniform(low=-Uy_lim, high=Uy_lim)
    delta = np.random.uniform(low=-del_lim, high=del_lim)

    return r, Uy, Ux, delta, Fxf, Fxr


def step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf, Fxr, Param, Veh, mu, a_f_in, a_r_in, DT):
    a = Veh["a"]
    b = Veh["b"]
    l = Veh["l"]
    h = Veh["h"]
    Cr = Veh["Cr"]
    Cf = Veh["Cf"]
    m = Veh["m"]
    Izz = Veh["Izz"]
    sig_f = Veh["sig_f"]
    sig_r = Veh["sig_r"]

    # Calculate slip angles:
    if Param["RELAX_LENGTH"]:
        a_f = a_f_in
        a_r = a_r_in

        # Then run updates to get next slips
        V = np.sqrt(Ux_t ** 2 + Uy_t ** 2)
        a_f_t1 = a_f + DT * ((V / sig_f) * (np.arctan((Uy_t + a * r_t) / Ux_t) - del_t - a_f))
        a_r_t1 = a_r + DT * ((V / sig_r) * (np.arctan((Uy_t - b * r_t) / Ux_t) - a_r))
    else:
        a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

        # We dont use next slips in this case.
        a_f_t1 = 0
        a_r_t1 = 0

    if Param["WEIGHT_TRANSFER"]:
        # Need another component in here for Fy sin del!

        # First with WT Calc ax:
        ax = (Fxf * np.cos(del_t) + Fxr) / m

        # Now Calc Normal loads which are now dep on ax
        Fzf = (b / l) * m * 9.81 - (h / l) * m * ax
        Fzr = (a / l) * m * 9.81 + (h / l) * m * ax
    else:
        Fzf = (m * 9.81 * b) / (l)
        Fzr = (m * 9.81 * a) / (l)

    # Calculate Forces
    Fyf = fiala(a_f, Cf, mu, Fzf)
    Fyr = fiala(a_r, Cr, mu, Fzr)

    # Calculate Derivatives
    Uy_dot = (Fyf * np.cos(del_t) + Fxf * np.sin(del_t) + Fyr) / m - r_t * Ux_t
    r_dot = (a * Fyf * np.cos(del_t) + a * Fxf * np.sin(del_t) - b * Fyr) / Izz
    Ux_dot = (Fxf * np.cos(del_t) + Fxr) / m + r_t * Uy_t

    Uy_t_1 = Uy_t + DT * Uy_dot
    r_t_1 = r_t + DT * r_dot
    Ux_t_1 = Ux_t + DT * Ux_dot

    return r_t_1, Uy_t_1, Ux_t_1, a_f_t1, a_r_t1


def gen_data(Param, Veh):
    # Unpack dicts
    N_SAMPLES = Param["N_SAMPLES"]
    Ux_lim = Param["UX_LIM"]
    T = Param["T"]
    DT = Param["DT"]
    N_SAMPLES = Param["N_SAMPLES"]
    N_STATE_INPUT = Param["N_STATE_INPUT"]
    TWO_FRIC = Param["TWO_FRIC"]
    Fx_norm = Param["FX_NORM"]
    RELAX_LENGTH = Param["RELAX_LENGTH"]

    m = Veh["m"]
    mu = Veh["mu"]
    mu_2 = Veh["mu_2"]
    a = Veh["a"]
    b = Veh["b"]
    Cr = Veh["Cr"]
    del_lim = Veh["del_lim"]
    Izz = Veh["Izz"]
    p_lim = Veh["p_lim"]
    b_bias = Veh["b_bias"]

    gen_data = np.zeros((N_SAMPLES, (N_STATE_INPUT * T + 2)))  # 2 targets r and Uy

    print("Starting Data Generation: ")
    if TWO_FRIC:
        for i in tqdm(range(N_SAMPLES)):
            if i < N_SAMPLES / 2.0:
                # run a single simulation for one trajectory of length T
                for j in range(T):
                    # FIll out the delay states!
                    if j == 0:
                        r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)
                        gen_data[i, 0] = r_t
                        gen_data[i, 1] = Uy_t
                        gen_data[i, 2] = Ux_t
                        gen_data[i, 3] = del_t
                        gen_data[i, 4] = Fxf_t / Fx_norm

                        # Calculate the initial slip from SS assumptions
                        a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                    else:
                        # Propogate the next state w/ dynamics model!
                        r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu,
                                                                  a_f, a_r, Param["DT"])

                        # And cache the next state in our data matrix
                        gen_data[i, (N_STATE_INPUT * j)] = r_t
                        gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                        gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                        # Sample the next set of controls!
                        _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)

                        # Then Cache them in the data!
                        gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                        gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm

                # For the Last T step we will get the vel targets we want to predict!
                r_t, Uy_t, _, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f, a_r,
                                                   Param["DT"])

                # And cache the targets!
                gen_data[i, (N_STATE_INPUT * T)] = r_t
                gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t

            else:
                for j in range(T):
                    # FIll out the delay states!
                    if j == 0:
                        r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu_2)
                        gen_data[i, 0] = r_t
                        gen_data[i, 1] = Uy_t
                        gen_data[i, 2] = Ux_t
                        gen_data[i, 3] = del_t
                        gen_data[i, 4] = Fxf_t / Fx_norm

                        # Calculate the initial slip from SS assumptions
                        a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                    else:
                        # Propogate the next state w/ dynamics model!
                        r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh,
                                                                  mu_2, a_f, a_r, Param["DT"])

                        # And cache the next state in our data matrix
                        gen_data[i, (N_STATE_INPUT * j)] = r_t
                        gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                        gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                        # Sample the next set of controls!
                        _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu_2)

                        # Then Cache them in the data!
                        gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                        gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm

                # For the Last T step we will get the vel targets we want to predict!
                r_t, Uy_t, _, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu_2, a_f, a_r,
                                                   Param["DT"])

                # And cache the targets!
                gen_data[i, (N_STATE_INPUT * T)] = r_t
                gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t
    else:

        for i in tqdm(range(N_SAMPLES)):

            # run a single simulation for one trajectory of length T
            for j in range(T):
                # FIll out the delay states!
                if j == 0:
                    r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)
                    gen_data[i, 0] = r_t
                    gen_data[i, 1] = Uy_t
                    gen_data[i, 2] = Ux_t
                    gen_data[i, 3] = del_t
                    gen_data[i, 4] = Fxf_t / Fx_norm

                    # Calculate the initial slip from SS assumptions
                    a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                    a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                else:
                    # Propogate the next state w/ dynamics model!
                    r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f,
                                                              a_r, Param["DT"])

                    # And cache the next state in our data matrix
                    gen_data[i, (N_STATE_INPUT * j)] = r_t
                    gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                    gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                    # Sample the next set of controls!
                    _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)

                    # Then Cache them in the data!
                    gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                    gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm

            # For the Last T step we will get the vel targets we want to predict!
            r_t, Uy_t, _, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f, a_r,
                                               Param["DT"])

            # Learn the State Derivs!

            # And cache the targets!
            gen_data[i, (N_STATE_INPUT * T)] = r_t
            gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t

    return gen_data


def gen_data_mod(Param, Veh):
    # Unpack dicts
    N_SAMPLES = Param["N_SAMPLES"]
    Ux_lim = Param["UX_LIM"]
    T = Param["T"]
    DT = Param["DT"]
    N_SAMPLES = Param["N_SAMPLES"]
    N_STATE_INPUT = Param["N_STATE_INPUT"]
    TWO_FRIC = Param["TWO_FRIC"]
    Fx_norm = Param["FX_NORM"]
    RELAX_LENGTH = Param["RELAX_LENGTH"]

    m = Veh["m"]
    mu = Veh["mu"]
    mu_2 = Veh["mu_2"]
    a = Veh["a"]
    b = Veh["b"]
    Cr = Veh["Cr"]
    del_lim = Veh["del_lim"]
    Izz = Veh["Izz"]
    p_lim = Veh["p_lim"]
    b_bias = Veh["b_bias"]

    # gen_data = np.zeros( (N_SAMPLES, (N_STATE_INPUT*T+2) ) ) #2 targets r and Uy
    gen_data = np.zeros((N_SAMPLES, (N_STATE_INPUT * T + 3)))  # 2 targets r, Uy and Ux

    print("Starting Data Generation: ")
    if TWO_FRIC:
        # print("if")
        for i in tqdm(range(N_SAMPLES)):
            if i < N_SAMPLES / 2.0:
                # run a single simulation for one trajectory of length T
                for j in range(T):
                    # FIll out the delay states!
                    if j == 0:
                        r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)
                        gen_data[i, 0] = r_t
                        gen_data[i, 1] = Uy_t
                        gen_data[i, 2] = Ux_t
                        gen_data[i, 3] = del_t
                        gen_data[i, 4] = Fxf_t / Fx_norm
                        print("Fx: ", gen_data[i, 4])

                        # Calculate the initial slip from SS assumptions
                        a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                    else:
                        # Propagate the next state w/ dynamics model!
                        r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu,
                                                                  a_f, a_r, Param["DT"])

                        # And cache the next state in our data matrix
                        gen_data[i, (N_STATE_INPUT * j)] = r_t
                        gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                        gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                        # Sample the next set of controls!
                        _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)

                        # Then Cache them in the data!
                        gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                        gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm
                        print("Fx: ", (N_STATE_INPUT * j) + 4)

                # For the Last T step we will get the vel targets we want to predict!
                r_t, Uy_t, Ux_t, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f, a_r,
                                                   Param["DT"])

                # And cache the targets!
                gen_data[i, (N_STATE_INPUT * T)] = r_t
                gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t
                gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_t

            else:
                for j in range(T):
                    # FIll out the delay states!
                    if j == 0:
                        r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu_2)
                        gen_data[i, 0] = r_t
                        gen_data[i, 1] = Uy_t
                        gen_data[i, 2] = Ux_t
                        gen_data[i, 3] = del_t
                        gen_data[i, 4] = Fxf_t / Fx_norm
                        print(gen_data[i, 4])

                        # Calculate the initial slip from SS assumptions
                        a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                        a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                    else:
                        # Propagate the next state w/ dynamics model!
                        r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh,
                                                                  mu_2, a_f, a_r, Param["DT"])

                        # And cache the next state in our data matrix
                        gen_data[i, (N_STATE_INPUT * j)] = r_t
                        gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                        gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                        # Sample the next set of controls!
                        _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu_2)

                        # Then Cache them in the data!
                        gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                        gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm
                        print(gen_data[i, (N_STATE_INPUT * j) + 4])

                # For the Last T step we will get the vel targets we want to predict!
                r_t, Uy_t, Ux_t, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu_2, a_f, a_r,
                                                   Param["DT"])

                # And cache the targets!
                gen_data[i, (N_STATE_INPUT * T)] = r_t
                gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t
                gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_t
    else:

        # print("else")
        for i in tqdm(range(N_SAMPLES)):

            # run a single simulation for one trajectory of length T
            for j in range(T):
                # FIll out the delay states!
                if j == 0:
                    r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)
                    gen_data[i, 0] = r_t
                    gen_data[i, 1] = Uy_t
                    gen_data[i, 2] = Ux_t
                    gen_data[i, 3] = del_t
                    gen_data[i, 4] = Fxf_t / Fx_norm
                    print(gen_data[i, 4])

                    # Calculate the initial slip from SS assumptions
                    a_f = np.arctan((Uy_t + a * r_t) / Ux_t) - del_t
                    a_r = np.arctan((Uy_t - b * r_t) / Ux_t)

                else:
                    # Propogate the next state w/ dynamics model!
                    r_t, Uy_t, Ux_t, a_f, a_r = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f,
                                                              a_r, Param["DT"])

                    # And cache the next state in our data matrix
                    gen_data[i, (N_STATE_INPUT * j)] = r_t
                    gen_data[i, (N_STATE_INPUT * j) + 1] = Uy_t
                    gen_data[i, (N_STATE_INPUT * j) + 2] = Ux_t

                    # Sample the next set of controls!
                    _, _, _, del_t, Fxf_t, Fxr_t = sample(Param, Veh, mu)

                    # Then Cache them in the data!
                    gen_data[i, (N_STATE_INPUT * j) + 3] = del_t
                    gen_data[i, (N_STATE_INPUT * j) + 4] = Fxf_t / Fx_norm
                    print(gen_data[i, (N_STATE_INPUT * j) + 4])

            # For the Last T step we will get the vel targets we want to predict!
            # r_t, Uy_t, _ , _, _                                                                                        = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f, a_r, Param["DT"])
            r_t, Uy_t, Ux_t, _, _ = step_dynamics(r_t, Uy_t, Ux_t, del_t, Fxf_t, Fxr_t, Param, Veh, mu, a_f, a_r,
                                                  Param["DT"])

            # Learn the State Derivs!

            # And cache the targets!
            gen_data[i, (N_STATE_INPUT * T)] = r_t
            gen_data[i, (N_STATE_INPUT * T) + 1] = Uy_t
            gen_data[i, (N_STATE_INPUT * T) + 2] = Ux_t

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
