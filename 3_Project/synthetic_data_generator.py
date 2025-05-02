__author__ = 'Dario Rodriguez'

# import autograd.numpy as np
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.io


# ODE system
def odes(t, y, p):
    # States
    T1, T2, T1s, T2s, V, E = y

    # Parameters
    lambda1, d1, epsilon, k1 = p["lambda1"], p["d1"], p["epsilon"], p["k1"]
    lambda2, d2, f, k2 = p["lambda2"], p["d2"], p["f"], p["k2"]
    delta, m1, m2, NT = p["delta"], p["m1"], p["m2"], p["NT"]
    c, rho1, rho2, lambda_E = p["c"], p["rho1"], p["rho2"], p["lambda_E"]
    bE, Kb, dE, Kd, delta_E = p["bE"], p["Kb"], p["dE"], p["Kd"], p["delta_E"]

    # Right-hand side of the ODE system
    dT1_dt = lambda1 - d1 * T1 - (1 - epsilon) * k1 * V * T1
    dT2_dt = lambda2 - d2 * T2 - (1 - f * epsilon) * k2 * V * T2
    dT1s_dt = (1 - epsilon) * k1 * V * T1 - delta * T1s - m1 * E * T1s
    dT2s_dt = (1 - f * epsilon) * k2 * V * T2 - delta * T2s - m2 * E * T2s
    dV_dt = NT * delta * (T1s + T2s) - c * V - ((1 - epsilon) * rho1 * k1 * T1 + (1 - f * epsilon) * rho2 * k2 * T2) * V
    dE_dt = lambda_E + ((bE * (T1s + T2s)) / (T1s + T2s + Kb)) * E - ((dE * (T1s + T2s)) / (T1s + T2s + Kd)) * E - delta_E * E

    return [dT1_dt, dT2_dt, dT1s_dt, dT2s_dt, dV_dt, dE_dt]


def solve_HIV_ode(params, t_span=(0, 200), t_step=0.5):

    # Fixed Initial conditions
    y0 = [0.9e6, 4000, 0.1, 0.1, 1, 12]

    # Time span
    t_eval = np.arange(t_span[0], t_span[1] + t_step, t_step)

    # Solve the system
    sol = solve_ivp(odes, t_span, y0, args=(params,), t_eval=t_eval,
                    method="LSODA", rtol=1e-12, atol=1e-12, dense_output=True)
    t, y = sol.t, sol.y

    return t, y

def add_noise_HV_sol(y, noise_type='gaussian', scale=0.01):

    std_per_dim = np.std(y, axis=1, keepdims=True)
    if noise_type == 'gaussian':
        noise = np.random.normal(0, std_per_dim * scale, y.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-std_per_dim * scale, std_per_dim * scale, y.shape)
    else:
        raise ValueError('Unknown noise type')
    noisy_sol = y + noise
    return noisy_sol
        


if __name__ == "__main__":

    # ---------------------------------------------------------
    #                     Parameters
    # ----------------------------------------------------------

    # Define the parameter values
    params = {
        "lambda1": 1e4,
        "d1": 0.01,
        "epsilon": 0,
        "k1": 8e-7,
        "lambda2": 31.98,
        "d2": 0.01,
        "f": 0.34,
        "k2": 1e-4, # 1e-4
        "delta": 0.7,  # 0.7,
        "m1": 1e-5,
        "m2": 1e-5,
        "NT": 100,
        "c": 13,
        "rho1": 1,
        "rho2": 1,
        "lambda_E": 1,
        'bE': 0.3,
        "Kb": 100,  # 100,
        "dE": 0.25,
        "Kd": 500,
        "delta_E": 0.1
    }


    # ----------------------------------------------------------
    #               Read data from MATLAB file
    # ----------------------------------------------------------
    mat_data = scipy.io.loadmat('data/hiv_data.mat')
    hiv_data = mat_data['hiv_data']

    # ----------------------------------------------------------
    #               Solve the set of ODEs
    # ----------------------------------------------------------
    t, y = solve_HIV_ode(params, t_step=1)

    # ----------------------------------------------------------
    #               Add noise to the solution
    # ----------------------------------------------------------

    y_noise = add_noise_HV_sol(y, noise_type='gaussian', scale=0.15)
    # Reshape the time vector
    t_new = t[np.newaxis, :]
    # Concatenate time and solution
    hiv_data_syn = np.concatenate((t_new, y_noise), axis=0)

    # ----------------------------------------------------------
    #               Store synthetic data
    # ----------------------------------------------------------
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(os.path.join(data_dir, 'hiv_data.txt'), hiv_data_syn.T, delimiter=',')

    # ----------------------------------------------------------
    #                       Plotting
    # ----------------------------------------------------------
    fontsize = 14
    fontsize_ticks = 13
    labels = ["T1", "T2", "T1*", "T2*", "V", "E"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    steps_plot = 4
    t_plot = t[::steps_plot]
    y_noise_plot = y_noise[:, ::steps_plot]


    ax[0, 0].plot(t, y[0])
    ax[0, 0].plot(hiv_data[:, 0], hiv_data[:, 1], 'o', ms=4)
    ax[0, 0].plot(t_plot, y_noise_plot[0], 'x', ms=4)

    ax[0, 1].plot(t, y[1])
    ax[0, 1].plot(hiv_data[:, 0], hiv_data[:, 2], 'o', ms=4)
    ax[0, 1].plot(t_plot, y_noise_plot[1], 'x', ms=4)

    ax[0, 2].plot(t, y[2])
    ax[0, 2].plot(hiv_data[:, 0], hiv_data[:, 3], 'o', ms=4)
    ax[0, 2].plot(t_plot, y_noise_plot[2], 'x', ms=4)

    ax[1, 0].plot(t, y[3])
    ax[1, 0].plot(hiv_data[:, 0], hiv_data[:, 4], 'o', ms=4)
    ax[1, 0].plot(t_plot, y_noise_plot[3], 'x', ms=4)

    ax[1, 1].plot(t, y[4])
    ax[1, 1].plot(hiv_data[:, 0], hiv_data[:, 5], 'o', ms=4)
    ax[1, 1].plot(t_plot, y_noise_plot[4], 'x', ms=4)

    ax[1, 2].plot(t, y[5], label='Predicted')
    ax[1, 2].plot(hiv_data[:, 0], hiv_data[:, 6], 'o', ms=4, label='Reference')
    ax[1, 2].plot(t_plot, y_noise_plot[5], 'x', ms=4,label='Noisy')

    ax[1, 0].set_xlabel("Time", fontsize=fontsize)
    ax[1, 1].set_xlabel("Time", fontsize=fontsize)
    ax[1, 2].set_xlabel("Time", fontsize=fontsize)
    ax[0, 0].set_ylabel(r"$T_1$", fontsize=fontsize)
    ax[0, 1].set_ylabel(r"$T_2$", fontsize=fontsize)
    ax[0, 2].set_ylabel(r"$T_1^*$", fontsize=fontsize)
    ax[1, 0].set_ylabel(r"$T_2^*$", fontsize=fontsize)
    ax[1, 1].set_ylabel("V", fontsize=fontsize)
    ax[1, 2].set_ylabel("E", fontsize=fontsize)

    ax[0, 0].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[0, 1].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[0, 2].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[1, 0].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[1, 1].tick_params(axis='both', labelsize=fontsize_ticks)
    ax[1, 2].tick_params(axis='both', labelsize=fontsize_ticks)

    ax[1, 2].legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    plt.show()
