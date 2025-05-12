__author__ = 'Dario Rodriguez'

import numpy as np
import os
import pickle
from pymcmcstat.MCMC import MCMC
from pymcmcstat import mcmcplot as mcp
from pymcmcstat import propagation as up
import matplotlib.pyplot as plt
from synthetic_data_generator import solve_HIV_ode

# --------------------------------------------------------
#                   User flags
# --------------------------------------------------------
flag_bayesian_inference = True
flag_postprocessing = True


# --------------------------------------------------------
#                   Load data
# --------------------------------------------------------
hiv_data = np.loadtxt('data/hiv_data.txt', delimiter=',')  # shape (N, 7) [t, T1, T2, T1s, T2s, V, E]
t_data = hiv_data[:, 0]
# Y_data = hiv_data[:, 1:]
Y_data = hiv_data[:, 6]

# --------------------------------------------------------
#                   Initial Conditions
# --------------------------------------------------------

Y0 = [0.9e6, 4000, 0.1, 0.1, 1.0, 12.0]

# -----------------------------------------
#        Nominal Parameters
# -----------------------------------------
nom_params = {
    "lambda1": 1e4,
    "d1": 0.01,
    "epsilon": 0,
    "k1": 8e-7,
    "lambda2": 31.98,
    "d2": 0.01,
    "f": 0.34,
    "k2": 1e-4,  # 1.23e-4,
    "delta": 0.6,  # 0.6,
    "m1": 1e-5,
    "m2": 1e-5,
    "NT": 100,
    "c": 13,
    "rho1": 1,
    "rho2": 1,
    "lambda_E": 1,
    'bE': 0.3,
    "Kb": 100,  # 88,
    "dE": 0.25,
    "Kd": 500,
    "delta_E": 0.1
}

# Identifiable parameters
param_keys = ["lambda1", "lambda2", "NT", "c", "bE", "Kb", "Kd", "delta_E", "delta", "lambda_E", "dE"]

def solve_reduced_HIV_ode(id_theta):
    params = nom_params.copy()
    for key, val in zip(param_keys, id_theta):
        params[key] = val

    # Solve the ODE system
    _, y_vec = solve_HIV_ode(params, t_step=1)
    y = y_vec.T[:, 5]

    return y  # shape (n_times, 6)


def ssq(theta, data):
    model_out = solve_reduced_HIV_ode(theta)
    residuals = data.ydata - model_out
    return np.sum(residuals ** 2)

# MCMC Setup
mcstat = MCMC()
mcstat.data.add_data_set(x=t_data.reshape(-1, 1), y=Y_data)


# --------------------------------------------------------------------
#                      MCMC with Original Model
# --------------------------------------------------------------------

if flag_bayesian_inference:

    # Add model parameters [name, initial, min, max]
    param_defs = [
        ('lambda1', 1e4, 1e4*0.9, 1e4*1.1),
        ('lambda2', 31.98, 31.98*0.9, 31.98*1.1),
        ('NT', 100, 100*0.9, 100*1.1),
        ('c', 13, 13*0.9, 13*1.1),
        ('bE', 0.3, 0.3*0.9, 0.3*1.1),
        ('Kb', 100, 100*0.9, 100*1.1),
        ('Kd', 500, 500*0.9, 500*1.1),
        ('delta_E', 0.1, 0.1*0.9, 0.1*1.1),
        ('delta', 0.7, 0.7*0.9, 0.7*1.1),
        ('lambda_E', 1, 1*0.9, 1*1.1),
        ('dE', 0.25, 0.25*0.9, 0.25*1.1)
        ]

    for name, val, low, high in param_defs:
        mcstat.parameters.add_model_parameter(name=name, theta0=val, minimum=low, maximum=high)

    mcstat.model_settings.define_model_settings(
        sos_function=ssq)

    mcstat.simulation_options.define_simulation_options(
        nsimu=100000, #20000
        updatesigma=True,
        method='dram'
    )
    mcstat.run_simulation()

    results = mcstat.simulation_results.results
    mcstat.chainstats(results['chain'], results)

    # Store results
    data_dir = 'bayesian_analysis'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(data_dir + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if flag_postprocessing:

    with open('bayesian_analysis/results.pkl', 'rb') as f:
        results = pickle.load(f)

    def predmodel(q, data):
        ymodel = solve_reduced_HIV_ode(q)
        return ymodel


    #  Post-processing
    burnin = 5000 # int(results['nsimu'] / 2)
    chain = results['chain'][burnin:, :]
    s2chain = results['s2chain'][burnin:, :]
    names = results['names']

    # plot chain panel
    settings = {
        'fig': dict(figsize=(15, 10)),
        'plot': dict(color='b', linestyle='-')
    }
    mcp.plot_chain_panel(chain, names, settings)

    # plot density panel
    mcp.plot_density_panel(chain, names, settings)


    # pairwise correlation
    settings_pair = dict(
        fig=dict(figsize=(12, 10))
    )
    f = mcp.plot_pairwise_correlation_panel(chain, names, settings_pair)

    plt.show()


    # Interval calculation and plotting
    pdata = mcstat.data
    intervals = up.calculate_intervals(chain, results, pdata, predmodel,
                                      waitbar=True, s2chain=s2chain)


    data_display = dict(
        marker='o',
        color='k',
        mfc='none',
        label='Data')
    model_display = dict(
        color='r')
    interval_display = dict(
        alpha=0.5)
    # for ii, interval in enumerate(intervals):
    #     fig, ax = up.plot_intervals(interval,
    #                                 time=mcstat.data.xdata[0],
    #                                 ydata=mcstat.data.ydata[0][:, ii],
    #                                 # data_display=data_display,
    #                                 model_display=model_display,
    #                                 interval_display=interval_display,
    #                                 ciset=dict(colors=['#c7e9b4']),
    #                                 piset=dict(colors=['#225ea8']),
    #                                 figsize=(12, 4)

    fig, ax = up.plot_intervals(intervals,
                                time=mcstat.data.xdata[0],
                                ydata=mcstat.data.ydata[0],
                                # data_display=data_display,
                                model_display=model_display,
                                interval_display=interval_display,
                                ciset=dict(colors=['#c7e9b4']),
                                piset=dict(colors=['#225ea8']),
                                figsize=(12, 4))

    ax.set_ylabel('')
    # ax.set_title(ylbls[ii + 1][0])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_xlabel('Time (Days)')

    plt.tight_layout()
    plt.show()





# chains = results.results['chain']
# mcmcplot.plot_chain_panel(chains)
# mcmcplot.plot_histograms(results)
#
# # === Posterior Predictions ===
# from pymcmcstat.functions.mcmcpred import mcmcpred
#
# numpred = 100
# out = mcmcpred(results, data={'xdata': t_data.reshape(-1, 1)}, model=hiv_model, numpred=numpred)
# pred = out['predicted_y']
# mean_pred = np.mean(pred, axis=0)
# std_pred = np.std(pred, axis=0)
#
# # === Plot Predictions ===
# fig, axs = plt.subplots(3, 2, figsize=(12, 10))
# labels = ['T1', 'T2', 'T1s', 'T2s', 'V', 'E']
# for i, ax in enumerate(axs.flat):
#     ax.plot(t_data, Y_data[:, i], 'ko', label='Data')
#     ax.plot(t_data, mean_pred[:, i], 'b-', label='Mean Prediction')
#     ax.fill_between(t_data,
#                     mean_pred[:, i] - 2*std_pred[:, i],
#                     mean_pred[:, i] + 2*std_pred[:, i],
#                     color='blue', alpha=0.2, label='95% CI')
#     ax.set_title(labels[i])
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.legend()
# plt.tight_layout()
# plt.show()
