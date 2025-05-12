import numpy as np
import chaospy as cp
import os
import pickle
from pymcmcstat.MCMC import MCMC
from pymcmcstat import mcmcplot as mcp
from pymcmcstat import propagation as up
import matplotlib.pyplot as plt
from synthetic_data_generator import solve_HIV_ode
from SALib.analyze import sobol
from SALib.sample import saltelli

# --------------------------------------------------------
#                   User flags
# --------------------------------------------------------
flag_samples = False
flag_fit_PCE = False
flag_plot_PCE = False
flag_bayesian_inference = True
flag_global_sensitivity = True



current_file_dir = os.path.dirname(os.path.abspath(__file__))

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
    "delta": 0.7,  # 0.6,
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

# ------------------------------------------------
#            Identifiable parameters
# ------------------------------------------------
param_keys = ["lambda1", "lambda2", "NT", "c", "bE", "Kb", "Kd", "delta_E", "delta", "lambda_E", "dE"]
# param_keys = ['bE', 'delta', 'd1', 'k2', 'lambda1', 'Kb']


def solve_reduced_HIV_ode(id_theta_dict):
    params = nom_params.copy()
    for key, val in id_theta_dict.items():
        params[key] = float(val)

    # Solve the ODE system
    _, y = solve_HIV_ode(params, t_step=1)

    return y.T  # shape

def surrogate_pce(theta):
    theta = np.atleast_2d(theta).T  # shape (6, 1)
    return cp.call(pce_model, theta).flatten()  # shape (201,)

def evaluate_model(params):
    param_sample_dict = {k: p for k, p in zip(param_keys, params) if k in param_keys}
    return solve_reduced_HIV_ode(param_sample_dict)[:, 5] # shape (n_times, )


# dist = cp.J(
#     cp.Uniform(0.298, 0.302),   # bE
#     cp.Uniform(0.67, 0.69),     # delta
#     cp.Uniform(7e-3, 10.5e-3),  # d1
#     cp.Uniform(0.9e-4, 1.3e-3), # k2
#     cp.Uniform(0.98e4, 1.02e4), # lambda1
#     cp.Uniform(70, 110),        # Kb
#
# )

pert = 0.05
dist = cp.J(
    cp.Uniform(1e4 * (1 - pert), 1e4 * (1 + pert)),   # lambda1
    cp.Uniform(31.98 * (1 - pert), 31.98 * (1 + pert)),     # lambda2
    cp.Uniform(100 * (1 - pert), 100 * (1 + pert)),         # NT
    cp.Uniform(13 * (1 - pert), 13 * (1 + pert)),           # c
    cp.Uniform(0.3 * (1 - pert), 0.3 * (1 + pert)),         # bE
    cp.Uniform(100 * (1 - pert), 100 * (1 + pert)),         # Kb
    cp.Uniform(500 * (1 - pert), 500 * (1 + pert)),         # Kd
    cp.Uniform(0.1 * (1 - pert), 0.1 * (1 + pert)),         # delta_E
    cp.Uniform(0.7 * (1 - pert), 0.7 * (1 + pert)),         # delta
    cp.Uniform(1 * (1 - pert), 1 * (1 + pert)),             # lambda_E
    cp.Uniform(0.25 * (1 - pert), 0.25 * (1 + pert))        # dE
)


n_samples = 300
samples = dist.sample(n_samples).T # shape (n_samples, n_params)
nodes = samples.T


# --------------------------------------------------------------------
#                      MCMC with Surrogate Model
# --------------------------------------------------------------------
if flag_samples:

    # Evaluate the full model
    model_outputs = np.array([evaluate_model(p) for p in samples])  # shape (n_samples, n_times)

    # Save model outputs
    with open('PCE/model_outputs.pkl', 'wb') as f:
        pickle.dump(model_outputs, f)

if flag_fit_PCE:

    # load model outputs
    with open(current_file_dir + '/PCE/model_outputs.pkl', 'rb') as f:
        model_outputs = pickle.load(f)

    poly_order = 3
    polynomial_expansion = cp.generate_expansion(poly_order, dist)
    pce_model = cp.fit_regression(polynomial_expansion, samples.T, model_outputs)

    # Store pce model
    with open('PCE/pce_model.pkl', 'wb') as f:
        pickle.dump(pce_model, f)
    with open('PCE/poly_expansion.pkl', 'wb') as f:
        pickle.dump(polynomial_expansion, f)

if flag_plot_PCE:

    # Load the PCE model
    with open('PCE/pce_model.pkl', 'rb') as f:
        pce_model = pickle.load(f)

    # Comparison of the predictions
    # param_sample = samples[0, :]
    theta_nom = [nom_params[k] for k in param_keys]
    main_model_output = evaluate_model(theta_nom)
    surrogate_model_output = surrogate_pce(theta_nom)

    # Plotting
    fontsize = 14
    tick_fontsize = 13
    fig, ax = plt.subplots()
    ax.plot(t_data, main_model_output, 'x', label='Full model')
    ax.plot(t_data, surrogate_model_output, label='Surrogate model')
    ax.legend(fontsize=fontsize)
    ax.set_xlabel('Time (days)', fontsize=fontsize)
    ax.set_ylabel('Immune effector cell count', fontsize=fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    plt.show()

if flag_bayesian_inference:

    # Load the PCE model
    with open('PCE/pce_model.pkl', 'rb') as f:
        pce_model = pickle.load(f)

    def surrogate_ssq(pars, data):
        pred = surrogate_pce(pars)  # shape (n_times, 6)
        residuals = data.ydata - pred
        return np.sum(residuals**2)

    # param_defs = [
    #     ('bE', 0.3, 0.25, 0.35),
    #     ('delta', 0.7, 0.01, 1.0),
    #     ('d1', 0.01, 0.001, 0.1),
    #     ('k2', 1e-4, 1e-5, 1e-3),
    #     ('lambda1', 1e4, 5000, 20000),
    #     ('Kb', 100, 60, 120)
    # ]

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


    mcstat = MCMC()
    mcstat.data.add_data_set(x=t_data.reshape(-1, 1), y=Y_data)

    # Add same priors
    for name, val, low, high in param_defs:
        mcstat.parameters.add_model_parameter(name=name, theta0=val, minimum=low, maximum=high)

    # Use surrogate likelihood
    mcstat.model_settings.define_model_settings(sos_function=surrogate_ssq)
    mcstat.simulation_options.define_simulation_options(nsimu=20000, updatesigma=True, method='dram')

    mcstat.run_simulation()

    # Postprocessing
    results = mcstat.simulation_results.results
    burnin = 5000 # int(results['nsimu'] / 2)
    chain = results['chain'][burnin:, :]
    s2chain = results['s2chain'][burnin:, :]
    names = results['names']

    # Store chain
    with open('PCE/pce_chain.pkl', 'wb') as f:
        pickle.dump(chain, f)

    # display chain stats
    mcstat.chainstats(chain, results)

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

    # # Evaluate surrogate at each posterior sample
    n_samples = chain.shape[0]
    predictions = []

    for i in range(n_samples):
        pred = surrogate_pce(chain[i, :])   # output shape: (n_times,)
        predictions.append(pred)

    predictions = np.array(predictions)  # shape (n_samples, n_times)

    lower = np.percentile(predictions, 2.5, axis=0)
    median = np.percentile(predictions, 50, axis=0)
    upper = np.percentile(predictions, 97.5, axis=0)

    plt.fill_between(t_data, lower, upper, color='gray', alpha=0.5, label='95% Credible Interval')
    plt.plot(t_data, median, 'r-', label='Median Prediction')
    plt.scatter(t_data, Y_data, color='black', s=10, label='Noisy Data')

    plt.xlabel('Time (days)', fontsize=14)
    plt.ylabel('Effector Cell Count', fontsize=14)
    plt.legend(fontsize=13)
    plt.title('Posterior Predictive with 95% Credible Interval', fontsize=14)
    plt.show()


    # def predmodel(q, data):
    #     ymodel = surrogate_pce(q)
    #     return ymodel
    #
    # # Interval calculation and plotting
    # pdata = mcstat.data
    # intervals = up.calculate_intervals(chain, results, pdata, predmodel,
    #                                    waitbar=True, s2chain=s2chain)
    #
    # data_display = dict(
    #     marker='o',
    #     color='k',
    #     mfc='none',
    #     label='Data')
    # model_display = dict(
    #     color='r')
    # interval_display = dict(
    #     alpha=0.5)
    #
    # fig, ax = up.plot_intervals(intervals,
    #                             time=mcstat.data.xdata[0],
    #                             ydata=mcstat.data.ydata[0],
    #                             # data_display=data_display,
    #                             model_display=model_display,
    #                             interval_display=interval_display,
    #                             ciset=dict(colors=['#c7e9b4']),
    #                             piset=dict(colors=['#225ea8']),
    #                             figsize=(12, 4))
    #
    # ax.set_ylabel('')
    # # ax.set_title(ylbls[ii + 1][0])
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # ax.set_xlabel('Time (Days)')
    #
    # plt.tight_layout()
    # plt.show()

if flag_global_sensitivity:

    # Load
    with open('PCE/pce_model.pkl', 'rb') as f:
        pce_model = pickle.load(f)

    # Load the PCE model
    with open('PCE/pce_chain.pkl', 'rb') as f:
        chain = pickle.load(f)


    # time_indices = [50, 100, 150, 200]
    selected_indices = [50]

    # Step 1: Flatten MCMC samples into shape (n_total_samples, n_params)
    posterior_samples = chain
    param_bounds = [[np.min(posterior_samples[:, i]), np.max(posterior_samples[:, i])] for i in range(len(param_keys))]

    # SALib problem definition
    problem = {
        'num_vars': 11,
        'names': param_keys,
        'bounds': param_bounds,
    }

    # For each time point you care about
    selected_time_indices = [50, 100, 150, 200]
    # selected_time_indices = [50]

    sobol_results = []
    # Loop over selected output time indices
    for t_idx in selected_time_indices:
        print(f"\nComputing Sobol indices at time index {t_idx}...")

        # Generate Sobol samples (can increase 512 for better accuracy)
        X = saltelli.sample(problem, 512, calc_second_order=False)

        # Evaluate the surrogate at each X
        Y = np.array([surrogate_pce(x)[t_idx] for x in X])

        # Analyze
        Si = sobol.analyze(problem, Y, calc_second_order=False)

        # Store
        sobol_results.append(Si)


        print(f"Sobol indices at t = {t_idx}")
        for name, s1, st in zip(param_keys, Si['S1'], Si['ST']):
            print(f"  {name}: S1 = {s1:.3f}, ST = {st:.3f}")

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axs = axs.flatten()

    for i, (Si, t) in enumerate(zip(sobol_results, selected_time_indices)):
        axs[i].bar(param_keys, Si['S1'], color='skyblue', label='S1 (First-order)')
        axs[i].bar(param_keys, Si['ST'], color='salmon', alpha=0.3, label='ST (Total-order)')
        axs[i].set_title(f'Time index {t}')
        axs[i].set_ylim([0, 1])
        axs[i].legend()
        axs[i].set_xticklabels(param_keys, rotation=45)

    plt.tight_layout()
    plt.show()