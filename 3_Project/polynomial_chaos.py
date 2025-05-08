import numpy as np
import chaospy as cp
import os
import pickle
from pymcmcstat.MCMC import MCMC
from pymcmcstat import mcmcplot as mcp
import matplotlib.pyplot as plt
from synthetic_data_generator import solve_HIV_ode

# --------------------------------------------------------
#                   User flags
# --------------------------------------------------------
flag_samples = False
flag_fit_PCE = False
flag_bayesian_inference = True
flag_global_sensitivity = False

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
# param_keys = ["lambda1", "lambda2", "NT", "c", "bE", "Kb", "Kd", "delta_E", "delta", "lambda_E", "dE"]
param_keys = ['bE', 'delta', 'd1', 'k2', 'lambda1', 'Kb']


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

dist = cp.J(
    cp.Uniform(0.298, 0.302),   # bE
    cp.Uniform(0.67, 0.69),     # delta
    cp.Uniform(7e-3, 10.5e-3),  # d1
    cp.Uniform(0.9e-4, 1.3e-3), # k2
    cp.Uniform(0.98e4, 1.02e4), # lambda1
    cp.Uniform(70, 110),        # Kb

)

n_samples = 100
samples = dist.sample(n_samples).T # shape (n_samples, n_params)
nodes = samples.T


# --------------------------------------------------------------------
#                      MCMC with Surrogate Model
# --------------------------------------------------------------------
if flag_samples:


    def evaluate_model(params):
        param_sample_dict = {k: p for k, p in zip(param_keys, params) if k in param_keys}
        return solve_reduced_HIV_ode(param_sample_dict)[:, 5] # shape (n_times, )

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

    # Comparison of the predictions
    # param_sample = samples[0, :]
    theta_nom = [nom_params[k] for k in param_keys]
    predicted_output = surrogate_pce(theta_nom)

    # # Plotting
    # fig, ax = plt.subplots()
    # ax.plot(t_data, full_model_output, label='Full model')
    # ax.plot(t_data, surrogate_model_output, label='Surrogate model')
    # ax.legend()
    #
    # plt.tight_layout()
    # plt.show()

if flag_bayesian_inference:

    # Load the PCE model
    with open('PCE/pce_model.pkl', 'rb') as f:
        pce_model = pickle.load(f)

    def surrogate_ssq(pars, data):
        pred = surrogate_pce(pars)  # shape (n_times, 6)
        residuals = data.ydata - pred
        return np.sum(residuals**2)

    param_defs = [
        ('bE', 0.3, 0.25, 0.35),
        ('delta', 0.7, 0.01, 1.0),
        ('d1', 0.01, 0.001, 0.1),
        ('k2', 1e-4, 1e-5, 1e-3),
        ('lambda1', 1e4, 5000, 20000),
        ('Kb', 100, 60, 120)
    ]

    mcstat = MCMC()
    mcstat.data.add_data_set(x=t_data.reshape(-1, 1), y=Y_data)

    # Add same priors
    for name, val, low, high in param_defs:
        mcstat.parameters.add_model_parameter(name=name, theta0=val, minimum=low, maximum=high)

    # Use surrogate likelihood
    mcstat.model_settings.define_model_settings(sos_function=surrogate_ssq)
    mcstat.simulation_options.define_simulation_options(nsimu=40000, updatesigma=True, method='dram')

    mcstat.run_simulation()

    # Postprocessing
    results = mcstat.simulation_results.results
    burnin = 5000 # int(results['nsimu'] / 2)
    chain = results['chain'][burnin:, :]
    s2chain = results['s2chain'][burnin:, :]
    names = results['names']

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




# # Add model parameters [name, initial, min, max]
# param_defs = [
#     ('bE', 0.3, 0.25, 0.35),
#     ('delta', 0.7, 0.01, 1.0),
#     ('d1', 0.01, 0.001, 0.1),
#     ('k2', 1e-4, 1e-5, 1e-3),
#     ('lambda1', 1e4, 5000, 20000),
#     ('Kb', 100, 60, 120)
# ]



# param_defs = [
#     ('lambda1', 1e4, 5000, 20000),
#     ('lambda2', 31.98, 20, 50),
#     ('NT', 100, 50, 150),
#     ('c', 13, 5, 20),
#     ('bE', 0.3, 0.25, 0.35),
#     ('Kb', 100, 60, 120),
#     ('Kd', 500, 100, 1000),
#     ('delta_E', 0.1, 0.01, 0.2),
#     ('delta', 0.7, 0.01, 1.0),
#     ('lambda_E', 1, 0.5, 2),
#     ('dE', 0.25, 0.1, 0.5)
#     ]


# dist = cp.J(
#     cp.Uniform(0.98e4, 1.02e4),   # lambda1
#     cp.Uniform(25, 35),           # lambda2
#     cp.Uniform(80, 120),          # NT
#     cp.Uniform(8, 18),            # c
#     cp.Uniform(0.28, 0.31),       # bE
#     cp.Uniform(65, 115),          # Kb
#     cp.Uniform(300, 700),                     # Kd
#     cp.Uniform(0.01, 0.2),                    # delta_E
#     cp.Uniform(0.01, 1.),                     # delta
#     cp.Uniform(0.5, 2),                       # lambda_E
#     cp.Uniform(0.1, 0.5)                      # dE
# )

