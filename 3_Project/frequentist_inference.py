__author__ = 'Dario Rodriguez'

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import SymLogNorm
import numpy as np
import pickle
import os
import seaborn as sns
import  pandas as pd
import corner
from scipy.stats import t
from scipy.optimize import least_squares
from synthetic_data_generator import solve_HIV_ode

flag_optimization = False
flag_statistics = True

# -----------------------------------------
#            Load data
# -----------------------------------------
QoI_idx = 5

hiv_data = np.loadtxt('data/hiv_data.txt', delimiter=',')  # shape (N, 7) [t, T1, T2, T1s, T2s, V, E]
t_data = hiv_data[:, 0]
Y_data = hiv_data[:, 1:]

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

# Choose parameters to estimate
param_keys = ["lambda1", "lambda2", "NT", "c", "bE", "Kb", "Kd", "delta_E", "delta", "lambda_E", "dE"]
x_scale = [1e4, 30, 100, 10, 1/10, 100, 500, 1/10, 1/10, 1, 1/10]
theta_true = [nom_params[key] for key in param_keys]

def residuals(theta, y_measured, param_keys, plot=False):
    # Build param dictionary
    params_est = nom_params.copy()
    for key, val in zip(param_keys, theta):
        params_est[key] = val

    # Solve system
    _, y_est = solve_HIV_ode(params_est, t_step=1)
    y_est = y_est.T

    # Residuals between simulated and measured
    res = y_est[:, QoI_idx] - y_measured[:, QoI_idx]

    if plot:
        fontsize = 14
        tick_fontsize = 13
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(t_data, y_est[:, QoI_idx], label='Estimated')
        ax[0].plot(t_data, y_measured[:, QoI_idx], 'x', label='Measured')
        ax[0].set_xlabel('Time', fontsize=fontsize)
        ax[0].set_ylabel('E', fontsize=fontsize)
        ax[0].legend(fontsize=fontsize)
        ax[0].tick_params(labelsize=tick_fontsize)

        ax[1].plot(t_data, res, 'o')
        ax[1].set_xlabel('Time', fontsize=fontsize)
        ax[1].set_ylabel('Residuals', fontsize=fontsize)
        ax[1].tick_params(labelsize=tick_fontsize)

        plt.tight_layout()
        plt.show()

    return res

def inverse_Fisher_matrix(param_sub_dict):

    params_est = nom_params.copy()
    for key, val in param_sub_dict.items():
        params_est[key] = val

    _, y_star_vec = solve_HIV_ode(params_est, t_step=1)
    y_star = y_star_vec[QoI_idx]

    S = np.zeros((len(y_star), len(param_sub_dict)))
    h = 5e-12

    for idx, param in enumerate(param_sub_dict.keys()):
        # Perturb param
        params_est_pert = params_est.copy()
        params_est_pert[param] = params_est[param] + h

        _, y_pert_vec = solve_HIV_ode(params_est_pert, t_step=1)
        y_pert = y_pert_vec[QoI_idx]

        # Compute gradient with scaling
        theta = params_est[param]
        S[:, idx] = (theta / y_star) * (y_pert - y_star) / h

    # Compute Fisher information matrix
    F = S.T @ S
    F_inv = np.linalg.inv(F)

    return F_inv


# -----------------------------------------
#        Least Squares Optimization
# -----------------------------------------
if flag_optimization:
    # Initial guess
    theta0 = theta_true
    # theta0 = [val * (1 + 0.2*np.random.randn()) for val in theta_true]  # With perturbation


    result = least_squares(residuals, theta0, args=(Y_data, param_keys), x_scale=x_scale,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9,
                           bounds=(0, np.inf), method='trf', verbose=2)

    # Get estimated parameters
    estimated_params = {key: val for key, val in zip(param_keys, result.x)}
    print("\nEstimated Parameters:")
    for key, val in estimated_params.items():
        print(f"{key}: {val:.4g}")

    # Store result as pickle
    data_dir = 'frequentist_analysis'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(data_dir + '/results.pkl', 'wb') as f:
        pickle.dump(result, f)

# -----------------------------------------
#           Error variance
# -----------------------------------------
if flag_statistics:

    data_dir = 'frequentist_analysis'
    # Load results from least squares
    with open(data_dir + '/results.pkl', 'rb') as f:
        result = pickle.load(f)

    # Get estimated parameters
    estimated_params = {key: float(val) for key, val in zip(param_keys, result.x)}

    # Compute residuals
    R = residuals(estimated_params.values(), Y_data, param_keys, plot=True)
    n, p = Y_data.shape

    # Compute error variance
    sigma_sq = (1 / (n - p)) * (R.T @ R)
    F_inv = inverse_Fisher_matrix(estimated_params)
    V = sigma_sq * F_inv

    df_V = pd.DataFrame(V, index=param_keys, columns=param_keys).applymap(lambda x: f'{x:.2e}')
    print(df_V)



    # Plot covariance as heat map
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(V, ax=ax, annot=True, fmt=".2f", cmap='coolwarm',
                norm=SymLogNorm(linthresh=1e-5, linscale=1.0, vmin=V.min(), vmax=V.max()),
                cbar=True,
                xticklabels=param_keys, yticklabels=param_keys, square=True, cbar_kws={'label': 'Covariance'})
    ax.set_title('Covariance Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()


    # Confidence intervals
    alpha = 0.05
    t_val = t.ppf(1 - alpha / 2, n - p)
    confidence_df = pd.DataFrame(index=param_keys, columns=['value', 'lower', 'upper'])
    for i, key in enumerate(param_keys):
        val = estimated_params[key]
        err = t_val * np.sqrt(V[i, i])
        confidence_df.loc[key] = [val, val - err, val + err]

    print(confidence_df)


    # # Generate samples from multivariate normal distribution
    # samples = np.random.multivariate_normal(np.array(list(estimated_params.values())), V, size=1000)
    #
    # # Create DataFrame
    # df = pd.DataFrame(samples, columns=param_keys)
    #
    # # Set up seaborn style
    # sns.set(style="ticks", font_scale=0.9)
    #
    # # Create pairplot
    # g = sns.pairplot(df, corner=True, plot_kws={'s': 5, 'alpha': 0.4})
    #
    # # Apply scientific notation to all axes
    # for ax in g.axes.flatten():
    #     if ax is not None:
    #         ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #         ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    #
    # # Resize the figure to fit all axes labels
    # g.fig.set_size_inches(15, 15)  # Adjust size as needed
    #
    # # Tight layout to prevent label cutoff
    # plt.tight_layout(pad=3.0)
    # plt.show()


    # # Create the corner plot
    # figure = corner.corner(samples,
    #                        labels=[param_keys[i] for i in range(11)],
    #                        show_titles=True,
    #                        title_fmt=".2f",
    #                        title_kwargs={"fontsize": 12},
    #                        label_kwargs={"fontsize": 10},
    #                        figsize=(15, 15),  # increase figure size
    #                        max_n_ticks=3)
    #
    # # Adjust layout to prevent label cut-off
    # plt.tight_layout(pad=0.5)
    # plt.show()