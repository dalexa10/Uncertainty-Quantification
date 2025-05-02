from  synthetic_data_generator import solve_HIV_ode
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

flag_compute = False
flag_postprocessing = False
flag_plot_KDE = True

current_file_dir = os.path.dirname(os.path.abspath(__file__))

params = {
    "lambda1": 1e4,
    "d1": 0.01,
    "epsilon": 0,
    "k1": 8e-7,
    "lambda2": 31.98,
    "d2": 0.01,
    "f": 0.34,
    "k2": 1e-4,  # 1e-4
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

t_span = (0, 200)
t_step = 0.1
t_eval = np.arange(t_span[0], t_span[1] + t_step, t_step)


if flag_compute:

    QoI_idx = 5

    pert = 0.1
    h = 1e-12

    S = np.zeros((len(t_eval), len(params)))

    t_star, y_star_vec = solve_HIV_ode(params, t_span, t_step)

    # Extract the state variable of interest (Immune effector cell)
    y_star = y_star_vec[QoI_idx]

    for idx, (key, theta) in enumerate(params.items()):

        # Perturb the parameter
        params_pert = params.copy()
        params_pert[key] = theta + h

        # Solve the ODE with perturbed parameters
        t, y_pert_vec = solve_HIV_ode(params_pert, t_span, t_step)
        y_pert = y_pert_vec[QoI_idx]

        # Compute the gradient with scaling
        S[:, idx] = (theta / y_star) * (y_pert - y_star) / h


    # Compute Fisher information matrix
    F = S.T @ S

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(F)
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Save the results
    with open('local_sensitivity/data_dict.pkl', 'wb') as f:
        pickle.dump({'F': F, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}, f)


if flag_postprocessing:
    # Load the results
    with open(current_file_dir +'/local_sensitivity/data_dict.pkl', 'rb') as f:
        data = pickle.load(f)

    F = data['F']
    eigenvalues = data['eigenvalues']
    eigenvectors = data['eigenvectors']
    eigenvectors_abs = np.abs(eigenvectors)

    eta = 10e-10

    # Find nonidentifiable parameters
    ratios = eigenvalues / eigenvalues[0]
    index_m = np.where(ratios > eta)[0][-1]

    non_id_params = []
    for i in range(len(params) - 1, index_m, -1):
        idx_param = np.argmax(eigenvectors_abs[:, i])
        non_id_params.append(list(params.keys())[idx_param])

    id_params = [param for param in params.keys() if param not in non_id_params]

    print('\n')
    print('Non-identifiable parameters:')
    print(non_id_params)
    print('\n')

    print('Identifiable parameters:')
    print(id_params)
    print('\n')


    N_samples = 10
    pert = 0.1
    pert_params = {}
    for key, theta in params.items():
        pert_params[key] = np.random.uniform(theta * (1 - pert), theta * (1 + pert), N_samples)

    params_cases = [dict(zip(pert_params.keys(), pert_param)) for pert_param in zip(*pert_params.values())]

    # ------------------------------------------------------
    #           Analysis of non-identifiable parameters
    # ------------------------------------------------------
    stat_data_dict = {}

    # All random parameters
    stat_data_dict['all_random'] = []
    for i in range(N_samples):
        t, y = solve_HIV_ode(params_cases[i], t_span, t_step)
        stat_data_dict['all_random'].append(y)

    stat_data_dict['all_random'] = np.array(stat_data_dict['all_random'])

    # Fixing noninfluential parameters
    for non_id_param in non_id_params:

        stat_data_dict[non_id_param] = []

        for i in range(N_samples):
            param_cases_fix = params_cases[i].copy()
            # Fix the parameter
            param_cases_fix[non_id_param] = params[non_id_param]
            # Solve the ODE
            _, y = solve_HIV_ode(param_cases_fix, t_span, t_step)
            # Store the results
            stat_data_dict[non_id_param].append(y)

    for non_id_param in non_id_params:
        stat_data_dict[non_id_param] = np.array(stat_data_dict[non_id_param])


    # Save the results
    with open(current_file_dir + '/local_sensitivity/stat_data_dict.pkl', 'wb') as f:
        pickle.dump(stat_data_dict, f)


if flag_plot_KDE:

    fontsize = 14
    tick_fontsize = 13

    # Load the results
    with open(current_file_dir + '/local_sensitivity/stat_data_dict.pkl', 'rb') as f:
        stat_data_dict = pickle.load(f)

    t_plot_ls = [50, 100, 150, 200]
    QoI_idx = 5 # Immune effector cell

    # Find index of t_plot in t_eval
    idx_plot_ls = [np.argmin(np.abs(t_eval - t)) for t in t_plot_ls]

    for non_id_param in stat_data_dict.keys():
        if non_id_param != 'all_random':

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            for i, t_plot in enumerate(t_plot_ls):
                # KDE for all random parameters
                sns.kdeplot(
                    stat_data_dict['all_random'][:, QoI_idx, idx_plot_ls[i]], label='Perturbed',
                    fill=True, alpha=0.5, linewidth=2, ax=ax[i], linestyle='-'
                )
                # KDE for fixed parameter
                sns.kdeplot(
                    stat_data_dict[non_id_param][:, QoI_idx, idx_plot_ls[i]], label='Fixed f',
                    fill=True, alpha=0.4, linewidth=2, ax=ax[i], linestyle='--'
                )

                ax[i].set_title(f'Time = {t_plot} days', fontsize=fontsize)
                ax[i].legend(fontsize=fontsize)

            [ax[i].tick_params(labelsize=tick_fontsize) for i in range(len(ax))]
            [ax[i].set_ylabel('PDF', fontsize=tick_fontsize) for i in range(len(ax))]

            fig.suptitle(f'KDE for {non_id_param} parameter', fontsize=fontsize)

            plt.tight_layout()
            plt.show()

