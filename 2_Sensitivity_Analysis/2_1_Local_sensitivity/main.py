# import numpy as np
from numpy import linalg as LA
import autograd.numpy as np
from autograd import jacobian


def gamma_fun(h, k, a, b):

    gam = np.sqrt((2 * (a + b) * h) / (a * b * k))

    return gam

def partials_gamma_fun(h, k, a, b):
    gam = gamma_fun(h, k, a, b)
    dgam_dh = 1 / (2 * gam) * (2 * (a + b) / (a * b * k))
    dgam_dk = - 1 / (2 * gam) * (2 * (a + b) * h / (a * b * k ** 2))
    return dgam_dh, dgam_dk


def c1_fun(theta, gam, L):

    Phi, h, k = theta
    c1 = -Phi / (k * gam) * ((np.exp(gam * L ) * (h + k * gam)) /
                             (np.exp(-gam *  L) * (h - k * gam) + np.exp(gam * L) * (h + k * gam)))
    return c1

def c2_fun(theta, gam, L):

    Phi, h, k = theta
    c1 = c1_fun(theta, gam, L)
    c2 = Phi / (k * gam) + c1

    return c2


def temperature_function(theta, x, T_amb=21.29, L=70, a=0.95, b=0.95):
    """ Analytic solution of steady-state temperature of an insulated rod """
    Phi, h, k = theta
    gam = gamma_fun(h, k, a, b)
    c1 = c1_fun(theta, gam, L)
    c2 = c2_fun(theta, gam, L)

    Ts = c1 * np.exp(- gam * x) + c2 * np.exp(gam * x) + T_amb

    return Ts

def partials_temperature_function_AD(theta, x):
    """ Analytic solution of steady-state temperature of an insulated rod """

    partials = jacobian(temperature_function, 0)(theta, x)
    dTs_dPhi, dTs_dh, dTs_dk = partials[:, 0], partials[:, 1], partials[:, 2]

    # Scaling
    dTs_dPhi = theta[0] * dTs_dPhi
    dTs_dh = theta[1] * dTs_dh
    dTs_dk = theta[2] * dTs_dk

    return dTs_dPhi, dTs_dh, dTs_dk

def partials_temperature_function(theta, x, L=70, a=0.95, b=0.95):

    Phi, h, k = theta
    # T_star = temperature_function(theta, x, L=L, a=a, b=b)
    gam = gamma_fun(h, k, a, b)
    c1  = c1_fun(theta, gam, L)
    dgam_dh, dgam_dk = partials_gamma_fun(h, k, a, b)

    # -----------------------------------------------------------------
    #       Auxiliary variables to simplify the equations
    # -----------------------------------------------------------------
    den = (np.exp(-gam * L) * (h - k * gam) + np.exp(gam * L) * (h + k * gam))
    e_gam_x = np.exp(gam * x)
    e_gam_m_x = np.exp(-gam * x)
    e_gam_L = np.exp(gam * L)
    e_gam_m_L = np.exp(-gam * L)

    dden_dh = (e_gam_m_L * (- L * dgam_dh * (h - k * gam) + 1 - k * dgam_dh) +
               e_gam_L * (L * dgam_dh * (h + k * gam) + 1 + k * dgam_dh))

    dden_dk = (e_gam_m_L * (-L * dgam_dk * (h - k * gam) - (gam + k * dgam_dk)) +
               e_gam_L * (L * dgam_dk * (h + k * gam) + gam + k * dgam_dk))


    # -----------------------------------------------------------------
    #                          Phi partials
    # -----------------------------------------------------------------
    dTs_dPhi = -((1 / (k * gam)) * (e_gam_L * (h + k * gam)) / den) * (e_gam_m_x + e_gam_x) + (1 / (k * gam)) * e_gam_x
    # Scaling
    dTs_dPhi = Phi * dTs_dPhi

    # -----------------------------------------------------------------
    #                          h partials
    # -----------------------------------------------------------------
    dTs_dh_t1 = (e_gam_m_x + e_gam_x) * ((Phi / (k * gam**2)) * dgam_dh * (e_gam_L * (h + k * gam)) / den +
                                         (-Phi / (k * gam) * ((e_gam_L * L * dgam_dh * (h + k * gam) +  e_gam_L * (1 + k * dgam_dh)) * den
                                                              - e_gam_L * (h + k * gam) * dden_dh) / (den**2)))
    dTs_dh_t2 = c1 * (- e_gam_m_x * x * dgam_dh + e_gam_x * x * dgam_dh)
    dTs_dh_t3 = (-Phi / (k * gam**2)) * dgam_dh * e_gam_x + (Phi / (k * gam)) * e_gam_x * x * dgam_dh
    dTs_dh = dTs_dh_t1 + dTs_dh_t2 + dTs_dh_t3
    # Scaling
    dTs_dh = h * dTs_dh

    # -----------------------------------------------------------------
    #                          k partials
    # -----------------------------------------------------------------
    dTs_dk_t1 = (e_gam_m_x + e_gam_x) * ((Phi / (k * gam)**2) * (gam  + k * dgam_dk) * ((e_gam_L * (h + k * gam)) / den) +
                                         (- Phi / (k * gam)) * ((e_gam_L * L * dgam_dk * (h + k * gam) + e_gam_L * (gam + k * dgam_dk)) * den -
                                         ((e_gam_L * (h + k * gam)) * dden_dk)) / den**2)
    dTs_dk_t2 = c1 * (- x * e_gam_m_x * dgam_dk + x * e_gam_x * dgam_dk)
    dTs_dk_t3 = - (Phi / (k * gam)**2) * (e_gam_x * (gam + k * dgam_dk)) + (Phi / (k * gam)) * e_gam_x * x * dgam_dk
    dTs_dk = dTs_dk_t1 + dTs_dk_t2 + dTs_dk_t3
    # Scaling
    dTs_dk = k * dTs_dk

    return dTs_dPhi, dTs_dh, dTs_dk

def compute_sensitivity_matrix(theta, x, L=70, a=0.95, b=0.95):
    """ Compute the sensitivity matrix of the temperature function with respect to the parameters """

    dTs_dPhi, dTs_dh, dTs_dk = partials_temperature_function(theta=theta, x=x, L=L, a=a, b=b)

    # Sensitivity matrix
    S = np.array([dTs_dPhi, dTs_dh, dTs_dk]).T

    return S

def compute_Fisher_matrix(S):
    """ Compute the Fisher matrix from the sensitivity matrix """

    # Fisher matrix
    F = S.T @ S

    return F

def generate_grid(x0, dx, N):

    x_vec = np.array([x0 + (i - 1) * dx for i in range(N)])

    return x_vec

def get_ordered_eig(matrix):
    eigvals, eigvecs = LA.eig(matrix)
    idx = np.argsort(eigvals)[::-1]  # Sort in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    theta_nom_dict = {'Phi':-18.4,
                    'h': 0.00191,
                    'k': 2.37}
    theta_vec = np.array(list(theta_nom_dict.values()))
    x_vec = generate_grid(10, 4, 15)
    T_star = temperature_function(theta=theta_vec, x=x_vec)
    alpha = 1.1 # Perturbation factor

    # ------------------------------------------------------
    #              Senstivity Matrix
    # ------------------------------------------------------
    S = compute_sensitivity_matrix(theta=theta_vec, x=x_vec)
    u_vec, sigma, v_vec = LA.svd(S)
    print('Sensitivity matrix singular values:', sigma)
    print('Sensitivity matrix right singular vectors: \n', v_vec.T)

    F = compute_Fisher_matrix(S)
    eig_F, evec_F = get_ordered_eig(F)
    print('Fisher matrix eigenvalues:', eig_F)
    print('Fisher matrix eigenvectors: \n', evec_F)


    # ------------------------------------------------------
    #              Pertubation Compared
    # ------------------------------------------------------
    # # Output when perturbing Phi
    N_samples = 3
    # perturbation = 0.1
    # Phi_range = np.linspace(theta_nom_dict['Phi'] * (1 - perturbation), theta_nom_dict['Phi'] * (1 + perturbation), N_samples)
    # h_range = np.linspace(theta_nom_dict['h'] * (1 - perturbation), theta_nom_dict['h'] * (1 + perturbation), N_samples)
    # k_range = np.linspace(theta_nom_dict['k'] * (1 - perturbation), theta_nom_dict['k'] * (1 + perturbation), N_samples)
    #
    # Ts_Phi = []
    # for i in range(N_samples):
    #     Ts_Phi.append(temperature_function(theta=[Phi_range[i], theta_nom_dict['h'], theta_nom_dict['k']], x=x_vec))
    # Ts_Phi = np.array(Ts_Phi)
    #
    # Ts_h = []
    # for i in range(N_samples):
    #     Ts_h.append(temperature_function(theta=[theta_nom_dict['Phi'], h_range[i], theta_nom_dict['k']], x=x_vec))
    # Ts_h = np.array(Ts_h)
    #
    # Ts_k = []
    # for i in range(N_samples):
    #     Ts_k.append(temperature_function(theta=[theta_nom_dict['Phi'], theta_nom_dict['h'], k_range[i]], x=x_vec))
    # Ts_k = np.array(Ts_k)
    #
    #
    # # Plotting
    # pert_ls = ['Perturbation: ' + str(0.9), 'Nominal', 'Perturbation: ' + str(1.1)]
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(N_samples):
    #     ax[0].plot(x_vec, Ts_Phi[i, :], label=pert_ls[i])
    #     ax[1].plot(x_vec, Ts_h[i, :], label=pert_ls[i])
    #     ax[2].plot(x_vec, Ts_k[i, :], label=pert_ls[i])
    #
    #
    # ax[0].set_title('Temperature with Phi perturbation')
    # ax[1].set_title('Temperature with h perturbation')
    # ax[2].set_title('Temperature with k perturbation')
    # ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # plt.tight_layout()
    # plt.show()

    # ------------------------------------------------------
    #               KDES Analysis
    # ------------------------------------------------------
    N_samples = 2000
    perturbation = 0.2
    Phi_range = [theta_nom_dict['Phi'] * (1 - perturbation), theta_nom_dict['Phi'] * (1 + perturbation)]
    h_range = [theta_nom_dict['h'] * (1 - perturbation), theta_nom_dict['h'] * (1 + perturbation)]
    k_range = [theta_nom_dict['k'] * (1 - perturbation), theta_nom_dict['k'] * (1 + perturbation)]

    # Sample parameters
    Phi_samples = np.random.uniform(*Phi_range, N_samples)
    h_samples = np.random.uniform(*h_range, N_samples)
    k_samples = np.random.uniform(*k_range, N_samples)

    # Evaluate temperature function
    Ts_samples = []
    for i in range(N_samples):
        theta = [Phi_samples[i], h_samples[i], k_samples[i]]
        Ts_samples.append(temperature_function(theta=theta, x=x_vec))

    # Evalute temperature with Phi fixed
    Ts_samples_Phi_fixed = []
    for i in range(N_samples):
        theta = [theta_nom_dict['Phi'], h_samples[i], k_samples[i]]
        Ts_samples_Phi_fixed.append(temperature_function(theta=theta, x=x_vec))

    # Evaluate temperature with h fixed
    Ts_samples_h_fixed = []
    for i in range(N_samples):
        theta = [Phi_samples[i], theta_nom_dict['h'], k_samples[i]]
        Ts_samples_h_fixed.append(temperature_function(theta=theta, x=x_vec))

    # Evaluate temperature with k fixed
    Ts_samples_k_fixed = []
    for i in range(N_samples):
        theta = [Phi_samples[i], h_samples[i], theta_nom_dict['k']]
        Ts_samples_k_fixed.append(temperature_function(theta=theta, x=x_vec))


    Ts_samples = np.array(Ts_samples)
    Ts_samples_Phi_fixed = np.array(Ts_samples_Phi_fixed)
    Ts_samples_h_fixed = np.array(Ts_samples_h_fixed)
    Ts_samples_k_fixed = np.array(Ts_samples_k_fixed)
    x_idx_ls = [0, 5, 10, 14]

    # PLot KDE
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    for i, x_idx in enumerate(x_idx_ls):
        # Plotting Phi
        ax[i].hist(Ts_samples[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with Phi perturbed')
        ax[i].hist(Ts_samples_Phi_fixed[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with Phi fixed')
        ax[i].set_title('Temperature at x = ' + str(x_vec[x_idx]))
        ax[i].legend()


    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    for i, x_idx in enumerate(x_idx_ls):
        # Plotting h
        ax[i].hist(Ts_samples[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with h perturbed')
        ax[i].hist(Ts_samples_h_fixed[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with h fixed')
        ax[i].set_title('Temperature at x = ' + str(x_vec[x_idx]))
        ax[i].legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    for i, x_idx in enumerate(x_idx_ls):
        # Plotting k
        ax[i].hist(Ts_samples[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with k perturbed')
        ax[i].hist(Ts_samples_k_fixed[:, x_idx], bins=30, density=True, alpha=0.5, label='KDE with k fixed')
        ax[i].set_title('Temperature at x = ' + str(x_vec[x_idx]))
        ax[i].legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    #           Sensitivity Methods Comparison
    # ------------------------------------------------------
    # Compute sensitivities at multiple points
    sens_dict = {'FD': {},
                 'CS': {},
                 'AN': {},
                 'AD': {}}

    for idx, (key, theta) in enumerate(theta_nom_dict.items()):
        sens_dict['FD'][key] = []
        sens_dict['CS'][key] = []

        for x in x_vec:

            # Finite difference
            theta_pert_vec = theta_vec.copy()
            theta_pert_vec[idx] = theta * alpha
            d_theta = theta_pert_vec[idx] - theta_vec[idx]
            sens_dict['FD'][key].append(theta_vec[idx] * ((temperature_function(theta_pert_vec, x) - temperature_function(theta_vec, x)) / d_theta))

            # Complex step
            theta_pert_im_vec = theta_vec.copy().astype(complex)
            theta_pert_im_vec[idx] = theta_vec[idx] + d_theta * 1j
            sens_dict['CS'][key].append(theta_vec[idx] * np.imag(temperature_function(theta_pert_im_vec, x)) / d_theta)

    # Analytic derivatives
    dTs_dPhi, dTs_dh, dTs_dk = partials_temperature_function(theta=theta_vec, x=x_vec)
    sens_dict['AN']['Phi'] = dTs_dPhi
    sens_dict['AN']['h'] = dTs_dh
    sens_dict['AN']['k'] = dTs_dk

    # Automatic differentiation derivatives
    dTs_dPhi_AD, dTs_dh_AD, dTs_dk_AD = partials_temperature_function_AD(theta=theta_vec, x=x_vec)
    sens_dict['AD']['Phi'] = dTs_dPhi_AD
    sens_dict['AD']['h'] = dTs_dh_AD
    sens_dict['AD']['k'] = dTs_dk_AD


    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, key in enumerate(theta_nom_dict.keys()):
        ax[i].plot(x_vec, sens_dict['FD'][key], label='Finite Difference')
        ax[i].plot(x_vec, sens_dict['CS'][key], label='Complex Step')
        ax[i].plot(x_vec, sens_dict['AN'][key], label='Analytic')
        ax[i].plot(x_vec, sens_dict['AD'][key], label='AD')

        ax[i].set_title(f'Sensitivity of {key}')
        ax[i].legend()

    plt.tight_layout()
    plt.show()




