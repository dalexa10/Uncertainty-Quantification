__author__ = 'Dario Rodriguez'

import numpy as np
import pickle
# from scipy.sparse.linalg import eigs
from numpy.linalg import eigh
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt


def compute_empirical_covariance(X):
    """
    Computes the empirical covariance matrix of a dataset X.
    :param
        X: data, matrix of size (n_observations, n_grid), assumed uncentered
    :return:
        C: empirical covariance, matrix of size (n_grid, n_grid)
    """
    # Compute the mean along the observation axis
    mean_vector = np.mean(X, axis=0)

    # Center the data
    X_cent = X - mean_vector

    # Empirical covariance matrix
    C = (X_cent.T @ X_cent) / (X.shape[0] - 1)

    return C

def compute_eigenvalues_eigenvectors(X):
    """
    Computes the eigenvalues and eigenvectors of a matrix Ca and sorts them in descending order.
    :param
        C: matrix, assumed symmetric
    :return:
        eigvals: eigenvalues of C, array of size (n_grid,)
        eigvecs: eigenvectors of C, matrix of size (n_grid, n_grid)
    """
    C = compute_empirical_covariance(X)
    eigvals, eigvecs = eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals_norm = eigvals[idx] / np.max(eigvals)
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvals_norm, eigvecs

def transcendental_eqn_even(L):
    return lambda eta: 1 - L * eta * np.tan(eta)

def transcental_eqn_odd(L):
    return lambda eta: L * eta + np.tan(eta)

def eig_analytic_even(eta_even, L, x):
    """
    Calculate the analytic eigenvalue and eigenfunction values for an even mode
    :param
        eta: even 'n' root of the transcendental equation
        L: correlation length
        x: grid points
    :return:
        eigval: eigenvalue
        eigvec: eigenfunction
    """
    # Eigenvalue
    lambda_even = 1 / (1 + L**2 * eta_even**2)

    # Eigenfunction evaluation
    phi_even = np.cos(eta_even * x) / np.sqrt(1 + np.sin(2 * eta_even) / (2 * eta_even + 1e-7))

    return lambda_even, phi_even

def eig_analytic_odd(eta_odd, L, x):
    """
    Calculate the analytic eigenvalue and eigenfunction values for an odd mode
    :param
        eta: odd 'n' root of the transcendental equation
        L: correlation length
        x: grid points
    :return:
        eigval: eigenvalue
        eigvec: eigenfunction
    """
    # Eigenvalue
    lambda_odd = 1 / (1 + L**2 * eta_odd**2)

    # Eigenfunction evaluation
    phi_odd = np.sin(eta_odd * x) / np.sqrt(1 - np.sin(2 * eta_odd) / (2 * eta_odd + 1e-7))

    return lambda_odd, phi_odd

def find_n_roots_transcendal(n, L):

    roots_odd = []
    roots_even = []

    n_even = n // 2
    n_odd = n - n_even

    # Find n_even roots
    for i in range(n_even):
        eta0_even = (2 * i + 1) * np.pi / 2 - 1e-4
        eta_root_even = root_scalar(transcendental_eqn_even(L), x0=eta0_even, method='newton')
        roots_even.append(eta_root_even.root)

    # Find n_odd roots
    for i in range(n_odd):
        eta0_odd = (i + 0.5) * np.pi
        eta_root_odd = root_scalar(transcental_eqn_odd(L), x0=eta0_odd, method='newton')
        roots_odd.append(eta_root_odd.root)

    roots = [val for pair in zip(roots_even, roots_odd) for val in pair]
    roots += roots_even[len(roots_odd):] + roots_odd[len(roots_even):]
    roots.insert(0, 0.) # Trivial solution

    return roots


def compute_analytic_eig(L, x, n=10):
    """
    Computes the analytic eigenvalues and eigenfunctions of the covariance matrix.
    :param
        L: correlation length
        x: grid points
        n: number of eigenvalues to compute
    :return:
        eigvals: eigenvalues of C, array of size (n_grid,)
        eigvecs: eigenvectors of C, matrix of size (n_grid, n_grid)
    """
    eig_vals_ls = []
    eig_vecs_ls = []

    # Find the 'n' roots of the transcendental equation
    roots = find_n_roots_transcendal(n, L)

    # Compute the eigenvalues and eigenfunctions
    for i, eta in enumerate(roots, start=1):
        if i % 2 == 0:
            eigval_i, eigvec_i = eig_analytic_even(eta, L, x)
        else:
            eigval_i, eigvec_i = eig_analytic_odd(eta, L, x)

        eig_vals_ls.append(eigval_i)
        eig_vecs_ls.append(eigvec_i)

    eigvals = np.array(eig_vals_ls)
    eigvecs = np.array(eig_vecs_ls)

    return eigvals, eigvecs

def analytic_mean_stochastic_process(x, D=2):
    """
    Computes the mean of the stochastic process.
    :param
        x: grid points
        D: diffusion coefficient
    :return:
        mean: mean of the stochastic process
    """
    alpha_mean = 1 + np.sin(np.pi * ((x/D) + 0.5))

    return alpha_mean

def consruct_KL_expansion(alpha_samples, eigvals, eigvecs, n_modes):
    """
    Constructs the Karhunen-Loeve expansion of the stochastic process.
    :param
        alpha_mean: mean of the stochastic process at the grid points
        eigvals: eigenvalues of C, array of size (n_grid,)
        eigvecs: eigenvectors of C, matrix of size (n_grid, n_grid)
        n_modes: number of modes to consider
    :return:
        KL_expansion: Karhunen-Loeve expansion of the stochastic process
    """

    # Compute the coefficients Y_i of the KL expansion
    alpha_mean = np.mean(alpha_samples, axis=0)
    Y_i = ((alpha_samples - alpha_mean) @ eigvecs[:, :n_modes]) / np.sqrt(eigvals[:n_modes])

    # Step Reconstruct the Process Using KLE
    alpha_reconstructed = alpha_mean + (Y_i @ (np.sqrt(eigvals[:n_modes]) * eigvecs[:, :n_modes]).T)

    return alpha_reconstructed



if __name__ == '__main__':

    # ----------------------------------------------------
    #                   Root Finding
    # ----------------------------------------------------

    # roots_01 = find_n_roots_transcendal(10, 0.01)
    # roots_1 = find_n_roots_transcendal(10, 1)
    # roots_10 = find_n_roots_transcendal(10, 10)

    # print(roots_01)
    # print(roots_1)
    # print(roots_10)

    # ----------------------------------------------------
    #                   Inputs
    # ----------------------------------------------------
    n_samples = 10
    n_eigenvals = 10
    n_KL_expansion = [1, 5, 10]

    # ----------------------------------------------------
    #                   Running code
    # ----------------------------------------------------
    samples_001 = np.loadtxt('data/samples_0.01.txt')
    samples_1 = np.loadtxt('data/samples_1.txt')
    samples_10 = np.loadtxt('data/samples_10.txt')
    x_grid = np.loadtxt('data/xgrid.txt')

    # ----------------------------------------------------
    #               Plot sample realizations
    # ----------------------------------------------------
    # idx_rnd = np.random.choice(samples_001.shape[0], size=n_samples, replace=False)
    # alpha_samples_001_rnd = samples_001[idx_rnd]
    # alpha_samples_1_rnd = samples_1[idx_rnd]
    # alpha_samples_10_rnd = samples_10[idx_rnd]
    #
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(n_samples):
    #     ax[0].plot(x_grid, alpha_samples_001_rnd[i], label=f'Sample {i}')
    #     ax[1].plot(x_grid, alpha_samples_1_rnd[i], label=f'Sample {i}')
    #     ax[2].plot(x_grid, alpha_samples_10_rnd[i], label=f'Sample {i}')
    #
    # ax[0].set_title('L=0.01')
    # ax[1].set_title('L=1')
    # ax[2].set_title('L=10')
    #
    # plt.tight_layout()
    # plt.show()

    # ----------------------------------------------------
    #             Plot mean and std of the process
    # ----------------------------------------------------
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #
    # # Analytic mean
    # ax[0].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
    # ax[1].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
    # ax[2].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
    #
    # # Empirical mean
    # ax[0].plot(x_grid, np.mean(samples_001, axis=0), label='Empirical Mean')
    # ax[1].plot(x_grid, np.mean(samples_1, axis=0), label='Empirical Mean')
    # ax[2].plot(x_grid, np.mean(samples_10, axis=0), label='Empirical Mean')
    #
    # # Empirical std
    # ax[0].fill_between(x_grid, np.mean(samples_001, axis=0) - 2 * np.std(samples_001, axis=0),
    #                    np.mean(samples_001, axis=0) + 2 * np.std(samples_001, axis=0), color='blue', alpha=0.1,
    #                    label='95% Confidence Interval')
    # ax[1].fill_between(x_grid, np.mean(samples_1, axis=0) - 2 * np.std(samples_1, axis=0),
    #                       np.mean(samples_1, axis=0) + 2 * np.std(samples_1, axis=0), color='blue', alpha=0.1,
    #                       label='95% Confidence Interval')
    # ax[2].fill_between(x_grid, np.mean(samples_10, axis=0) - 2 * np.std(samples_10, axis=0),
    #                         np.mean(samples_10, axis=0) + 2 * np.std(samples_10, axis=0), color='blue', alpha=0.1,
    #                         label='95% Confidence Interval')
    #
    # plt.tight_layout()
    # plt.show()


    # ----------------------------------------------------------------------
    #                       Eigenvalues and Eigenfunctions
    # ----------------------------------------------------------------------

    eigval_an01, eigvec_an01 = compute_analytic_eig(0.01, x_grid, n=11)
    eigval_emp01, eigval_norm_emp01, eigvec_emp01 = compute_eigenvalues_eigenvectors(samples_001)
    eigval_an1, eigvec_an1 = compute_analytic_eig(1, x_grid, n=11)
    eigval_emp1, eigval_norm_emp1, eigvec_emp1 = compute_eigenvalues_eigenvectors(samples_1)
    eigval_an10, eigvec_an10 = compute_analytic_eig(10, x_grid, n=11)
    eigval_emp10, eigval_norm_emp10, eigvec_emp10 = compute_eigenvalues_eigenvectors(samples_10)


    # # ---------------------------------------------------------------------
    # #                       Eigenvalues comparison
    # # ---------------------------------------------------------------------

    fig, ax = plt.subplots()
    idx_ls = np.arange(1, 11)
    ax.plot(idx_ls, eigval_an01[:idx_ls.shape[0]], label='Analytic L=0.01')
    ax.plot(idx_ls, eigval_an1[:idx_ls.shape[0]], label='Analytic L=1')
    ax.plot(idx_ls, eigval_an10[:idx_ls.shape[0]], label='Analytic L=10')
    ax.plot(idx_ls, eigval_norm_emp01[:idx_ls.shape[0]], label='Empirical L=0.01')
    ax.plot(idx_ls, eigval_norm_emp1[:idx_ls.shape[0]], label='Empirical L=1')
    ax.plot(idx_ls, eigval_norm_emp10[:idx_ls.shape[0]], label='Empirical L=10')

    # Set log scale to y axis
    ax.set_yscale('log')

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    plt.show()

    # ---------------------------------------------------------------------
    #                       Eigenfunctions comparison
    # ---------------------------------------------------------------------
    # n_eigenfunctions = 5
    #
    # fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    # for i in range(0, n_eigenfunctions):
    #     ax[0, 0].plot(x_grid, eigvec_an01[i, :], label=f'Analytic L=0.01, eig {i}')
    #     ax[0, 1].plot(x_grid, eigvec_an1[i, :], label=f'Analytic L=1, eig {i}')
    #     ax[0, 2].plot(x_grid, eigvec_an10[i, :], label=f'Analytic L=10, eig {i}')
    #
    #     ax[1, 0].plot(x_grid, eigvec_emp01[:, i], label=f'Empirical L=0.01, eig {i}')
    #     ax[1, 1].plot(x_grid, eigvec_emp1[:, i], label=f'Empirical L=1, eig {i}')
    #     ax[1, 2].plot(x_grid, eigvec_emp10[:, i], label=f'Empirical L=10, eig {i}')
    #
    # ax[1, 0].legend()
    # ax[1, 1].legend()
    # ax[1, 2].legend()
    #
    # plt.tight_layout()
    # plt.show()

    # ------------------------------------------------------------
    #                        KL Expansion
    # ------------------------------------------------------------
    samples_01_rc = {}
    samples_1_rc = {}
    samples_10_rc = {}

    for n in n_KL_expansion:
        samples_01_rc[n] = consruct_KL_expansion(samples_001, eigval_emp01, eigvec_emp01, n)
        samples_1_rc[n] = consruct_KL_expansion(samples_1, eigval_emp1, eigvec_emp1, n)
        samples_10_rc[n] = consruct_KL_expansion(samples_10, eigval_emp10, eigvec_emp10, n)

    # Store the reconstructed samples as pkl file
    with open('data/samples_01_rc.pkl', 'wb') as file:
        pickle.dump(samples_01_rc, file)

    with open('data/samples_1_rc.pkl', 'wb') as file:
        pickle.dump(samples_1_rc, file)

    with open('data/samples_10_rc.pkl', 'wb') as file:
        pickle.dump(samples_10_rc, file)

    # Plot the KL expansion mean and std
    fig, ax = plt.subplots(3, 4, figsize=(15, 15))

    ax[0, 0].plot(x_grid, np.mean(samples_001, axis=0), label='L=0.01')
    ax[0, 0].fill_between(x_grid, np.mean(samples_001, axis=0) - 2 * np.std(samples_001, axis=0),
                       np.mean(samples_001, axis=0) + 2 * np.std(samples_001, axis=0), color='blue', alpha=0.1,
                       label='95% Confidence Interval')

    ax[1, 0].plot(x_grid, np.mean(samples_1, axis=0), label='L=1')
    ax[1, 0].fill_between(x_grid, np.mean(samples_1, axis=0) - 2 * np.std(samples_1, axis=0),
                          np.mean(samples_1, axis=0) + 2 * np.std(samples_1, axis=0), color='blue', alpha=0.1,
                          label='95% Confidence Interval')

    ax[2, 0].plot(x_grid, np.mean(samples_10, axis=0), label='L=10')
    ax[2, 0].fill_between(x_grid, np.mean(samples_10, axis=0) - 2 * np.std(samples_10, axis=0),
                          np.mean(samples_10, axis=0) + 2 * np.std(samples_10, axis=0), color='blue', alpha=0.1,
                          label='95% Confidence Interval')



    for i, n in enumerate(n_KL_expansion, start=1):
        # ax[0, i].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
        ax[0, i].plot(x_grid, np.mean(samples_01_rc[n], axis=0), label='Empirical Mean')
        ax[0, i].fill_between(x_grid, np.mean(samples_01_rc[n], axis=0) - 2 * np.std(samples_01_rc[n], axis=0),
                              np.mean(samples_01_rc[n], axis=0) + 2 * np.std(samples_01_rc[n], axis=0), color='blue', alpha=0.1,
                              label='95% Confidence Interval')
        ax[0, i].set_title(f'L=0.01, n={n}')

        # ax[1, i].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
        ax[1, i].plot(x_grid, np.mean(samples_1_rc[n], axis=0), label='Empirical Mean')
        ax[1, i].fill_between(x_grid, np.mean(samples_1_rc[n], axis=0) - 2 * np.std(samples_1_rc[n], axis=0),
                              np.mean(samples_1_rc[n], axis=0) + 2 * np.std(samples_1_rc[n], axis=0), color='blue', alpha=0.1,
                              label='95% Confidence Interval')
        ax[1, i].set_title(f'L=1, n={n}')

        # ax[2, i].plot(x_grid, analytic_mean_stochastic_process(x_grid), label='Analytic Mean')
        ax[2, i].plot(x_grid, np.mean(samples_10_rc[n], axis=0), label='Empirical Mean')
        ax[2, i].fill_between(x_grid, np.mean(samples_10_rc[n], axis=0) - 2 * np.std(samples_10_rc[n], axis=0),
                              np.mean(samples_10_rc[n], axis=0) + 2 * np.std(samples_10_rc[n], axis=0), color='blue', alpha=0.1,
                              label='95% Confidence Interval')
        ax[2, i].set_title(f'L=10, n={n}')

    # plt.tight_layout()
    plt.show()

