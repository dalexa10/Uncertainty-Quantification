import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def Helmholtz_energy_fun(theta, P_vec):
    alpha_1, alpha_11, alpha_111 = theta
    psi = alpha_1 * P_vec**2 + alpha_11 * P_vec**4 + alpha_11*P_vec**6
    return psi

def Helmholtz_energy_scalar(theta):
    alpha_1, alpha_11, alpha_111 = theta
    y = alpha_1 * ((0.8)**3 / 3) + alpha_11 * ((0.8)**5 / 5) + alpha_111 * ((0.8)**7 / 7)
    return y

def compute_sensitivity_matrix(P_vec):
    S = np.zeros((len(P_vec), 3))
    for i, P in enumerate(P_vec):
        S[i, 0] = P**2
        S[i, 1] = P**4
        S[i, 2] = P**6
    return S

def compute_Fisher_matrix(S):
    """ Compute the Fisher matrix from the sensitivity matrix """

    # Fisher matrix
    F = S.T @ S

    return F

def compute_Sobol_indices(theta_nom, M, pert=0.2):

    # Compute limits
    a = theta_nom - pert * theta_nom
    b = theta_nom + pert * theta_nom

    # Sample parameters
    rnd_matrix = np.random.rand(M, 3)
    rnd_hat_matrix = np.random.rand(M, 3)

    # Denormalize parameters
    A = rnd_matrix * (b - a) + a
    B = rnd_hat_matrix * (b - a) + a
    D = np.concat([A, B], axis=0)

    C_i_ls = []
    for i in range(A.shape[1]):
        C_i = A.copy()
        C_i[:, i] = B[:, i]
        C_i_ls.append(C_i)

    # Compute vector of model outputs
    y_A = Helmholtz_energy_scalar(theta=[A[:, 0], A[:, 1], A[:, 2]])

    # Fix less influential parameter (guessing it's only alpha_111)
    fix_alpha111 = theta_nom[2] * np.ones(M)
    # fix_alpha11 = theta_nom[1] * np.ones(M)
    y_A_np = Helmholtz_energy_scalar(theta=[A[:, 0], A[:, 1], fix_alpha111])

    y_B = Helmholtz_energy_scalar(theta=[B[:, 0], B[:, 1], B[:, 2]])
    y_C_ls = []
    for i in range(len(C_i_ls)):
        y_C = Helmholtz_energy_scalar(theta=[C_i_ls[i][:, 0], C_i_ls[i][:, 1], C_i_ls[i][:, 2]])
        y_C_ls.append(y_C)
    y_D = Helmholtz_energy_scalar(theta=[D[:, 0], D[:, 1], D[:, 2]])

    E_y_D = np.mean(y_D)

    # Compute first order and total Sobol indices
    S_ls = []
    S_T_ls = []
    for i in range(len(C_i_ls)):
        y_C = y_C_ls[i]
        S = ((1 / M) * (y_B.T@y_C - y_B.T@y_A)) / ((1/ (2 * M)) * y_D.T@y_D - E_y_D**2)
        S_ls.append(S)
        S_T = ((1 / (2 * M)) * (y_A.T@y_A - 2 * y_A.T @ y_C + y_C.T @ y_C)) / ((1 / (2 * M)) * y_D.T@y_D - E_y_D**2)
        S_T_ls.append(S_T)

    return S_ls, S_T_ls, (y_A, y_A_np)


def Morris_screening_forward_diff(theta_nom, R, Delta, pert=0.2):

    # Compute limits
    a = theta_nom - pert * theta_nom
    b = theta_nom + pert * theta_nom

    # Sample parameters
    rnd_matrix = np.random.rand(R, 3)

    # Denormalize parameters
    theta_j_matrix = rnd_matrix * (b - a) + a

    # Compute derivatives
    d_ij = np.zeros((R, 3))
    for i in range(3):
        for j in range(R):
            theta_j_pDelta = theta_j_matrix[j, :].copy()
            theta_j_pDelta[i] = theta_j_pDelta[i] + Delta
            d_ij[j, i] = (Helmholtz_energy_scalar(theta_j_pDelta) - Helmholtz_energy_scalar(theta_j_matrix[j, :])) / Delta

    # Compute Morris screening
    mu_i_star = (1 / R) * np.sum(abs(d_ij), axis=0)
    mu_i = (1 / R) * np.sum(d_ij, axis=0)
    sigma_sqr_i = (1 / (1 - R)) * np.sum((d_ij - mu_i)**2, axis=0)

    return mu_i_star, sigma_sqr_i


def get_ordered_eig(matrix):
    eigvals, eigvecs = LA.eig(matrix)
    idx = np.argsort(eigvals)[::-1]  # Sort in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

def compute_analytical_Sobol_indices(c_coeffs, sigma_sqr_coeffs):

    D_i = c_coeffs**2 * sigma_sqr_coeffs
    D = np.sum(D_i)
    S_i = D_i / D

    return S_i


if __name__ == '__main__':

    P_vec = np.linspace(0, 0.8, 17)
    theta_nom = np.array([-382.7, 760.3, 155.1])
    c_coeffs = np.array([0.8**3 / 3, 0.8**5 / 5, 0.8**7 / 7])
    sigma_sqr_coeffs = np.array([117, 3000, 4627.7])


    # ------------------------------------------------------
    #              Senstivity Matrix
    # ------------------------------------------------------
    S = compute_sensitivity_matrix(P_vec=P_vec)
    u_vec, sigma, v_vec = LA.svd(S)
    print('Sensitivity matrix singular values:', sigma)
    print('Sensitivity matrix right singular vectors: \n', v_vec.T)

    F = compute_Fisher_matrix(S)
    eig_F, evec_F = get_ordered_eig(F)
    print('Fisher matrix eigenvalues:', eig_F)
    print('Fisher matrix eigenvectors: \n', evec_F)

    # ------------------------------------------------------
    #              Morris Screening
    # ------------------------------------------------------
    mu_i_star, sigma_i_sqr = Morris_screening_forward_diff(theta_nom=theta_nom, R=50, Delta=1e-4)
    print(mu_i_star)
    print(sigma_i_sqr)

    # ------------------------------------------------------
    #              Sobol Indices
    # ------------------------------------------------------
    M = 10000
    S_ls, S_T_ls, Y_A = compute_Sobol_indices(theta_nom, M)
    print(S_ls)
    print(np.sum(np.array(S_ls), axis=0))
    print(S_T_ls)

    # ------------------------------------------------------
    #               Plot KDE of Ys
    # ------------------------------------------------------
    fig, ax = plt.subplots()
    ax.hist(Y_A[0], bins=50, density=True, alpha=0.5, label='Y_A')
    ax.hist(Y_A[1], bins=50, density=True, alpha=0.5, label='Y_A_np')
    ax.set_xlabel('Y')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------
    #              Analytical Sobol Indices
    # ------------------------------------------------------
    S_i_an = compute_analytical_Sobol_indices(c_coeffs=c_coeffs, sigma_sqr_coeffs=sigma_sqr_coeffs)
    print(S_i_an)
