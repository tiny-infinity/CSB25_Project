from sympy import *
from scipy.linalg import solve_continuous_lyapunov
import numpy as np

def _calculate_jacobian(system_odes, variables, steady_state_vector):
    sub_dict = {var: steady_state_vector[i] for i, var in enumerate(variables)}
    gen_matrix = Matrix(system_odes)
    print(gen_matrix.subs(sub_dict))
    jac_matrix = gen_matrix.jacobian(variables)
    return np.array(jac_matrix.subs(sub_dict)).astype(np.float64)

def _solve_lyapunov(jacobian_matrix, d=0.1):
    Q = -2 * d * np.eye(len(jacobian_matrix))
    try:
        return solve_continuous_lyapunov(jacobian_matrix, Q)
    except np.linalg.LinAlgError:
        return np.zeros_like(jacobian_matrix)

def _global_covariance_matrix(ss_vectors_linear, ss_freqs, dimensions, system_odes, gene_symbols, d_value):
    local_covariances_list, stable_ss_vectors, stable_ss_freqs = [], [], []
    for i, vector in enumerate(ss_vectors_linear):
        print(f"\n--- Analyzing Polished State {i+1} ---")
        local_jacobian = _calculate_jacobian(system_odes, gene_symbols, vector)
        eigenvalues, _ = np.linalg.eig(local_jacobian)
        print("Eigenvalues: ", eigenvalues)
        print("Real parts of Jacobian eigenvalues:", np.real(eigenvalues))

        if np.all(np.real(eigenvalues) < -1e-9):
            print(">>> State is STABLE. Including in analysis.")
            stable_ss_vectors.append(vector)
            stable_ss_freqs.append(ss_freqs[i])
            local_covariances_list.append(_solve_lyapunov(local_jacobian, d=d_value))
        else:
            print(f">>> Warning: State is UNSTABLE. Discarding.")
            
    if len(stable_ss_vectors) < 2: return None, None, None, None

    stable_ss_freqs = np.array(stable_ss_freqs) / np.sum(stable_ss_freqs)
    mu_global = np.sum(stable_ss_freqs[:, np.newaxis] * stable_ss_vectors, axis=0)
    sigma_global = np.zeros((dimensions, dimensions))
    for i in range(len(stable_ss_vectors)):
        mean_diff = stable_ss_vectors[i] - mu_global
        sigma_global += stable_ss_freqs[i] * (local_covariances_list[i] + np.outer(mean_diff, mean_diff))
    return sigma_global, local_covariances_list, stable_ss_vectors, stable_ss_freqs