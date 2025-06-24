import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve_continuous_lyapunov
from scipy.stats import multivariate_normal
from sympy import symbols, Matrix, lambdify, pprint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from sympy import *
from polisher import *

try:
    from make_sense_of_RACIPE import steady_states, parameter_set
except ImportError:
    print("FATAL ERROR: Could not import 'steady_states' and 'parameter_set'.")
    print("Please ensure the 'make_sense_of_RACIPE' module is in your Python path.")
    exit()

#----------------------Building the Landscape--------------------------------------------------------------

def _generate_steady_state_vectors(network, param_id, node_list):
    """
    For a given parameter ID (PS.No), the function returns a list of NumPy arrays of the steady state
    vectors (inverse log transformed)
    """
    full_df = steady_states(network, node_list)
    corr_SS = full_df.loc[full_df['PS.No'] == param_id]
    if corr_SS.empty: return [], [], None
    param_SS_vectors, freq_SS = [], []
    for _, row in corr_SS.iterrows():
        try:
            log_values = [float(row[var]) for var in node_list]
            state_vector = 2**np.array(log_values, dtype=np.float64)
            param_SS_vectors.append(state_vector)
            freq_SS.append(float(row['Freq']) / 10000.0)
        except (ValueError, TypeError) as e:
            print(f"!!! WARNING: Could not parse row. Error: {e}")
            continue
    print("SS Vectors found:", [f"State {i+1}: {vec[:2].round(2)}..." for i, vec in enumerate(param_SS_vectors)])
    return param_SS_vectors, freq_SS, corr_SS

def _negative_regulation(in_var, tld, hill, fc_par):
    """
    Model for negative regulation as used in RACIPE
    """
    K_n = tld**hill
    x_n = in_var**hill
    regulation_strength = K_n / (x_n + K_n)
    return fc_par + (1.0 - fc_par) * regulation_strength

def _positive_regulation(in_var, tld, hill, fc_par):
    """
    Model for positive regulation
    """
    K_n = tld**hill
    x_n = in_var**hill
    numerator = fc_par + (1.0 - fc_par) * (K_n / (x_n + K_n))
    return numerator / fc_par

def _generate_odes(adjacency_matrix, gene_symbols, params_row):
    """
    Takes as input the adjacency_matrix, symbolics for the nodes and the row of the dataframe
    containing the parameter set of interest
    Returns a list of symbolic ODEs for the system of interest
    """
    odes = []
    for i in range(len(gene_symbols)):
        regulated_gene = gene_symbols[i]
        production_term = float(params_row.get(f'Prod_of_{regulated_gene.name}', 50.0))
        k_deg = float(params_row.get(f'Deg_of_{regulated_gene.name}', 1.0))
        degradation_term = k_deg * regulated_gene
        regulation_product = 1
        for j in range(len(gene_symbols)):
            regulation_type = adjacency_matrix[i][j]
            if regulation_type != 0:
                regulator_gene = gene_symbols[j]
                param_prefix = f'of_{regulator_gene.name}To{regulated_gene.name}'
                s = float(params_row.get(f'Trd_{param_prefix}', 16.0))
                n = float(params_row.get(f'Num_{param_prefix}', 4.0))
                if regulation_type > 0:
                    l = float(params_row.get(f'Act_{param_prefix}', 10.0))
                    regulation_product *= _positive_regulation(regulator_gene, s, n, l)
                elif regulation_type < 0:
                    l = float(params_row.get(f'Inh_{param_prefix}', 0.1))
                    regulation_product *= _negative_regulation(regulator_gene, s, n, l)
        ode = production_term * regulation_product - degradation_term
        odes.append(ode)
        pprint(ode)
    return odes

def _calculate_jacobian(system_odes, variables, steady_state_vector):
    """
    Calculates the Jacobian matrix at a given steady state vector.
    This version assumes the input vector is a NumPy array.
    """
    # Create the substitution dictionary from the numpy array and symbols
    sub_dict = {var: steady_state_vector[i] for i, var in enumerate(variables)}
    
    gen_matrix = Matrix(system_odes)
    jac_matrix = gen_matrix.jacobian(variables)
    
    return np.array(jac_matrix.subs(sub_dict)).astype(np.float64)

def _solve_lyapunov(jacobian_matrix, d=0.1):
    """Solves the continous lyapunov equation for a pre-calculated Jacobian matrix
    Checks for conditions:
        - Jacobian should be symetric, postive and semi-definite
    Returns a numpy matrix
    """
    Q = -2 * d * np.eye(len(jacobian_matrix))
    try:
        return solve_continuous_lyapunov(jacobian_matrix, Q)
    except np.linalg.LinAlgError:
        print("Warning: Lyapunov equation could not be solved.")
        return np.zeros_like(jacobian_matrix)
    
def _global_covariance_matrix(steady_states_vectors, ss_df, freq_ss, dimensions, system_odes, gene_symbols, d_value):
    """
    Takes in:
        - list of steady state vectors
        - the dataframe of the steady_states
        - the list of frequencies of steady states
        - the dimensions of the system
        - the list of symbolic variables for the nodes
        - the value of noise to consider while solving the Lyapunov equation
    Returns the global covariance matrix for a given network+parameter set, which can then be used
    to do PCA.
    """
    
    local_covariances_list, stable_ss_vectors, stable_ss_freqs = [], [], []
    for i, (index, ss_row_log) in enumerate(ss_df.iterrows()):
        print(f"\n--- Analyzing State {i+1} (Index {index}) ---")
        ss_row_linear = ss_row_log.copy()
        for gene_sym in gene_symbols:
            if gene_sym.name in ss_row_linear:
                ss_row_linear[gene_sym.name] = 2**ss_row_linear[gene_sym.name]
        local_jacobian = _calculate_jacobian(system_odes, gene_symbols, ss_row_linear)
        eigenvalues, _ = np.linalg.eig(local_jacobian)
        print("Real parts of Jacobian eigenvalues:", np.real(eigenvalues))
        if np.all(np.real(eigenvalues) < -1e-9):
            print(">>> State is STABLE. Including in analysis.")
            stable_ss_vectors.append(steady_states_vectors[i])
            stable_ss_freqs.append(freq_ss[i])
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

def _pca_analysis(global_cov_matrix):
    """
    Takes in the global covariance matrix and returns as output the first two 
    principal components along the with the list of explained variance %ages.
    """
    eigevals, eigenvectors = eigh(global_cov_matrix)
    sorted_indices = np.argsort(eigevals)[::-1]
    PC1 = eigenvectors[:, sorted_indices[0]]
    PC2 = eigenvectors[:, sorted_indices[1]]
    explained_variance_ratio = eigevals[sorted_indices] / np.sum(eigevals)
    return PC1, PC2, explained_variance_ratio

def _project_vectors(vectors, transformation_matrix):
    """
    Projects the steady vectors on the PC plane
    """
    return [transformation_matrix.T @ vec for vec in vectors]

from scipy.optimize import root
from sympy import symbols

def _generate_drift_function(system_odes, gene_symbols):
    """
    Creates a numerical function from the symbolic ODEs for use in numerical solvers
    that expect a function with the signature f(y).
    """
    # 1. Ensure gene_symbols is a tuple to guarantee how lambdify behaves.
    # This makes f_numeric always expect separate arguments: f(A, B, ...)
    symbols_as_tuple = tuple(gene_symbols)
    
    f_numeric = lambdify(symbols_as_tuple, system_odes, modules="numpy")

    # 2. Use the splat operator (*) to unpack the vector 'x' into separate arguments
    # This matches the expected f(A, B, ...) signature.
    return lambda x: np.array(f_numeric(*x), dtype=np.float64)

# --- 1. Define the sample system ---
node_names = ['A', 'B', 'C','D']
sample_gene_symbols = symbols(node_names)
sample_param_set = parameter_set("four_node_team")[parameter_set('four_node_team')["PS.No"] == 123].iloc[0]
sample_odes = _generate_odes([[0,1,-1, -1],[1,0,-1,-1],[-1,-1,0,1],[-1,-1,1,0]], sample_gene_symbols, sample_param_set)

# --- 2. Generate the drift function for the root-finder ---
drift_function = _generate_drift_function(sample_odes, sample_gene_symbols)

# --- 3. Load the imprecise steady states from RACIPE data ---
sample_ss, _, _ = _generate_steady_state_vectors("four_node_team", 123, node_names)

polished_steady_states, report = polish_steady_states(sample_ss,[[0,1,-1, -1],[1,0,-1,-1],[-1,-1,0,1],[-1,-1,1,0]],sample_param_set,node_names)

print(polished_steady_states)



