# -*- coding: utf-8 -*-
"""
Full analysis script for Dynamical Reaction Landscape (DRL) and
Minimum Action Path (MAP) calculation from RACIPE output.

This version is a complete and stable implementation, faithfully reproducing
the logic of the reference MATLAB code and foundational papers. It includes
a symmetry-breaking initial path to ensure robust convergence.
"""
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve_continuous_lyapunov
from scipy.stats import multivariate_normal
from sympy import symbols, Matrix, lambdify, pprint
from scipy.optimize import minimize, root, Bounds
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
import action_solver
from define_system import *
from visualizer import *
import define_system
import covariance_module as gcov

# --- Actual Data Loading ---
try:
    from make_sense_of_RACIPE import steady_states, parameter_set
    from polisher import polish_racipe_dataframe
except ImportError as e:
    print(f"FATAL ERROR: Could not import a required module: {e}")
    print("Please ensure 'make_sense_of_RACIPE.py' and 'state_polisher.py' are in your Python path.")
    exit()

def _pca_analysis(global_cov_matrix):
    eigevals, eigenvectors = eigh(global_cov_matrix)
    sorted_indices = np.argsort(eigevals)[::-1]
    PC1 = eigenvectors[:, sorted_indices[0]]
    PC2 = eigenvectors[:, sorted_indices[1]]
    explained_variance_ratio = eigevals[sorted_indices] / np.sum(eigevals)
    return PC1, PC2, explained_variance_ratio

def _project_vectors(vectors, transformation_matrix):
    return [transformation_matrix.T @ vec for vec in vectors]


def run_drl_analysis(network_name, adjacency_matrix, param_id, d_coefficient=0.1, grid_resolution=2000, padding_factor=1.2, 
                     visualize=False, node_list=['A','B'],
                     # --- PARAMETERS ALIGNED WITH MATLAB & ROBUSTNESS ---
                     N_points=60,     # MATLAB: params.N = 60
                     T_max=30.0,       # MATLAB: params.TMax = 1
                     k_steps=50,      # Increased for more controlled relaxation
                     c_param=1e18):   # MATLAB: params.c = 1e18
    
    print(f"\n--- Starting DRL/AMAM Analysis for Network: '{network_name}', Param ID: {param_id} ---")
    
    # 1. Load data from RACIPE
    num_dims = len(adjacency_matrix)
    
    full_ss_df = steady_states(network_name, node_list)
    racipe_df_subset = full_ss_df.loc[full_ss_df['PS.No'] == param_id]
    
    if racipe_df_subset.empty: return None
        
    network_params = parameter_set(network_name).loc[parameter_set(network_name)['PS.No'] == param_id].iloc[0]
    
    # 2. Polish steady states
    polished_df, _ = polish_racipe_dataframe(racipe_df_subset, adjacency_matrix, network_params, node_list)
    
    # 3. Prepare polished data for analysis
    polished_ss_vectors = [2**np.array(row[node_list].values, dtype=np.float64) for _, row in polished_df.iterrows()]
    polished_freqs = polished_df['Freq'].values / 10000.0
    
    # 4. Generate symbolic ODEs
    gene_symbols = symbols(node_list)
    system_odes = define_system._generate_odes(adjacency_matrix, gene_symbols, network_params)
    for ode in system_odes:
        pprint(ode)
    
    # 5. Calculate global covariance using polished states
    sigma_global, local_sigmas, stable_vectors, stable_freqs = gcov._global_covariance_matrix(
        polished_ss_vectors, polished_freqs, num_dims, system_odes, gene_symbols, d_value=d_coefficient
    )
    
    if sigma_global is None: return None
    print(f"\nFound {len(stable_vectors)} stable states. Proceeding...")
    
    # 6. PCA and visualization setup
    pc1, pc2, explained_variance = _pca_analysis(sigma_global)
    print(f"PCA complete. PC1 explains {explained_variance[0]:.2%}, PC2 explains {explained_variance[1]:.2%}.")
    print("PC1:", pc1)
    print("PC2:", pc2)
    V = np.column_stack((pc1, pc2))

    log_stable_vectors = [np.log2(np.maximum(vec, 1e-12)) for vec in stable_vectors]
    projected_ss = _project_vectors(log_stable_vectors, V)
    print("Projected Steady States: ", projected_ss)

    # --- 1. DEFINE LANDSCAPE BOUNDARIES ---
    proj_array = np.array(projected_ss)
    center = proj_array.mean(axis=0)
    max_dist = np.max(np.linalg.norm(proj_array - center, axis=1)) if proj_array.size > 0 else 1.0
    grid_width = max_dist * 2 * padding_factor
    
    x_min, x_max = center[0] - grid_width, center[0] + grid_width
    y_min, y_max = center[1] - grid_width, center[1] + grid_width
    
    # --- 2. CONSTRUCT THE ENERGY LANDSCAPE U(x) ---
    X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_resolution), np.linspace(y_min, y_max, grid_resolution))
    pos = np.dstack((X_grid, Y_grid))
    
    P = np.zeros_like(X_grid)
    for i in range(len(projected_ss)):
        proj_cov = V.T @ local_sigmas[i] @ V
        P += stable_freqs[i] * multivariate_normal(mean=projected_ss[i], cov=proj_cov).pdf(pos)
        
    U = -np.log(np.maximum(P, 1e-100))
    print("Potential energy landscape constructed.")

    if len(stable_vectors) < 2:
        print("Fewer than two stable states found. Cannot calculate transition paths.")
        if visualize:
            # The visualizer is called with no path information.
            visualize_landscape(U, X_grid, Y_grid, projected_ss, None, None, None, None)
        return None

    # If 2 or more states are found, proceed to calculate the path between the first two.
    print(f"\n{len(stable_vectors)} stable states found. Calculating path between the first two.")

    drift_function = define_system._generate_drift_function(system_odes, gene_symbols)
    
    print("\nCalculating path from State A to State B...")
    map_a_to_b, time_mesh_ab = action_solver.aMAM_advanced_solver(
        drift_function, stable_vectors[0], stable_vectors[1], N=N_points, T_max=T_max, k_max=k_steps, c=c_param, initial_path='log'
    )
    print("\nCalculating path from State B to State A...")
    map_b_to_a, time_mesh_ba = action_solver.aMAM_advanced_solver(
        drift_function, stable_vectors[1], stable_vectors[0], N=N_points, T_max=T_max, k_max=k_steps, c=c_param, initial_path='log'
    )

    
    plot_path_diagnostics(map_a_to_b, time_mesh_ab, map_b_to_a, time_mesh_ba,var_names=node_list)
    
    action_a_to_b = action_solver._action_cost(map_a_to_b, time_mesh_ab, drift_function)
    action_b_to_a = action_solver._action_cost(map_b_to_a, time_mesh_ba, drift_function)
    
    proj_path_ab = _project_vectors([np.log2(np.maximum(p, 1e-12)) for p in map_a_to_b], V)
    proj_path_ba = _project_vectors([np.log2(np.maximum(p, 1e-12)) for p in map_b_to_a], V)
    
    path_grid_indices_ab = (np.array([(np.abs(Y_grid[:, 0] - p[1])).argmin() for p in proj_path_ab]), np.array([(np.abs(X_grid[0, :] - p[0])).argmin() for p in proj_path_ab]))
    path_U_values_ab = U[path_grid_indices_ab]
    path_grid_indices_ba = (np.array([(np.abs(Y_grid[:, 0] - p[1])).argmin() for p in proj_path_ba]), np.array([(np.abs(X_grid[0, :] - p[0])).argmin() for p in proj_path_ba]))
    path_U_values_ba = U[path_grid_indices_ba]

    saddle_idx_ab, U_saddle_ab = np.argmax(path_U_values_ab), np.max(path_U_values_ab)
    saddle_coords_ab = proj_path_ab[saddle_idx_ab]
    saddle_idx_ba, U_saddle_ba = np.argmax(path_U_values_ba), np.max(path_U_values_ba)
    saddle_coords_ba = proj_path_ba[saddle_idx_ba]
    U_stable_A = path_U_values_ab[0]
    U_stable_B = path_U_values_ba[0] 
    barrier_height_ab = U_saddle_ab - U_stable_A
    barrier_height_ba = U_saddle_ba - U_stable_B
    print(f"\nBarrier Height A -> B: {barrier_height_ab:.4f}")
    print(f"Barrier Height B -> A: {barrier_height_ba:.4f}")
    
    if visualize:
        visualize_landscape(U, X_grid, Y_grid, projected_ss, saddle_coords_ab, proj_path_ab, saddle_coords_ba, proj_path_ba)
    
    results_dict = {
        "actions": {"A_to_B": action_a_to_b, "B_to_A": action_b_to_a},
        "barrier_heights": {"A_to_B": barrier_height_ab, "B_to_A": barrier_height_ba}
    }
    
    return results_dict

if  __name__ == '__main__':
    try:
        adj_matrix=[[0,1,-1,-1],
                    [1,0,-1,-1],
                    [-1,-1,0,1],
                    [-1,-1,1,0]]
        adj_matrix=[[1,-1],[-1,1]]
        param_id_to_test = 145

        results = run_drl_analysis(
            network_name="MISA", 
            adjacency_matrix=adj_matrix, 
            param_id=param_id_to_test, 
            d_coefficient=0.1, 
            visualize=True
        )
        
        if results:
            print("\n\n" + "="*25 + " FINAL RESULTS " + "="*25)
            print(results)
            
    except Exception as e:
        import traceback
        traceback.print_exc()

from scipy.integrate import solve_ivp,odeint

def simulating_kinetics(T, init_conds):
    all_params = parameter_set("MISA")
    sample_params = all_params.loc[5] 
    gene_vars = symbols(['A', 'B'])
    sample_odes = define_system._generate_odes([[1,-1],[-1,1]], gene_vars, sample_params)
    print("System ODEs:")
    for ode in sample_odes:
        pprint(ode)
    sample_drift_func = define_system._generate_drift_function(sample_odes, gene_vars)
    def ode_system(t, y):
        return sample_drift_func(y)
    t_span = (0, T)
    t_eval = np.linspace(0, T, 1000)
    solution = solve_ivp(ode_system, t_span, init_conds, t_eval=t_eval, dense_output=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(solution.y[0], solution.y[1], 'r-', lw=2, label='Trajectory')
    ax[0].plot(solution.y[0, 0], solution.y[1, 0], 'go', ms=10, label='Start')
    ax[0].plot(solution.y[0, -1], solution.y[1, -1], 'ko', ms=10, label='End')
    ax[0].set_title("Phase Portrait with Drift Field")
    ax[0].set_xlabel("Concentration of A")
    ax[0].set_ylabel("Concentration of B")
    x_lims = [0,600]
    y_lims = [0,600]
    X, Y = np.meshgrid(np.linspace(x_lims[0], x_lims[1], 80),
                       np.linspace(y_lims[0], y_lims[1], 80))
    dX = np.zeros_like(X)
    dY = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            derivatives = sample_drift_func([X[i, j], Y[i, j]])
            dX[i, j] = derivatives[0]
            dY[i, j] = derivatives[1]
    mag = np.sqrt(dX**2 + dY**2)
    mag[mag == 0] = 1.0
    unit_x, unit_y = dX / mag, dY / mag
    ax[0].quiver(X, Y, unit_x, unit_y, alpha=0.8, color='blue')
    ax[0].legend()
    ax[0].grid(True, linestyle='--')
    ax[1].plot(solution.t, solution.y[0], label='Concentration of A')
    ax[1].plot(solution.t, solution.y[1], label='Concentration of B')
    ax[1].set_title("Time Series")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Concentration")
    
    ax[1].legend()
    ax[1].grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

