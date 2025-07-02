import numpy as np
import pandas as pd
from sympy import symbols, lambdify,pprint
from scipy.spatial.distance import jensenshannon
from make_sense_of_RACIPE import parameter_set, steady_states
import matplotlib.pyplot as plt
import network_generator
# --- Helper Functions for Rate Generation ---
# (These are the corrected versions from our previous discussion)

def _generate_production_lambdas(adjacency_matrix, gene_symbols, params_row):
    """Generates a list of numerical functions for the production rate of each gene."""
    production_odes = []
    for i in range(len(gene_symbols)):
        regulated_gene = gene_symbols[i]
        production_term = float(params_row.get('g', 50.0))
        regulation_product = 1
        for j in range(len(gene_symbols)):
            regulation_type = adjacency_matrix[i][j]
            if regulation_type != 0:
                regulator_gene = gene_symbols[j]
                param_prefix = f'{regulator_gene.name}to{regulated_gene.name}'
                s = float(params_row.get(f'thrs_{param_prefix}', 16.0))
                n = float(params_row.get(f'n', 4.0))
                if regulation_type > 0:
                    l = float(params_row.get(f'lambda_pos', 10.0))
                    K_n_pos = s**n
                    x_n_pos = regulator_gene**n
                    numerator = l + (1.0 - l) * (K_n_pos / (x_n_pos + K_n_pos))
                    regulation_product *= numerator / l
                elif regulation_type < 0:
                    l = float(params_row.get(f'lambda_neg', 0.1))
                    K_n_neg = s**n
                    x_n_neg = regulator_gene**n
                    reg_strength = K_n_neg / (x_n_neg + K_n_neg)
                    regulation_product *= l + (1.0 - l) * reg_strength
        production_odes.append(production_term * regulation_product)
    production_lambdas = [lambdify(tuple(gene_symbols), ode, 'numpy') for ode in production_odes]
    return production_lambdas

def _generate_degradation_lambdas(gene_symbols, params_row):
    """Generates a list of numerical functions for the degradation rate of each gene."""
    degradation_odes = []
    for i in range(len(gene_symbols)):
        regulated_gene = gene_symbols[i]
        k_deg = float(params_row.get(f'k', 1.0))
        degradation_odes.append(k_deg * regulated_gene)
    degradation_lambdas = [lambdify(tuple(gene_symbols), ode, 'numpy') for ode in degradation_odes]
    return degradation_lambdas

# --- Langevin Simulator ---

def simulate_chemical_langevin(production_funcs, degradation_funcs, initial_state, total_time, dt):
    """Simulates a single trajectory using the Chemical Langevin Equation (CLE)."""
    num_dimensions = len(initial_state)
    n_steps = int(total_time / dt)
    time_points = np.linspace(0, total_time, n_steps + 1)
    trajectory = np.zeros((n_steps + 1, num_dimensions))
    trajectory[0] = initial_state
    current_state = initial_state.copy()
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        np.clip(current_state, 0, None, out=current_state)
        prod_rates = np.array([p_func(*current_state) for p_func in production_funcs])
        deg_rates = np.array([d_func(*current_state) for d_func in degradation_funcs])
        drift = (prod_rates - deg_rates) * dt
        prod_rand = np.random.normal(size=num_dimensions)
        deg_rand = np.random.normal(size=num_dimensions)
        prod_noise = np.sqrt(prod_rates) * prod_rand * sqrt_dt
        deg_noise = np.sqrt(deg_rates) * deg_rand * sqrt_dt
        current_state += drift + prod_noise - deg_noise
        trajectory[i + 1] = current_state
    return time_points, trajectory

# --- New Modular Analysis Function ---

def analyze_trajectory_divergence(
    param_set: pd.Series,
    ss_data: pd.DataFrame,
    node_list: list,
    adjacency_matrix: np.ndarray,
    sim_time: float = 30000.0,
    dt: float = 0.01,
    num_bins: int = 30
) -> float:
    """
    Calculates the Jensen-Shannon Divergence between trajectories from the first two steady states.

    Args:
        param_set: A pandas Series for a single parameter set.
        ss_data: A pandas DataFrame with the steady states for the given param_set.
        node_list: A list of gene names.
        adjacency_matrix: The network topology.
        sim_time: Total simulation time for Langevin dynamics.
        dt: Time step for Langevin dynamics.
        num_bins: The number of bins to use for histogramming each dimension.

    Returns:
        The calculated Jensen-Shannon Divergence (JSD), or np.nan if less than two
        steady states are provided.
    """
    # 1. Input validation
    if len(ss_data) < 2:
        print(f"  Warning: Fewer than 2 steady states found. Cannot calculate JSD.")
        return np.nan

    # 2. Generate the rate functions for the given system
    syms = symbols(node_list)
    prod_functions = _generate_production_lambdas(adjacency_matrix, syms, param_set)
    deg_functions = _generate_degradation_lambdas(syms, param_set)

    # 3. Run Langevin simulations starting from each steady state
    traj_arrays = []
    for i in range(len(ss_data)):
        temp_ss = ss_data.iloc[i]
        start_state = np.array([2**float(temp_ss[node]) for node in node_list], dtype=np.float64)
        _, traj = simulate_chemical_langevin(prod_functions, deg_functions, start_state, sim_time, dt)
        traj_arrays.append(traj)

    # 4. Calculate JSD between the first two trajectories
    # Stack all trajectories to find global min/max for consistent binning
    all_points = np.vstack(traj_arrays)
    bin_min = np.min(all_points, axis=0)
    bin_max = np.max(all_points, axis=0)

    # Define N-dimensional bins based on the range of each dimension
    bins = [np.linspace(bin_min[i], bin_max[i], num_bins) for i in range(len(node_list))]

    # Create histograms for the first two trajectories
    p_counts, _ = np.histogramdd(traj_arrays[0], bins=bins)
    q_counts, _ = np.histogramdd(traj_arrays[1], bins=bins)

    # Normalize to get probability distributions
    epsilon = 1e-10
    P = p_counts.ravel() / (p_counts.sum() + epsilon)
    Q = q_counts.ravel() / (q_counts.sum() + epsilon)

    # Calculate and return the JSD
    js_divergence_sq = jensenshannon(P, Q, base=2)**2
    return js_divergence_sq

sample_params=pd.read_csv("1_four_node_parameters.csv")
params_row=sample_params[sample_params['para_indx']==1]
funcs=_generate_production_lambdas([[1,-1],[-1,1]],symbols(['A','B']),params_row)
pprint(funcs)

if __name__ == '__main__':
    # 1. Define network properties
    network_name = "0_four_node"
    adj_matrix,nodes = network_generator.topo_to_adj("0_four_node.topo")
    NUM_REALISATIONS = 10

    # 2. Load all parameters and steady states once
    all_params = pd.read_csv(f'{network_name}_parameters.csv')
    all_ss = pd.read_csv(f'{network_name}_solution.csv')

    # 3. List to store the results from all runs
    all_results_list = []

    # 4. Outer Loop: Iterate through each parameter set you want to test
    # As an example, we'll test the first 5 unique parameter IDs.
    # To run for all, you can change this to: for param_id in all_params['PS.No'].unique():
    for param_id in all_params['para_indx'].unique()[1:3]:
        print(f"--- Processing Parameter Set ID: {param_id} ---")

        # Get the specific parameter set and its corresponding steady states
        current_params = all_params[all_params['para_indx'] == param_id].iloc[0]
        current_ss = all_ss[all_ss['para_indx'] == param_id]

        # Check if there are enough steady states before starting realisations
        if len(current_ss) < 2:
            print(f"  Skipping Param ID {param_id}: Fewer than 2 steady states found.")
            continue
            
        # 5. Inner Loop: Run multiple realisations for the current parameter set
        for i in range(NUM_REALISATIONS):
            print(f"  ...Running Realisation {i + 1}/{NUM_REALISATIONS}")

            # Call the modular function to get the JSD for this single realisation
            jsd = analyze_trajectory_divergence(
                param_set=current_params,
                ss_data=current_ss,
                node_list=nodes,
                adjacency_matrix=adj_matrix
            )
            
            # Store the result, including the parameter ID and realisation number
            result_record = {
                'param_id': param_id,
                'realisation': i + 1,
                'JSD': jsd
            }
            all_results_list.append(result_record)

    # 6. After all loops are finished, create and save the final DataFrame
    if all_results_list:
        results_df = pd.DataFrame(all_results_list)

        print("\n--- Final JSD Results (All Runs) ---")
        print(results_df.to_string()) # .to_string() prints the whole DataFrame

        # You can also calculate summary statistics for each parameter set
        summary_df = results_df.groupby('param_id')['JSD'].agg(['mean', 'std', 'count']).reset_index()
        print("\n--- Summary Statistics per Parameter ID ---")
        print(summary_df)

        # Save the detailed results to a .csv file
        output_filename = "jsd_multiple_parameter_sets.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"\nDetailed results saved to {output_filename}")
    else:
        print("\nNo results were generated.")