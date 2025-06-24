import numpy as np
from scipy.integrate import odeint
from joblib import Parallel, delayed
import pandas as pd

# Load and sort the topology
try:
    topo = np.loadtxt('four_node_team.txt', skiprows=1) # Assuming 1 header line, adjust if needed
except FileNotFoundError:
    print("Error: 'four_node_team.txt' not found. Please ensure it's in the correct directory.")
    exit()

topo = topo[np.argsort(topo[:, 1])]
num_nodes = int(np.max(topo[:, 1]))

# Construct node_neigh
node_neigh = [[] for _ in range(num_nodes)]
for node in range(1, num_nodes + 1):
    indices = np.where(topo[:, 1] == node)[0]
    for i in indices:
        regulator_id = topo[i, 0]
        # Find regulators of the current direct regulator
        neigh_ind = np.where(topo[:, 1] == regulator_id)[0]
        if neigh_ind.size > 0:
            node_neigh[node - 1].append(topo[neigh_ind])

# Parameter sampling
num_para_set = 10

g = 1 + (100 - 1) * np.random.rand(num_para_set)
k = 1 / (1 + (100 - 1) * np.random.rand(num_para_set))
n = np.random.randint(1, 7, size=num_para_set)
lambda_pos = 1 + 99 * np.random.rand(num_para_set)
lambda_neg = 1 / lambda_pos
para_sets = np.zeros((5 + topo.shape[0], num_para_set))
para_sets[0:5, :] = np.array([g, k, n, lambda_pos, lambda_neg])

link_counter = 0
for node in range(num_nodes):
    med = np.median(g / k)
    th = 0.02 * med + 1.96 * med * np.random.rand(num_para_set)
    
    # Loop through the direct regulators of the current node
    for i in range(len(node_neigh[node])):
        expr = g / k
        
        # Loop through the regulators of the regulator
        for j in range(node_neigh[node][i].shape[0]):
            link = node_neigh[node][i][j]
            if link[2] == 1:
                expr *= (1 - (1 - lambda_pos) * ((g / k) ** n / ((g / k) ** n + th ** n))) / lambda_pos
            else:
                expr *= (1 - (1 - lambda_neg) * ((g / k) ** n / ((g / k) ** n + th ** n)))
        
        med_1 = np.median(expr)
        
        # Use the simple counter for indexing. The index is 5 (for global params) + the counter value.
        # This will iterate from 5 to 5 + (total_links - 1), which is always within bounds.
        para_sets[5 + link_counter, :] = 0.02 * med_1 + 1.96 * med_1 * np.random.rand(num_para_set)
        
        # Increment the counter for the next link
        link_counter += 1

# --- START OF FIX: Add this block to define init_cond ---
print("Generating random initial conditions...")
num_init_cond = 1000
tol = 10 # Tolerance for clustering steady states
init_cond = (1 + (100 - 1) * np.random.rand(num_nodes, num_init_cond)) * (1 + (100 - 1) * np.random.rand(num_nodes, num_init_cond))
# --- END OF FIX ---

# Define the ODE model
# --- START OF CORRECTION 2 ---
def ode_model(y, t, topo, num_nodes, node_neigh, para):
    dy = np.zeros(num_nodes)
    link_counter = 0 # Use the same robust counter logic here
    
    for node in range(num_nodes): # node index is 0 to num_nodes-1
        dy[node] = 1.0
        # Find all regulatory links targeting the current node (node + 1)
        node_regulators_indices = np.where(topo[:, 1] == node + 1)[0]
        
        for idx in range(len(node_regulators_indices)):
            reg_link_topo_row = topo[node_regulators_indices[idx]]
            reg_node_id = int(reg_link_topo_row[0])
            reg_type = int(reg_link_topo_row[2])
            
            y_reg = y[reg_node_id - 1] # -1 to convert from 1-based to 0-based index
            
            # Access the correct threshold using the simple counter
            thr = para[5 + link_counter]
            p_n = para[2] # Hill coefficient
            
            if reg_type == 1: # Activation
                dy[node] *= (1 - (1 - para[3]) * (y_reg ** p_n / (y_reg ** p_n + thr ** p_n))) / para[3]
            else: # Inhibition
                dy[node] *= (1 - (1 - para[4]) * (y_reg ** p_n / (y_reg ** p_n + thr ** p_n)))
                
            link_counter += 1 # Increment for each link processed
            
        dy[node] = dy[node] * para[0] - para[1] * y[node]

    return dy
# --- END OF CORRECTION 2 ---

# --- START OF FIXES ---

# 1. Modify simulate_one to RETURN its results instead of trying to modify a shared variable.
def simulate_one(para_indx):
    """Simulates one parameter set and returns the unique steady states found."""
    para = para_sets[:, para_indx]
    t_span = np.linspace(0, 1000, 2) # Only need start and end for odeint's final state
    
    # Nested function for running one initial condition
    def run_single_ic(ic):
        sol = odeint(ode_model, ic, t_span, args=(topo, num_nodes, node_neigh, para))
        return sol[-1]

    # Run simulations in parallel for all initial conditions
    y_end = Parallel(n_jobs=-1)(delayed(run_single_ic)(init_cond[:, i]) for i in range(init_cond.shape[1]))
    y_end = np.array(y_end)

    # Find unique steady states using rounding and a tolerance
    if y_end.shape[0] == 0:
        return (0, np.array([]), para_indx) # Return empty if simulation fails

    y_unique_temp = np.unique(np.round(y_end), axis=0)

    if y_unique_temp.shape[0] <= 1:
        y_unique = y_unique_temp
    else:
        mask = np.ones(len(y_unique_temp), dtype=bool)
        for i in range(len(y_unique_temp)):
            if mask[i]:
                for j in range(i + 1, len(y_unique_temp)):
                    if np.sum(np.abs(y_unique_temp[i] - y_unique_temp[j])) < tol:
                        mask[j] = False # Mark the duplicate for removal
        y_unique = y_unique_temp[mask]
    
    # Return a tuple containing the number of states, the states themselves, and the parameter index
    return (len(y_unique), y_unique, para_indx)

# --- MAIN SIMULATION BLOCK ---

# 2. Call the parallel processes and COLLECT the results into a list.
print("Starting parallel simulations for all parameter sets...")
list_of_results = Parallel(n_jobs=-1)(delayed(simulate_one)(i) for i in range(num_para_set))
print("...Simulations finished.")

# 3. Process the collected results in a standard (serial) loop.
y_steady = {i: [] for i in range(1, 11)}
for num_states, states, p_idx in list_of_results:
    if 0 < num_states <= 10:
        # Append each state vector along with its parameter set ID
        for state_vector in states:
            y_steady[num_states].append(np.concatenate(([p_idx + 1], state_vector)))

# Convert lists to numpy arrays for easier viewing
for k in y_steady:
    y_steady[k] = np.array(y_steady[k])
    
# --- END OF FIXES ---

# Now you can inspect the final y_steady dictionary
print("\n--- Results Summary ---")
for num_states, results_matrix in y_steady.items():
    if results_matrix.shape[0] > 0:
        num_sets = len(np.unique(results_matrix[:, 0]))
        print(f"Found {num_sets} parameter sets with {num_states} stable state(s).")

# Example: Print the results for bistable systems
if len(y_steady.get(2, [])) > 0:
    print("\nBistable Results (Parameter Set ID, States...):")
    print(pd.DataFrame(y_steady[2]))