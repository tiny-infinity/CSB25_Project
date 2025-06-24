import pandas as pd
from tqdm import tqdm # A library for beautiful progress bars
import matplotlib.pyplot as plt
# Import your main analysis function from the script we've been working on
# Make sure to replace 'your_main_script_name' with the actual name of your file
from DRL_Analysis import run_drl_analysis, parameter_set

def batch_analyze_bistable(network_name,adj_matrix):
    """
    Finds and analyzes all parameter sets that result in exactly 2 stable states.
    """
    # 1. Load the full parameter data
    all_params_df = parameter_set(network_name) # Or your network name

    # 2. Find all parameter sets that have exactly 2 stable states
    # IMPORTANT: Verify the column name 'Number_of_steady_states' matches your data file.
    try:
        filtered_df = all_params_df[all_params_df['No.of SS'] == 2]
    except KeyError:
        print("FATAL ERROR: Column 'Number_of_steady_states' not found in your parameter data.")
        print("Please check the column name in your data file and update the script.")
        return None

    # 3. Get the list of parameter set IDs from the filtered DataFrame
    all_param_ids = filtered_df['PS.No'].unique()[:5]

    if len(all_param_ids) == 0:
        print("No parameter sets with exactly 2 stable states were found.")
        return None

    print(f"Found {len(all_param_ids)} parameter sets with 2 stable states. Starting analysis...")

    # 4. Loop through each ID, run the analysis, and collect results
    results_list = []
    #adj_matrix = [[1, -1], [-1, 1]] # Change if your model is different

    for param_id in tqdm(all_param_ids, desc="Analyzing Bistable Sets"):
        try:
            result = run_drl_analysis(
                network_name=network_name,
                adjacency_matrix=adj_matrix,
                param_id=param_id,
                d_coefficient=0.1,
                visualize=False # Keep False for batch processing
            )
            if result:
                results_list.append({
                    'param_id': param_id,
                    'action_A_to_B': result['actions']['A_to_B'],
                    'action_B_to_A': result['actions']['B_to_A'],
                    'barrier_A_to_B': result['barrier_heights']['A_to_B'],
                    'barrier_B_to_A': result['barrier_heights']['B_to_A']
                })
        except Exception as e:
            print(f"---! Error analyzing Param ID {param_id}: {e}. Skipping. !---")
            continue

    # 5. Convert results to a DataFrame and save
    if results_list:
        summary_df = pd.DataFrame(results_list)
        summary_df.to_csv('analysis_summary_bistable.csv', index=False)
        print("\nBatch analysis complete. Results saved to 'analysis_summary_bistable.csv'")
        return summary_df
    else:
        print("\nAnalysis finished with no valid bistable results to save.")
        return None


def plot_action_vs_barrier(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['barrier_A_to_B'], df['barrier_B_to_A'], alpha=0.5, label='Transition A -> B')
    #plt.scatter(, df['action_B_to_A'], alpha=0.5, label='Transition B -> A')
    
    # Use log scale for the y-axis to handle the large range of action values
    plt.yscale('log')
    
    plt.title('Transition Action vs. Potential Barrier Height', fontsize=16)
    plt.xlabel('Potential Barrier Height (U)', fontsize=12)
    plt.ylabel('Transition Action (Log Scale)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

adj_matrix_4=[[0,1,-1,-1],
              [1,0,-1,-1],
              [-1,-1,0,1],
              [-1,-1,1,0]]
if __name__ == '__main__':
    summary_df = batch_analyze_bistable("four_node_team",adj_matrix_4)
    if summary_df is not None:
        print("\n--- Summary of Bistable Systems ---")
        print(summary_df.head())
    plot_action_vs_barrier(summary_df)