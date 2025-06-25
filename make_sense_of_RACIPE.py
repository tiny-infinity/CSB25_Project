from RACIPE_parser import convert_dat_to_csv, batch_process_racipe_directory
import pandas as pd
batch_process_racipe_directory("/Users/govindrnair/CSB_EnergyLandscape/RACIPE_data",'/Users/govindrnair/CSB_EnergyLandscape/RACIPE_data')

"""
RACIPE generates outputs in the form of .dat files, which have parameters/steady state values in different columns.
This module is to generate cleaner, readable forms of this info that can be used by the rest of the program.

RACIPE outputs (useful to us):
- network.prs : Lists all the parameters involved with their maximum and minimum value and the type of regulation they represent
- network_parameters.dat : Lists each parameter set with number of steady states, followed by the values of each parameter
                           The order in which the parameters appear are the same as in the .prs file.
- network_solution.dat : Lists parameter sets along with steady states. For multistable systems, each set is listed twice, with values at
                         each steady state along with their frequncies (number of inital conditions that converge to the given steady state)
NOTE : None of these files have columns named. Have to make that happen in our module.

What we need:
- Parameter Sets : A dataframe containing all the parameter sets with the columns named
- Steady States : A dataframe containing all the steady states with the serial number conserved from the first dataframe. 
"""

def parameter_set(network_name):
    """
    Args:
        network_name(str): The name of the network as input into RACIPE
    Output:
        parameter_set(dataframe): Parameter set with appropriate column names
    """
    df=pd.read_csv(f"{network_name}.csv")
    columns=['PS.No','No.of SS']+list(df['Parameter'])
    parameter_set=pd.read_csv(f"{network_name}_parameters.csv",header=None,names=columns)
    return parameter_set

def steady_states(network_name,node_list):
    """
    Args:
        network_name(str): The name of the network as input into RACIPE
        node_list(list): List of nodes
    Output:
        steady_states(dataframe): Steady states with frequency and values of variables (in order of appearance in .topo file)
                                  at each steady state. PS No. conserved from parameter_set
    """
    variables=node_list #works only for MISA have to find a fix around this to generalize it
    columns=['PS.No','SS','Freq']+variables
    steady_states=pd.read_csv(f"{network_name}_solution.csv",header=None,names=columns)
    del steady_states['SS']
    return steady_states










    