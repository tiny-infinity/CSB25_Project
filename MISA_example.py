import numpy as np
import random as rn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
misa_params = []

with open("MISA.prs", "r") as f:
    next(f)  # skip the first line
    for line in f:
        if line.strip():  # skip empty lines
            misa_params.append(line.strip().split()[0])


header_names = ['Number of SS']
header_names.extend(misa_params)
param_sets = pd.read_csv('MISA_parameters.csv', header=None, names=header_names)

# 2. Define regulation functions (unchanged)
# --- Corrected Helper Functions ---

def positive_regulation(variable, fold_change_lambda, threshold, hill):
    """
    Implements the consistent formula for activation (fold-change > 1).
    H_act = 1 + (lambda - 1) * (A^n / (T^n + A^n))
    """
    variable = max(0, variable)
    # Corrected the denominator from (threshold + variable)**hill
    activation_term = (variable**hill) / (threshold**hill + variable**hill)
    return (1 - ((1-fold_change_lambda) * activation_term))/fold_change_lambda

def negative_regulation(variable, fold_change_lambda, threshold, hill):
    """
    Implements the formula from the paper for repression (fold-change < 1).
    H_rep = lambda + (1 - lambda) * (T^n / (T^n + A^n))
    """
    variable = max(0, variable)
    # Corrected the denominator from (threshold + variable)**hill
    repression_term = (threshold**hill) / (threshold**hill + variable**hill)
    return 1 - ((1 - fold_change_lambda) * repression_term)

# --- Corrected ODE function ---
# Note: The 'Act_of_...' and 'Inh_of_...' parameters now both correspond
# to the 'lambda' (maximum fold change) value for that interaction.

def MISA_ODE_test(t, y):
    A, B = y
    a,b,S=0.5,0.5,0.5
    k=0.1

    dA = (a*(A**4)/(S**4 + A**4))+(b*(S**4)/(S**4 + B**4))-(k*A)
    dB = (a*(B**4)/(S**4 + B**4))+(b*(S**4)/(S**4 + A**4))-(k*B)
    
    return [dA, dB]

def MISA_ODE_general(t, y, params):
    A, B = y
    
    # Calculate the fold-change multiplier from self-activation and mutual-inhibition
    # This assumes a multiplicative model for combining regulatory inputs
    fold_change_A = positive_regulation(A, params['Act_of_AToA'], params['Trd_of_AToA'], params['Num_of_AToA']) * \
                    negative_regulation(B, params['Inh_of_BToA'], params['Trd_of_BToA'], params['Num_of_BToA'])
    
    fold_change_B = positive_regulation(B, params['Act_of_BToB'], params['Trd_of_BToB'], params['Num_of_BToB']) * \
                    negative_regulation(A, params['Inh_of_AToB'], params['Trd_of_AToB'], params['Num_of_AToB'])

    # The ODEs are Basal Production * Fold Change - Degradation
    dA = params['Prod_of_A'] * fold_change_A - params['Deg_of_A'] * A
    dB = params['Prod_of_B'] * fold_change_B - params['Deg_of_B'] * B
    
    return [dA, dB]

#MISA_ODE_general(t=0,y=[0,0], params=param_sets.iloc[0].drop('Number of SS').to_dict())
# --- Simulation Setup ---
# 4. Select a parameter set and convert it to a dictionary
params_series = param_sets.iloc[7].drop('Number of SS')
params_dict = params_series.to_dict()


params_dict = params_series.to_dict()

# 5. Set simulation conditions
initial_conditions = [0,50] # Start with high A and low B
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 5000)

def solve_misa_ode(init_conds,params_dict):
    sol = solve_ivp(MISA_ODE_general, t_span, initial_conditions, t_eval=t_eval,args=(params_dict,))
    print("ODE solved successfully.")
    return sol

def test_misa_ode(init_conds):
    sol = solve_ivp(MISA_ODE_test, t_span, initial_conditions, t_eval=t_eval)
    print("ODE solved successfully.")
    return sol

if __name__ == "__main__":
    # 6. Solve the ODE
    steady_states_count=[]
    for i in range(1000):
        initial_conditions = [rn.uniform(0, 500), rn.uniform(0, 500)]
        sol = test_misa_ode(initial_conditions)
        steady_states_count.append([sol.y[0][-1], sol.y[1][-1]])
        print(sol.y[0][-1], sol.y[1][-1])
        
    #need to remove duplicates and count the frequency of each steady state
    
    
    

        
    
    








