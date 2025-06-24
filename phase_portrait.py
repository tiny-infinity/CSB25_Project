from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from DRL_Analysis import _generate_odes,_generate_drift_function
from make_sense_of_RACIPE import parameter_set
from sympy import symbols

def two_dim_system(X, Y):
    sample_params = parameter_set("MISA")[parameter_set("MISA")['PS.No']==8]
    print(sample_params)

    # Your existing code for generating ODEs
    gene_vars = symbols(['A', 'B'])
    sample_odes = _generate_odes([[1,-1],[-1,1]], gene_vars, sample_params)
    
    # Note: _generate_drift_function returns a single function that takes a vector
    sample_drift_func = _generate_drift_function(sample_odes, gene_vars)
    print(sample_drift_func)
    return dX, dY

def two_dim_ode(state, time):
    X, Y = state
    res_X, res_Y = two_dim_system(X, Y)
    return [res_X, res_Y]

# Initial setup
time_range = np.linspace(0, 100, 10000)
init_conds = [10,4]

# Solve ODE
soln = odeint(two_dim_ode, init_conds, time_range)

# Dynamically adjust mesh grid based on trajectory extents
x_extent = max(abs(np.min(soln[:, 0])), abs(np.max(soln[:, 0]))) + 1  # Add padding
y_extent = max(abs(np.min(soln[:, 1])), abs(np.max(soln[:, 1]))) + 1

X, Y = np.meshgrid(np.linspace(-x_extent, x_extent, 50),
                   np.linspace(-y_extent, y_extent, 50))

# Recalculate the vector field on the new grid
dX, dY = two_dim_system(X, Y)
magnitude = np.sqrt(dX**2 + dY**2)
dX_norm, dY_norm = dX / magnitude, dY / magnitude

fig, ax = plt.subplots(1,2,figsize=(12,6))
# Plot results

ax[0].plot(soln[:, 0], soln[:, 1], label="Trajectory",color="blue")
ax[0].scatter(init_conds[0], init_conds[1], marker='o', c='r', label="Initial Condition")
ax[0].quiver(X, Y, dX_norm, dY_norm, alpha=0.6)
ax[0].grid(True)
ax[0].set_title("2D Flow Visualizer")
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].legend()
ax[0].hlines(0, -x_extent, x_extent, linestyles='solid', color="black")
ax[0].vlines(0, -y_extent, y_extent, linestyles='solid', color="black")

#plotting energy surface diagrams

ax[1].plot(time_range, soln[:, 0], label="X",color="blue")
ax[1].plot(time_range, soln[:, 1], label="Y",color="red",linewidth=0.5)
ax[1].set_xlabel("Time")
fig.tight_layout()
plt.legend()
plt.show()

