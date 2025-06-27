# Project : Analysing transitions between phenotypic states of cellular networks
This is the repository for my summer project at the Cancer Systems Biology laboratory @ IISc. The broad objective is to analyse transitions between states in various multi-stable systems.

Updates and other details about the project can be found [here](https://docs.google.com/presentation/d/1Wb02X7PI-c182syrV49iUH8WHvz6C3KffE5MbGOBU-U/edit?slide=id.g35cf70c6234_0_0#slide=id.g35cf70c6234_0_0).

# Making sense of RACIPE data

RACIPE generates its results in a less-than-ideally formatted way (.dat files with no column headings etc.). These modules are to generate more neatly formatted, more readable results out of RACIPE outputs.

Functions:
- ```parameter_set``` : Generates a dataframe containing all the parameter sets with the columns named. The order of the parameters is taken from the .prs file that RACIPE provides as output.
- ```steady_states``` : Generates a dataframe containing all the steady states with the serial number conserved from the first dataframe.

```RACIPE_parser``` is a module that converts .dat files to .csv files to make it easier for Python to read it.

# Polishing RACIPE-generated steady states

During our analysis we encountered some instabilies while calculating the MAP. While trying to debug this, it was observed that plugging in the steady states provided by RACIPE into the rate equations do not always give values close enough to zero ('close' here being in the order of ```1e-9``` or lesser).

To fix this, a function to "polish" the steady states has been implemented. The RACIPE steady states are taken as initial conditions and the ODEs of the system are solved for a longer time to get more accurate steady states.

Implementing this fixed the issue of the kinetics not going to zero at steady state. Unfortunately, this bug was only one of many.

# Calculating the Minimum Action Path

We implement the adaptive Minimum Action Method (aMAM) devised by Zhou, Ren and E here.

```_action_S``` is a function to calculate the action using the approximation mentioned in the paper, based on the Friedlin-Wentzell action functional.

```_monitor_function``` is measure of the "velocity" of the system. A higher value of the function is indicative of a region where the system is changing rapidly.

```_remesh_path_and_time``` performs the adaptive remeshing that iteratively changes the allocation of points on the time mesh so as to capture the systems' dynamics accurately. Details about how the remeshing is done can be found in the paper.

```aMAM_advanced_solver``` puts the functions together to calculate the path of minimum action. This is done by iterating over the mesh until a value of q below the minimum threshold is achieved.

**Optimization Parameters for the solver**


1.   ```k_max``` - the (maximum) number of remeshing steps to perform. If a q lesser than ```qmin``` is achieved earlier, the loop ends.
2.   ```N``` - the number of points on the path.
3. ```c``` - Remeshing strength : A measure of how sensitive the monitor function is to the velocity of the system.
4. ```qmin``` - The minimum threshold of q.
5. ```T_max``` - The simulation time for the system.
6. ```init_path``` - The initial guess for the path. It can take ```'linear'```  (default), ```'ode'``` and ```'log'``` as inputs.

# Defining the System

```_generate_odes ``` constructs symbolic ODEs using the SymPy library, taking in RACIPE generated parameters.
Here we are using a multiplicative model of gene regulation, where the influence of each regulator of a gene is accounted for by multiplying its shifted-Hill's function to the production term.

The function returns a list of symbolic ODEs.

```_generate_drift_function``` uses SymPy's ```lambdify``` function to convert the ODEs into functions that can take inputs. We will use these later to plot trajectories and time series.

# Dimensional Reduction


The following functions are used to derive the global covariance matrix that will be used to construct the dimensionally-reduced energy landscape.

The local covariance matrix for each steady state is calculated by solving the continous Lyapunov equation, for which the Jacobian matrix at that steady state is required.

```global_covariance_matrix ``` calculates the global covariance matrix using the weighted sum mentioned in the paper's Methods section.

# Visualizing the Landscape

```visualize_landscape``` generates a 2D-contour plot of the dimensionally-reduced system with first principal component as the X-axis and the second principal component as the Y-axis.

**Note**: In the MATLAB implementation provided in the Supplementary Material for the paper by Kang and Li, a global covariance is calculated but not used to visualize the landscape and is instead replaced with a pre-defined matrix.
In our implementation, the multivariate Gaussians are built using the calculated global covariance matrix.

```plot_path_diagnostics``` is used to check if the system is behaving normally. It plots the time series trajectory of the system as derived from the (supposed) minimum action path, along with the phase portrait (only for 2D systems - we were initially working with a 2-node system). A well-behaved system should ideally not show any unphysical trajectories - these include hooks, loops etc on the phase plane.

# Main Analysis Pipeline

```run_drl_analysis``` runs the pipeline for a given parameter set of a given network.

**Pipeline**


1.   Load the data from RACIPE for the chosen parameter set. This includes the set of all the parameters that describe the system, along with the list of steady states of the set with their frequencies.
2.   Polishing the steady states provided by RACIPE to get more accurate steady states.
3. Calculaing the global covariance matrix by solving the continous Lyanpunov equation to get the local covariances, followed by finding the weighted sum.
4. Reducing the system to 2 dimensions using Principal Component Analysis.
5. Deriving a probability distribution using the dimensionally-reduced system by projecting the system onto the PC-plane.
6. Constructing the landscape using said probability distribution.
7. Calculating the path of minimum action using the aMAM solver
8. Projecting the path onto the PC-plane to visualize it, as well as locating saddle points (The saddle point here is approximated as the point on the path that has the maximum potential when projected onto the plane).

The function returns the values of the action cost for traversal from state A to state B as well as the other way 'round.