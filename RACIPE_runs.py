"""
This module takes in a network topology in the form of a .topo file and runs RACIPE
on it to generate a set of parameter sets. It then runs the ODEs defined in MISA_ODE_general.py
and saves the steady states and their corresponding parameters to a .csv file.

It will have to access the command line and therefore will have to use the subprocess module.

Maybe will modify to select out only the parameter sets giving bistability. 
"""

