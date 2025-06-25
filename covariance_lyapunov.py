"""
This code snippet is designed to compute the covariance Lyapunov function for a system of differential equations, 
specifically for a two-variable system. It includes functions to calculate the global mean and covariance matrix 
from steady states.
It's supposed to work as a module that can be imported and used in other scripts.

The idea is to input the dynamics of the system (in the form of the ODEs that describes) along with its steady states 
and get as output the global covariance matrix
"""

import numpy as np
from scipy.integrate import solve_ivp
from sympy import *

