"""
This module takes in steady states, their frequencies and the covariance matrix of the system generated 
from covariance_lyapunov.py, and returns the dimensionally-reduced proabability distribution of the system.

It uses PCA to reduce the dimensions of the steady states.

"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
from covariance_module import *

def pca_analysis(global_cov_matrix):
   eigevals, eigenvectors = eigh(global_cov_matrix)
   sorted_indices = np.argsort(eigevals)[::-1]  
   sorted_eigevals = eigevals[sorted_indices]
   sorted_eigenvectors = eigenvectors[:, sorted_indices]
   PC1 = sorted_eigenvectors[:, 0]
   PC2 = sorted_eigenvectors[:, 1]
   total_variance = np.sum(eigevals)
   explained_variance_ratio = eigevals[sorted_indices] / total_variance
   return PC1, PC2, explained_variance_ratio

def multiply_principal_components(PC1, PC2):
    return np.dot(PC1, PC2)

def transformation_matrix(PC1, PC2):
    return np.column_stack((PC1, PC2)).T

def project_steady_states(steady_states, transformation_matrix):
    steady_states_array = np.array(steady_states) #need to externally generate this
    projections=[]
    for i in range(steady_states_array.shape[0]):
        projections.append(transformation_matrix @ steady_states_array[i]) 
    return projections

X_grid, Y_grid = np.meshgrid(np.linspace(-100, 100, 10000), np.linspace(-100, 100, 10000))

def probabiliy_distribution(projected_steady_states,ss_frequencies):
    probability_surface=np.zeros((X_grid.shape[0], X_grid.shape[1]))
    for i in range(len(projected_steady_states)):
        mean_2d = projected_steady_states[i]
        weight = ss_frequencies[i]
        rv= multivariate_normal(mean=mean_2d, cov=np.array([[1,0], [0, 1]]))
        gaussian_surface = rv.pdf(np.dstack((X_grid, Y_grid)))
        probability_surface += weight * gaussian_surface
    return probability_surface

def potential_energy(probability_surface):
    return -np.log(np.maximum(probability_surface, 1e-100))



