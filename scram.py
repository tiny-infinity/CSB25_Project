import numpy as np
from scipy.stats import multivariate_normal

# --- Assume you have these variables from the previous steps ---
a1,a2= np.meshgrid(np.linspace(-50, 50, 250), np.linspace(-50, 50, 250))
# a1 and a2 are your 2D meshgrid arrays (e.g., shape (250, 250))
# mu_pca is a list of your 2D projected stable state means
# alpha is the list of corresponding weights
# N_stable_states is the number of stable states (e.g., 2 for MISA)

# --- Inside Step 5: Construct the Landscape ---

# 1. Create the grid (You've already planned this)
# grid_range = np.linspace(-50, 50, 250)
# a1, a2 = np.meshgrid(grid_range, grid_range)

# This packs the grid points into a format the PDF function needs
# pos will have shape (250, 250, 2)
pos = np.dstack((a1, a2))

# 2. Initialize your final probability surface
P = np.zeros_like(a1)

# 3. Define the "width" of your valleys. This is a small 2x2 covariance matrix.
# A larger value makes the valleys wider. Start with something like this.
valley_covariance = np.array([[15, 0], [0, 15]])

# 4. Loop through each stable state to create and add its Gaussian hump
for j in range(N_stable_states):
    # Get the 2D center for this stable state's Gaussian
    mean_2d = mu_pca[j]

    # Get the weight for this stable state
    weight = alpha[j]

    # ---> THIS IS THE KEY CALCULATION <---
    # Create the multivariate normal distribution object
    rv = multivariate_normal(mean=mean_2d, cov=valley_covariance)

    # Calculate the probability density at every point on the grid
    # rv.pdf(pos) returns a 2D array with the same shape as a1 and a2
    gaussian_surface = rv.pdf(pos)

    # Weight this surface and add it to the total probability landscape
    P += weight * gaussian_surface

# 5. Now you can calculate the potential energy U
# U = -np.log(np.maximum(P, 1e-100))