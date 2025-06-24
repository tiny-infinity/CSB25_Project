import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize


def adaptive_reparametrize(path, num_points):
    """
    Redistribute points along path to be equidistant (arc length parametrization).
    """
    tck, u = splprep(path.T, s=0)
    u_new = np.linspace(0, 1, num_points)
    new_path = np.array(splev(u_new, tck)).T
    return new_path


def _action_cost(x, f, N, dt, dim):
    """
    Compute Freidlin-Wentzell action functional for the path x.
    """
    x = x.reshape((N, dim))
    cost = 0.0
    for i in range(N - 1):
        dx = (x[i + 1] - x[i]) / dt
        fx = f(x[i])
        cost += 0.25 * np.sum((dx - fx)**2) * dt
    return cost


def aMAM_solver(f, x0, x1, N=100, T=1.0, max_iter=100, lr=1e-2, reparam_every=10):
    """
    Adaptive Minimum Action Method (aMAM) for computing Minimum Action Paths.

    Parameters:
    - f: function f(x) giving deterministic dynamics (vector field)
    - x0, x1: initial and final states (1D arrays)
    - N: number of discretized points along the path
    - T: total transition time
    - max_iter: maximum number of optimization iterations
    - lr: learning rate for gradient descent
    - reparam_every: reparametrize path every this many iterations

    Returns:
    - path: N x dim array of MAP points
    """
    dim = len(x0)
    path = np.linspace(x0, x1, N)
    dt = T / (N - 1)

    for it in range(max_iter):
        new_path = path.copy()
        for i in range(1, N - 1):
            xi = path[i]
            xi_minus = path[i - 1]
            xi_plus = path[i + 1]

            dx = (xi_plus - xi_minus) / (2 * dt)
            f_i = f(xi)
            grad = dx - f_i
            new_path[i] -= lr * grad

        if (it + 1) % reparam_every == 0:
            new_path = adaptive_reparametrize(new_path, N)

        path = new_path

    return path


def integrate_aMAM_into_drl(network_drift_func, mu_A, mu_B, project_to_drl_space=None, **kwargs):
    """
    Wrapper to apply aMAM to your DRL pipeline using provided drift function and endpoints.

    Parameters:
    - network_drift_func: callable, your system's deterministic vector field f(x)
    - mu_A, mu_B: numpy arrays, coordinates of two stable states in full state space
    - project_to_drl_space: optional function to map trajectory into 2D DRL/PCA space
    - kwargs: parameters for aMAM_solver (e.g., N, T, max_iter)

    Returns:
    - path: array of shape (N, dim)
    - projected_path: array of shape (N, 2) if project_to_drl_space is provided
    """
    path = aMAM_solver(network_drift_func, mu_A, mu_B, **kwargs)
    if project_to_drl_space:
        return path, np.array([project_to_drl_space(p) for p in path])
    return path, None
