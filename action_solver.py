import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def _action_S(phi_flat, time_mesh, M_dims, f_drift, phi_start, phi_end):
    """
    Calculates the standard Friedlin-Wentzell action functional.
    """
    phi_internal = phi_flat.reshape((M_dims, -1))
    full_path = np.column_stack((phi_start, phi_internal, phi_end))
    delta_ts = np.diff(time_mesh)
    phir = full_path[:, 1:]
    phil = full_path[:, :-1]
    phihalf = (phir + phil) / 2.0 # Midpoint rule
    d_phi_dt = (phir - phil) / (delta_ts + 1e-12)
    f_at_phihalf = f_drift(phihalf)
    integrand = np.sum((d_phi_dt - f_at_phihalf)**2, axis=0)
    action = 0.5 * np.sum(integrand * delta_ts) # Discrete form of the action integral
    return action

def _action_cost(path, time_mesh, drift_func):
    """
    Convenience function to calculate the action of a final, complete path.
    """
    return _action_S(path.T[:, 1:-1].flatten(), time_mesh, path.shape[1], drift_func, path[0], path[-1])

def _monitor_function(path, time_mesh, c):
    """
    Calculates the monitor function w(t) to guide remeshing.
    """
    phir = path[:, 1:]
    phil = path[:, :-1]
    delta_ts = np.diff(time_mesh)
    velocity = (phir - phil) / (delta_ts + 1e-12)
    velocity_sq_mag = np.sum(velocity**2, axis=0)
    w = np.sqrt(1.0 + c * velocity_sq_mag) # Equation (11) from Zhou et al.
    return w

def remesh_path_and_time(old_phi, old_time_mesh, n_points, c):
    """
    Performs adaptive remeshing based on the moving mesh strategy.
    """
    monitor = _monitor_function(old_phi, old_time_mesh, c)
    delta_ts_old = np.diff(old_time_mesh)
    alpha = np.zeros(len(old_time_mesh))
    alpha[1:] = np.cumsum(monitor * delta_ts_old)
    if alpha[-1] > 0:
        alpha /= alpha[-1]

    # Handle cases where path is flat, causing non-unique alpha values
    unique_indices = np.unique(alpha, return_index=True)[1]
    alpha_unique = alpha[unique_indices]
    old_time_mesh_unique = old_time_mesh[unique_indices]
    old_phi_unique = old_phi[:, unique_indices]
    
    path_interp_kind = 'cubic'
    if len(unique_indices) < 4:
        path_interp_kind = 'linear'
    
    alpha_new = np.linspace(0, 1, n_points)
    time_mesh_new = interp1d(alpha_unique, old_time_mesh_unique, kind='linear', fill_value="extrapolate")(alpha_new)
    phi_new_T = interp1d(alpha_unique, old_phi_unique.T, kind=path_interp_kind, axis=0, fill_value="extrapolate")(alpha_new)
    
    # Directly return the newly interpolated path without damping
    phi_new = phi_new_T.T 
    return phi_new, time_mesh_new

def _action_cost(path, time_mesh, drift_func):
    """
    Calculating the action of a complete, fully formed path.
    Used to calculate the final action cost after the optimization is complete
    """
    phi = path.T
    delta_ts = np.diff(time_mesh)
    phir = phi[:, 1:]
    phil = phi[:, :-1]
    phihalf = (phir + phil) / 2.0
    d_phi_dt = (phir - phil) / (delta_ts + 1e-12)
    f_at_phihalf = drift_func(phihalf)
    integrand = np.sum((d_phi_dt - f_at_phihalf)**2, axis=0)
    action = 0.5 * np.sum(integrand * delta_ts)
    return action

def aMAM_advanced_solver(f_drift, x0, x1, N, T_max, k_max, c, qmin=3.0, phi_init=None, initial_path='log'):
    """
    aMAM solver with flexible initial path.
    initial_path: 'linear' (default), 'log', or 'ode'
    """
    M_dims = len(x0)
    if phi_init is not None:
        phi = phi_init.T
    else:
        if initial_path == 'log':
            x0_log = np.log(np.maximum(x0, 1e-12))
            x1_log = np.log(np.maximum(x1, 1e-12))
            path_init_log = np.array([np.linspace(s, e, N) for s, e in zip(x0_log, x1_log)]).T
            phi = np.exp(path_init_log).T
        elif initial_path == 'ode':
            from scipy.integrate import solve_ivp
            def ode_func(t, y): return f_drift(y)
            t_span = (0, T_max)
            t_eval = np.linspace(0, T_max, N)
            sol = solve_ivp(ode_func, t_span, x0, t_eval=t_eval, vectorized=True)
            phi = sol.y
            phi[:, -1] = x1
        else:
            path_init_T = np.array([np.linspace(s, e, N) for s, e in zip(x0, x1)]).T
            phi = path_init_T.T

    time_mesh = np.linspace(-T_max / 2.0, T_max / 2.0, N)
    num_internal_points = N - 2

    # --- TIGHTENED BOUNDS: restrict to a region around the endpoints ---
    lower_bound_vec = np.minimum(x0, x1) * 0.1
    upper_bound_vec = np.maximum(x0, x1) * 10
    # Ensure strictly positive and not exceeding a reasonable max
    lower_bound_vec = np.clip(lower_bound_vec, 1e-6, None)
    upper_bound_vec = np.clip(upper_bound_vec, None, 1e6)
    bounds = []
    for _ in range(num_internal_points):
        for lb, ub in zip(lower_bound_vec, upper_bound_vec):
            bounds.append((lb, ub))

    optimizer_options = {'maxiter': 300, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-8, 'maxcor': 10}

    for k in range(k_max):
        print(f"\n--- aMAM Remeshing Cycle {k+1}/{k_max} ---")
        objective_func = lambda p: _action_S(p, time_mesh, M_dims, f_drift, x0, x1)
        phi_internal_flat = phi[:, 1:-1].flatten()
        result = minimize(objective_func, phi_internal_flat, method='L-BFGS-B',
                          bounds=bounds, options=optimizer_options)
        phi_optimized_internal = result.x.reshape((M_dims, num_internal_points))
        phi_optimized = np.column_stack((x0, phi_optimized_internal, x1))
        w = _monitor_function(phi_optimized, time_mesh, c)
        delta_ts = np.diff(time_mesh)
        wdelt = w * delta_ts
        q = np.max(wdelt) / np.min(wdelt) if np.min(wdelt) > 0 else float('inf')
        print(f"Mesh quality q = {q:.2f}")
        if q > qmin and k < k_max - 1:
            phi, time_mesh = remesh_path_and_time(phi_optimized, time_mesh, N, c)
        else:
            phi = phi_optimized
            print("Convergence criterion met or final iteration. Halting remeshing.")
            break

    print("--- aMAM solver finished ---")
    return phi.T, time_mesh