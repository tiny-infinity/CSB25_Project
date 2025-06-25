import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from action_solver import aMAM_advanced_solver, _action_cost

# --- Example 1: Semicircle Potential ---

def potential_semicircle(phi):
    """Calculates the semicircle potential V(x,y)."""
    x, y = phi
    r_sq = x**2 + y**2
    # Add epsilon to avoid division by zero
    return (1 - r_sq)**2 + y**2 / (r_sq + 1e-9)

def drift_semicircle(phi):
    """Calculates the drift b = -grad(V) for the semicircle potential."""
    x, y = phi
    r_sq = x**2 + y**2
    # Add epsilon to avoid division by zero at the origin
    denom_sq = (r_sq + 1e-9)**2
    
    # Partial derivatives of V
    dV_dx = -4 * x * (1 - r_sq) - (2 * x * y**2) / denom_sq
    dV_dy = -4 * y * (1 - r_sq) + (2 * y * x**2) / denom_sq
    
    return -np.array([dV_dx, dV_dy])

def test_semicircle():
    """
    Tests the aMAM solver on the semicircle potential example.
    The MAP is the upper arc of a unit circle, and the action is 2.0.
    """
    print("\n" + "="*25)
    print("Testing Semicircle Example")
    print("="*25)
    
    x0 = np.array([-1.0, 0.0])
    x1 = np.array([1.0, 0.0])
    
    # Solver parameters from the paper (Fig. 4, Fig. 6)
    N = 100        # Number of points on the path
    T_max = 30     # Corresponds to T=30, interval [-15, 15]
    k_max = 10     # Max remeshing iterations
    c = 8000       # Monitor function constant from paper 

    final_path, final_time_mesh = aMAM_advanced_solver(drift_semicircle, x0, x1, N, T_max, k_max, c)
    final_action = _action_cost(final_path, final_time_mesh, drift_semicircle)
    
    print(f"\nFinal Action: {final_action:.4f} (Expected: ~2.0)")

    # Plotting
    grid = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(grid, grid)
    Z = potential_semicircle([X, Y])
    
    theta = np.linspace(0, np.pi, 100)
    x_exact = np.cos(theta)
    y_exact = np.sin(theta)
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20)-1, cmap='viridis_r')
    plt.plot(x_exact, y_exact, 'k--', label='Exact MAP (Unit Circle)', linewidth=2)
    plt.plot(final_path[:, 0], final_path[:, 1], 'r-o', label='aMAM Calculated Path')
    plt.title('Semicircle Potential: aMAM Result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# --- Example 2: Mueller Potential ---

def potential_mueller(phi):
    """Calculates the Mueller potential V(x,y)."""
    x, y = phi
    # Parameters from the paper 
    D = np.array([-200, -100, -170, 15])
    A = np.array([-1, -1, -6.5, 0.7])
    B = np.array([0, 0, 11, 0.6])
    C = np.array([-10, -10, -6.5, 0.7])
    x_prime = np.array([1, 0, -0.5, -1])
    y_prime = np.array([0, 0.5, 1.5, 1])

    val = 0
    for i in range(4):
        val += D[i] * np.exp(A[i]*(x - x_prime[i])**2 + B[i]*(x - x_prime[i])*(y - y_prime[i]) + C[i]*(y - y_prime[i])**2)
    return val

def drift_mueller(phi):
    """Calculates the drift b = -grad(V) for the Mueller potential."""
    x, y = phi
    # Parameters from the paper
    D = np.array([-200, -100, -170, 15])
    A = np.array([-1, -1, -6.5, 0.7])
    B = np.array([0, 0, 11, 0.6])
    C = np.array([-10, -10, -6.5, 0.7])
    x_prime = np.array([1, 0, -0.5, -1])
    y_prime = np.array([0, 0.5, 1.5, 1])
    
    dV_dx, dV_dy = 0.0, 0.0
    for i in range(4):
        exp_term = np.exp(A[i]*(x - x_prime[i])**2 + B[i]*(x - x_prime[i])*(y - y_prime[i]) + C[i]*(y - y_prime[i])**2)
        common_term = D[i] * exp_term
        dV_dx += common_term * (2*A[i]*(x-x_prime[i]) + B[i]*(y-y_prime[i]))
        dV_dy += common_term * (B[i]*(x-x_prime[i]) + 2*C[i]*(y-y_prime[i]))
        
    return -np.array([dV_dx, dV_dy])

def test_mueller():
    """Tests the aMAM solver on the Mueller potential."""
    print("\n" + "="*25)
    print("Testing Mueller Potential Example")
    print("="*25)
    
    # Find the potential minima by numerical optimization
    # Start points are chosen by inspecting Fig. 11 in the paper
    min1_res = minimize(lambda p: potential_mueller(p), (-0.5, 1.5))
    min2_res = minimize(lambda p: potential_mueller(p), (0.6, 0.0))
    x0 = min1_res.x
    x1 = min2_res.x
    print(f"Found Minima: Start (x0) = {x0}, End (x1) = {x1}")
    
    # Solver parameters
    N = 100       # Number of points
    T_max = 24    # Corresponds to T=12 from Fig. 11
    k_max = 15    # Max remeshing iterations
    c = 4000      # Monitor function constant (from Fig 12 caption )

    final_path, final_time_mesh = aMAM_advanced_solver(drift_mueller, x0, x1, N, T_max, k_max, c)
    final_action = _action_cost(final_path, final_time_mesh, drift_mueller)
    
    print(f"\nFinal Action: {final_action:.4f}")

    # Plotting
    grid_x = np.linspace(-1.5, 1.0, 100)
    grid_y = np.linspace(-0.5, 2.0, 100)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = potential_mueller([X, Y])
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.linspace(-150, 150, 30), cmap='jet')
    plt.plot(final_path[:, 0], final_path[:, 1], 'r-o', markersize=4, label='aMAM Calculated Path')
    plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start Minimum')
    plt.plot(x1[0], x1[1], 'bo', markersize=10, label='End Minimum')
    plt.title('Mueller Potential: aMAM Result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# --- Example 3: Simple Nongradient System ---

def drift_nongradient(phi):
    """Drift term for the simple nongradient system. """
    x, y = phi
    dxdt = x * (3 - x - 2*y)
    dydt = y * (3 - y - 2*x)
    return np.array([dxdt, dydt])

def test_nongradient():
    """Tests the aMAM solver on the simple nongradient system."""
    print("\n" + "="*25)
    print("Testing Simple Nongradient Example")
    print("="*25)

    x0 = np.array([0.0, 3.0])
    x1 = np.array([3.0, 0.0])

    # Solver parameters from Fig. 14 discussion
    N = 50       # Number of points 
    T_max = 30 # A value where the action is stable in Fig. 14
    k_max = 10   # Max remeshing iterations
    c = 1000   # A reasonable constant

    # The path may start near (0,0), so use a linear initial guess
    final_path, final_time_mesh = aMAM_advanced_solver(drift_nongradient, x0, x1, N, T_max, k_max, c, initial_path='ode')
    final_action = _action_cost(final_path, final_time_mesh, drift_nongradient)
    
    # Benchmark from the paper is ~1.5768 
    print(f"\nFinal Action: {final_action:.4f} (Expected: ~1.5768)")

    # Plotting
    grid = np.linspace(-0.5, 3.5, 20)
    X, Y = np.meshgrid(grid, grid)
    U, V = drift_nongradient([X, Y])
    mags=np.sqrt(U**2 + V**2)
    resU,resV=U/mags,V/mags

    plt.figure(figsize=(7, 7))
    # Vector field
    plt.quiver(X, Y, resU, resV, color='gray', width=0.003)
    # Separatrix y=x
    plt.plot(grid, grid, 'k--', label='Separatrix (y=x)')
    plt.plot(final_path[:, 0], final_path[:, 1], 'r-o', label='aMAM Calculated Path')
    plt.plot(x0[0], x0[1], 'go', markersize=10, label='Start State (0,3)')
    plt.plot(x1[0], x1[1], 'bo', markersize=10, label='End State (3,0)')
    plt.title('Simple Nongradient System: aMAM Result')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 3.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == '__main__':
    test_nongradient()
    