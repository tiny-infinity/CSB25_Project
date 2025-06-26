import matplotlib.pyplot as plt
import numpy as np

def visualize_landscape(U, X_grid, Y_grid, projected_ss, saddle_coords_ab, proj_path_ab, saddle_coords_ba, proj_path_ba):
    """
    Standard visualization function without outlier masking for paths.
    """
    plt.figure(figsize=(10, 8))
    vmax_clip = np.percentile(U, 99)
    x_min_plot, x_max_plot = X_grid.min(), X_grid.max()
    y_min_plot, y_max_plot = Y_grid.min(), Y_grid.max()

    contour = plt.contourf(X_grid, Y_grid, U, levels=100, cmap='viridis_r', vmax=vmax_clip)
    plt.colorbar(contour, label='Potential Energy (U = -ln(P))')

    # Plot stable states
    if projected_ss:
        plt.plot(projected_ss[0][0], projected_ss[0][1], 'o', color='white', markeredgecolor='black', markersize=8, label='Stable State A')
        if len(projected_ss) > 1:
            plt.plot(projected_ss[1][0], projected_ss[1][1], 's', color='cyan', markeredgecolor='black', markersize=8, label='Stable State B')

    # Plot saddle points
    if saddle_coords_ab is not None:
        plt.plot(saddle_coords_ab[0], saddle_coords_ab[1], 'X', color='red', markersize=9, markeredgecolor='white', label='Saddle Point(s)', zorder=5)
    if saddle_coords_ba is not None and (saddle_coords_ab is None or not np.allclose(saddle_coords_ab, saddle_coords_ba)):
        plt.plot(saddle_coords_ba[0], saddle_coords_ba[1], 'X', color='red', markersize=9, markeredgecolor='white', zorder=5)

    # Plot the raw projected paths directly
    if proj_path_ab is not None:
        path_array_ab = np.array(proj_path_ab)
        plt.plot(path_array_ab[:, 0], path_array_ab[:, 1], 'w--', linewidth=1.5, label='Path A -> B', zorder=4)
    if proj_path_ba is not None:
        path_array_ba = np.array(proj_path_ba)
        plt.plot(path_array_ba[:, 0], path_array_ba[:, 1], 'm:', linewidth=2.0, label='Path B -> A', zorder=4)

    plt.title('Potential Energy Landscape with Transition Paths', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # Set plot limits to the landscape boundaries to maintain a consistent view
    plt.xlim(x_min_plot, x_max_plot)
    plt.ylim(y_min_plot, y_max_plot)
    plt.gca().set_aspect('equal', 'box')
    plt.show()

def plot_path_diagnostics(map_a_to_b, time_mesh_ab, map_b_to_a, time_mesh_ba, var_names=['A', 'B']):
        dims=len(var_names)
        fig, ax = plt.subplots(1,2, figsize=(6,6))
        fig.suptitle('Minimum Action Path Diagnostic Plots', fontsize=16)
        for i in range(dims):
             ax[0].plot(time_mesh_ab, map_a_to_b[:,i], label=f'{var_names[i]}')
        ax[0].set_title('Time Series Trajectory (A to B)')
        ax[0].set_xlabel('Time (from solver)')
        ax[0].set_ylabel('Concentration')
        ax[0].legend()
        ax[0].grid(True, linestyle='--', alpha=0.5)
        for i in range(dims):
             ax[1].plot(time_mesh_ba, map_b_to_a[:,i], label=f'{var_names[i]}')
        ax[1].set_title('Time Series Trajectory (A to B)')
        ax[1].set_xlabel('Time (from solver)')
        ax[1].set_ylabel('Concentration')
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()