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

def plot_path_diagnostics(map_a_to_b, time_mesh_ab, map_b_to_a, time_mesh_ba, var_names=['A', 'B','C','D']):
        fig, ax = plt.subplots(2, 2, figsize=(6,6))
        fig.suptitle('Minimum Action Path Diagnostic Plots', fontsize=16)
        x_ab, y_ab = map_a_to_b[:, 0], map_a_to_b[:, 1]
        ax[0, 0].plot(time_mesh_ab, x_ab, label=f'Concentration of {var_names[0]}')
        ax[0, 0].plot(time_mesh_ab, y_ab, label=f'Concentration of {var_names[1]}')
        ax[0, 0].set_title('Time Series Trajectory (A to B)')
        ax[0, 0].set_xlabel('Time (from solver)')
        ax[0, 0].set_ylabel('Concentration')
        ax[0, 0].legend()
        ax[0, 0].grid(True, linestyle='--', alpha=0.5)
        ax[0, 1].plot(x_ab, y_ab, 'b-')
        ax[0, 1].plot(x_ab[0], y_ab[0], 'o', c='cyan', mec='black', ms=8, label='Start (State A)')
        ax[0, 1].plot(x_ab[1], y_ab[1], 's', c='white', mec='black', ms=8, label='End (State B)')
        ax[0, 1].set_title('Phase Portrait (A to B)')
        ax[0, 1].set_xlabel(f'Concentration of {var_names[0]}')
        ax[0, 1].set_ylabel(f'Concentration of {var_names[1]}')
        ax[0, 1].legend()
        ax[0, 1].grid(True, linestyle='--', alpha=0.5)
        x_ba, y_ba = map_b_to_a[:, 0], map_b_to_a[:, 1]
        ax[1, 0].plot(time_mesh_ba, x_ba, label=f'Concentration of {var_names[0]}')
        ax[1, 0].plot(time_mesh_ba, y_ba, label=f'Concentration of {var_names[1]}')
        ax[1, 0].set_title('Time Series Trajectory (B to A)')
        ax[1, 0].set_xlabel('Time (from solver)')
        ax[1, 0].set_ylabel('Concentration')
        ax[1, 0].legend()
        ax[1, 0].grid(True, linestyle='--', alpha=0.5)
        ax[1, 1].plot(x_ba, y_ba, 'm:')
        ax[1, 1].plot(x_ba[0], y_ba[0], 's', c='white', mec='black', ms=8, label='Start (State B)')
        ax[1, 1].plot(x_ba[-1], y_ba[-1], 'o', c='cyan', mec='black', ms=8, label='End (State A)')
        ax[1, 1].set_title('Phase Portrait (B to A)')
        ax[1, 1].set_xlabel(f'Concentration of {var_names[0]}')
        ax[1, 1].set_ylabel(f'Concentration of {var_names[1]}')
        ax[1, 1].legend()
        ax[1, 1].grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()