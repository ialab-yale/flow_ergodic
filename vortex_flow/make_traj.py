import os, sys, cmocean
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods.flow_emmd import Flow_EMMD

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.titlesize': 28,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})


def run_animation(all_centers, trajectory, dt, color_history, stream_args, filename='flow_traj.mp4'):

    trajectory = trajectory[:, :2]
    M, N, _ = all_centers.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')

    X_LIM, Y_LIM = (0.5, 4.5), (0.5, 4.5)
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.set_aspect('equal')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([0.5, 4.5])
    ax.set_yticks([0.5, 4.5])
    ax.tick_params(axis='x', which='major', pad=11, labelsize=30)
    ax.tick_params(axis='y', which='major', pad=11, labelsize=30)
    ax.grid(True, linestyle=':', alpha=0.25, linewidth=0.8, zorder=0)

    ax.streamplot(
        stream_args['X'], stream_args['Y'],
        stream_args['U_norm'], stream_args['V_norm'],
        color='#4a4a4a', density=0.75, linewidth=1.5, arrowsize=1.5,
        zorder=1,
    )

    initial_x = all_centers[:, 0, 0]
    initial_y = all_centers[:, 0, 1]
    initial_color = color_history[0]

    c_min, c_max = -0.5, 2.0

    centers_scatter = ax.scatter(
        initial_x, initial_y,
        s=160,
        c=initial_color,
        cmap=cmocean.cm.deep,
        vmin=c_min,
        vmax=c_max,
        zorder=4,
        edgecolors='white',
        linewidths=0.7,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.12)
    cbar = fig.colorbar(centers_scatter, cax=cax)
    cbar.set_label('Visitation', fontsize=30, labelpad=12)
    cbar.set_ticks([])
    cbar.ax.tick_params(size=0)

    traj_point, = ax.plot(
        trajectory[0, 0], trajectory[0, 1],
        '^', color='#E8401C', markersize=18, alpha=1.0,
        markeredgecolor='white', markeredgewidth=1.5,
        label='Agent', zorder=6,
    )
    traj_history, = ax.plot(
        [], [],
        '-', color='#E8401C', linewidth=2.5, alpha=0.9,
        label='Path', zorder=5,
    )

    time_text = ax.text(
        0.03, 0.97, '',
        transform=ax.transAxes,
        fontsize=32,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.75, edgecolor='none'),
    )

    # --- Update Function ---
    def update(frame):
        current_centers_x = all_centers[:, frame, 0]
        current_centers_y = all_centers[:, frame, 1]

        coords = np.vstack([current_centers_x, current_centers_y]).T
        centers_scatter.set_offsets(coords)
        centers_scatter.set_array(color_history[frame])

        current_traj_point = trajectory[frame]
        traj_point.set_data([current_traj_point[0]], [current_traj_point[1]])

        path_so_far = trajectory[:frame + 1]
        traj_history.set_data(path_so_far[:, 0], path_so_far[:, 1])

        time_text.set_text(f"$t = {frame * float(dt):.2f}$")

        return centers_scatter, traj_point, traj_history, time_text

    # --- Animation Call ---
    ani = FuncAnimation(
        fig,
        update,
        frames=N,
        blit=True,
        interval=dt * 1000,
        repeat=True,
    )

    ani.save(
        filename,
        writer='ffmpeg',
        fps=1 / dt,
        dpi=150,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'],
    )


if __name__ == "__main__":
    data = np.load(f"flow_data/flow_data.npz")

    centers = np.swapaxes(data['P_HISTORY'], 0, 1)
    dt, T = data['DT'], data['STEPS']

    stream_args = {
        'X': data['X'],
        'Y': data['Y'],
        'U_norm': data['U_norm'],
        'V_norm': data['V_norm']
    }

    flow_args = {
        'X': data['X'],
        'Y': data['Y'],
        'U': data['U'],
        'V': data['V'],
        'centers': centers
    }

    args = {
            'T': T, 
            'h': 0.03, 
            'dt': dt, 
            'max_dx': 5,
            'N_traj': 1,
            'power': 0.5, 
            'dim': 2,
            'data_path': '../vortex_flow/flow_data/flow_data.npz',
        }

    flow = Flow_EMMD(args, x_0=np.array([1.5, 1.5]))
    flow.load_data(flow_args=flow_args)

    flow.solve_flow()

    ## Info Map Calculations ================================================================
    info_term = flow.compute_kernels(flow.trajectory, flow.vert_traj)
    info_sum = jnp.sum(info_term, axis=0)

    point_color = jnp.cumsum(info_term, axis=0)
    ## ======================================================================================

    trajectory = np.hstack((flow.trajectory, np.ones((flow.trajectory.shape[0], 1))))

    run_animation(centers, trajectory, dt, np.asarray(point_color), stream_args, filename="vortex_emmd.mp4")
