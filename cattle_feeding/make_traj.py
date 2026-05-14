import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.flow_emmd import Flow_EMMD
from cattle_flow import Cow_Flows

def run_animation(centers, trajectory, dt, wall_corners, sinks, U, V, grid):
    n_cows, T_frames, _ = centers.shape
    trajectory = trajectory[:, :2]

    x_min, x_max = wall_corners[0, 0], wall_corners[1, 0]
    y_min, y_max = wall_corners[0, 1], wall_corners[2, 1]

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.set_xlim(x_min - 0.25, x_max + 0.25)
    ax.set_ylim(y_min - 0.25, y_max + 0.25)
    ax.set_aspect('equal')
    ax.axis('off')

    x_g = grid[0, :, 0]
    y_g = grid[:, 0, 1]
    Mag = np.sqrt(U**2 + V**2) + 1e-6
    ax.streamplot(x_g, y_g, U / Mag, V / Mag,
                  color='gray', density=1.0, linewidth=1.0, arrowsize=1.2, zorder=1)

    ax.plot(wall_corners[:, 0], wall_corners[:, 1],
            color='black', linewidth=2.5, solid_capstyle='round', zorder=2)

    ax.plot(sinks[:, 0], sinks[:, 1],
            'o', color='chocolate', markersize=14, zorder=4)

    scatter = ax.scatter(
        centers[:, 0, 0], centers[:, 0, 1],
        c='black',
        marker=Cow_Flows._cow_marker(),
        s=300, edgecolors='none',
        zorder=10
    )

    traj_line,  = ax.plot([], [], '-', color='red', linewidth=3, zorder=8)
    traj_point, = ax.plot(trajectory[0, 0], trajectory[0, 1],
                          'D', color='red', markersize=8, zorder=9)

    time_text = ax.text(0.02, 0.97, 't = 0.0 s',
                        transform=ax.transAxes,
                        fontsize=9, va='top', ha='left',
                        color='black', fontfamily='monospace')

    ax.invert_xaxis()
    ax.invert_yaxis()

    def update(k):
        scatter.set_offsets(centers[:, k, :2])
        traj_line.set_data(trajectory[:k+1, 0], trajectory[:k+1, 1])
        traj_point.set_data([trajectory[k, 0]], [trajectory[k, 1]])
        time_text.set_text(f't = {k * dt:.1f} s')
        sys.stdout.write(f"\rFrame {k + 1}/{T_frames}")
        sys.stdout.flush()
        return scatter, traj_line, traj_point, time_text

    ani = FuncAnimation(fig, update, frames=T_frames,
                        blit=True, interval=dt * 1000, repeat=False)

    print('\nSaving animation...')
    ani.save('cattle_traj.mp4', writer='ffmpeg', fps=30,
             savefig_kwargs={'facecolor': 'white'})
    print("Saved to 'cattle_traj.mp4'.")
    plt.close(fig)


if __name__ == '__main__':
    data = np.load('cattle_data.npz')

    centers = np.swapaxes(data['centers'], 0, 1)
    dt, T = data['dt'], data['T']
    wall_corners = data['wall_corners']
    sinks = data['sinks']
    U, V, grid = data['U'], data['V'], data['grid']

    args = {
        'T': T,
        'h': 0.11,
        'dt': dt,
        'power': 0.25
    }

    flow = Flow_EMMD(args, x_0=np.array([3.0, 3.0]))
    flow.load_data(flow_args={'centers': centers})
    flow.solve_flow()

    run_animation(centers, flow.trajectory, dt, wall_corners, sinks, U, V, grid)