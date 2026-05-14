import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import sys

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 24,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

class Vortex_Flows:
    def __init__(self, input_args):
        self.input_args = input_args
        self.N = input_args.get('N', 30)
        self.T = input_args.get('T', 300)
        
        self.dt     = 0.05
        self.k_d    = 4.0
        self.k_spin = 2.0
        self.k_att  = 1.0
        self.stab_r = 0.01
        self.center = np.array([2.5, 2.5])
        self.k_wr       = 0.05
        self.crit_dist  = 0.15
        self.max_wall_f = 5.0
        self.wall_pad   = 0.2
        self.extent = [0.5, 4.5, 0.5, 4.5]

        self.P = np.zeros((self.N, 2))
        self.V = np.zeros((self.N, 2))
        self.t = 0.0
        self.centers = []

        self.init_pos()

    def init_pos(self):
        x_min, x_max, y_min, y_max = self.extent
        self.P[:, 0] = np.random.uniform(x_min + self.wall_pad, x_max - self.wall_pad, self.N)
        self.P[:, 1] = np.random.uniform(y_min + self.wall_pad, y_max - self.wall_pad, self.N)
        self.V = np.zeros((self.N, 2))

    def get_vortex_flow(self, p):
        r_vec = p - self.center
        r_perp = np.array([-r_vec[1], r_vec[0]])
        sr = np.sqrt(np.dot(r_vec, r_vec) + self.stab_r**2)
        return -self.k_att * r_vec + (self.k_spin / sr) * r_perp

    def get_wall_rep(self, p):
        x_min, x_max, y_min, y_max = self.extent
        F = np.zeros(2)
        for d, sign, axis in [
            (p[0] - x_min, +1, 0),
            (x_max - p[0], -1, 0),
            (p[1] - y_min, +1, 1),
            (y_max - p[1], -1, 1)]:
            if 1e-6 < d < self.crit_dist:
                F[axis] += sign * min(self.k_wr / d, self.max_wall_f)
        return F

    def calc_accel(self, i):
        p_i, v_i = self.P[i], self.V[i]
        F = self.get_vortex_flow(p_i) + self.get_wall_rep(p_i)
        F -= self.k_d * v_i
        return F

    def step(self):
        dv_dt = np.array([self.calc_accel(i) for i in range(self.N)])
        self.V += dv_dt * self.dt
        self.P += self.V  * self.dt
        self.t += self.dt
        self.P[:, 0] = np.clip(self.P[:, 0], self.extent[0] + 1e-4, self.extent[1] - 1e-4)
        self.P[:, 1] = np.clip(self.P[:, 1], self.extent[2] + 1e-4, self.extent[3] - 1e-4)
        self.centers.append(self.P.copy())

    ## =========================================================================

    def make_flow(self):
        print(f"Running simulation (T = {self.T})...")
        for i in range(self.T):
            if i % 50 == 0:
                sys.stdout.write(f"\rStep {i}/{self.T}")
                sys.stdout.flush()
            self.step()
        print(f"\rSim complete.")

    def flow_grid(self, res=30):
        x_min, x_max, y_min, y_max = self.extent
        Xm, Ym = np.meshgrid(np.linspace(x_min, x_max, res),
                              np.linspace(y_min, y_max, res))
        Um, Vm = np.zeros_like(Xm), np.zeros_like(Ym)
        for i in range(res):
            for j in range(res):
                v = self.get_vortex_flow(np.array([Xm[i, j], Ym[i, j]]))
                Um[i, j], Vm[i, j] = v
        return Xm, Ym, Um, Vm

    def animate(self, save=True, filename='vortex_flow.mp4'):
        x_min, x_max, y_min, y_max = self.extent

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xticks([x_min, x_max])
        ax.set_yticks([y_min, y_max])
        ax.tick_params(axis='both', which='major', pad=11, labelsize=30)
        ax.grid(True, linestyle=':', alpha=0.25, linewidth=0.8, zorder=0)

        Xm, Ym, Um, Vm = self.flow_grid()
        Mag = np.sqrt(Um**2 + Vm**2) + 1e-6
        ax.streamplot(Xm, Ym, Um / Mag, Vm / Mag,
                      color='dimgray', density=0.75, linewidth=1.5, arrowsize=1.5, zorder=1)

        P0 = self.centers[0] if self.centers else self.P
        scatter = ax.scatter(P0[:, 0], P0[:, 1],
                             s=160, c='steelblue', edgecolors='white', linewidths=0.7, zorder=4)

        time_text = ax.text(
            0.03, 0.97, '',
            transform=ax.transAxes,
            fontsize=32, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.75, edgecolor='none'),
        )

        T_frames = len(self.centers)

        def update(k):
            scatter.set_offsets(self.centers[k])
            time_text.set_text(f'$t = {k * self.dt:.2f}$')
            sys.stdout.write(f"\rFrame {k + 1}/{T_frames}")
            sys.stdout.flush()
            return scatter, time_text

        ani = FuncAnimation(fig, update, frames=T_frames,
                            blit=True, interval=self.dt * 1000, repeat=False)

        if save:
            print(f"\nSaving animation...")
            ani.save(filename, writer=animation.FFMpegWriter(fps=30),
                     savefig_kwargs={'facecolor': 'white'})
            print(f"\nSaved to {filename}")
        else:
            plt.show()

        plt.close(fig)

    def save_flow(self, filename='vortex_data.npz'):
        x_min, x_max, y_min, y_max = self.extent
        Xm, Ym, Um, Vm = self.flow_grid()

        wall_corners = np.array([[x_min, y_min], [x_max, y_min],
                                  [x_max, y_max], [x_min, y_max], [x_min, y_min]])

        np.savez(
            filename,
            centers = np.array(self.centers),
            wall_corners = wall_corners,
            center = self.center,
            grid = np.stack([Xm, Ym], axis=-1),
            U = Um,
            V = Vm,
            dt = self.dt,
            N = self.N,
            T = len(self.centers),
        )
        print(f"Data saved to '{filename}'")

if __name__ == '__main__':
    input_args = {
        'N': 75,
        'T': 100,
    }

    flow_maker = Vortex_Flows(input_args)

    flow_maker.make_flow()
    flow_maker.animate(save=True)
    flow_maker.save_flow(filename='flow_data/vortex_data.npz')
