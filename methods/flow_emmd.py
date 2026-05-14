import dill as pkl
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
import jaxopt
import trimesh
import open3d as o3d
from jax.scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import io
import imageio.v3 as iio
import pickle

jax.config.update('jax_enable_x64', True)

try:
    from .augmented_lagrange_wrapper import AugmentedLagrangeWrapper
except:
    from augmented_lagrange_wrapper import AugmentedLagrangeWrapper

class Flow_EMMD:
    def __init__(self, input_args, x_0=None):
        self.input_args = input_args

        self.T = input_args.get('T', 1000)
        self.h = input_args.get('h', 0.1)
        self.dt = input_args.get('dt', 0.1)
        self.max_dx = input_args.get('max_dx', 5)
        self.N_traj = input_args.get('N_traj', 1)
        self.power = input_args.get('power', 0.05)
        self.dim = input_args.get('dim', 2)

        self.data_path = input_args.get('data_path', 'gulf_data.pkl')

        if x_0 is None:
            if self.dim == 2:
                self.x0 = jnp.array([-95.5, 25.0])
            elif self.dim == 3:
                self.x0 = jnp.array([0.08, 0.58, -0.299])
        else:
            self.x0 = x_0

        if self.dim == 2:
            self.xf = jnp.array([-81, 24.0])
        elif self.dim == 3:
            self.xf = jnp.array([-0.2, -0.3, -0.07])

        self.U = jnp.zeros((self.N_traj, self.T, self.dim))
        self.X = jnp.linspace(self.x0, self.xf, num=self.T) + \
                 0.5 * jax.random.normal(jax.random.PRNGKey(0), shape=(self.N_traj, self.T, self.dim))

        self.info_dist = input_args.get('info_dist', lambda x: 1.0)
        self.mesh_path = input_args.get('mesh_path', '../obj_files/bunny_mesh.obj')

        self.solver_c = 1.0
        self.reg_weight = 0.001

    def load_mesh(self, mesh_path="bunny.obj"):
        print('Loading mesh...')

        tri = trimesh.load(mesh_path)
        if isinstance(tri, trimesh.Scene):
            tri = trimesh.util.concatenate(tuple(tri.geometry.values()))
        self.mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(np.asarray(tri.vertices)),
            triangles=o3d.utility.Vector3iVector(np.asarray(tri.faces))
        )

        print('Processing mesh...')

        mesh_verts = np.asarray(self.mesh.vertices)
        centroid = mesh_verts.mean(axis=0)
        mesh_verts = mesh_verts - centroid

        max_dim = np.max(mesh_verts.max(axis=0) - mesh_verts.min(axis=0))
        mesh_verts = mesh_verts / max_dim

        self.mesh.vertices = o3d.utility.Vector3dVector(mesh_verts)
        self.mesh.compute_vertex_normals()

        num_pts = 3200

        mesh_indices = np.random.choice(np.arange(len(self.mesh.vertices)), size=num_pts, replace=False)
        self.points = np.array(self.mesh.vertices)[mesh_indices]

        self.points = jnp.array([ ## Rotating 90 degrees about x-axis
            self.points[:, 0],
            self.points[:, 1],
            -self.points[:, 2]
        ]).T

        print('Mesh Loaded.')

    ## 3D Tools =================================================================================

    def flow_tornado(self, v):
        z_max = 0.01
        c_rad = 5
        omega = 5
        z_offset = 0.1
        k_att = 0.5
        k_vert = 1

        x, y, z = v[0], v[1], v[2]
        r_sq = x**2 + y**2

        S_z = jnp.abs(z) + z_offset

        dx_dt = S_z * (-(omega * y) - (k_att * x))
        dy_dt = S_z * ((omega * x) - (k_att * y))

        radial_focus = jnp.exp(-c_rad * r_sq)
        height_damping = jnp.maximum(0.0, z_max - z)
        dz_dt = k_vert * radial_focus * height_damping

        dvdt = jnp.array([dx_dt, dy_dt, -dz_dt])
        return v + self.dt * dvdt

    def make_dynamic_info_map(self, centers, samples):
        def info_t(center):
            raw = jax.vmap(lambda x: jnp.exp(-jnp.sum((x - center)**2)/self.h))(samples)
            return raw / jnp.sum(raw)
        return jax.vmap(info_t)(centers)

    def compute_kernels(self, X, centers_t):
        def kernel_at_t(x_t, centers_t_t):
            return jax.vmap(lambda c: self.RBF_kernel(x_t, c, self.h))(centers_t_t)
        return jax.vmap(kernel_at_t)(X, centers_t)

    def make_custom_info_map(self, centers, samples, bandwidth=0.005):
        grid = (jnp.linspace(jnp.min(centers[0, :, 0]), jnp.max(centers[0, :, 0]), 100),
                jnp.linspace(jnp.min(centers[0, :, 1]), jnp.max(centers[0, :, 1]), 100))
        self.utility_grid = jnp.ones((100, 100)) / (100*100)
        utility_grid = RegularGridInterpolator(grid, self.utility_grid, fill_value=0.)

        def info_at(center):
            weights_at = utility_grid(center)
            weights_at = jnp.where(weights_at < 0.0, 0.0, weights_at) + 1e-8

            def kernel_weight(center, weight):
                raw = jax.vmap(lambda x: jnp.exp(-jnp.sum((x - center)**2)/bandwidth))(samples)
                return weight * raw

            return jax.vmap(kernel_weight)(center, weights_at)

        return jax.vmap(info_at)(centers)

    ## ============================================================================================

    ## 2D Tools ===================================================================================

    def flow_2d(self, x):
        return x + self.dt * jnp.array([self.u_interp(x)[0], self.v_interp(x)[0]])

    def compute_moving_centers(self, x0):
        return jax.lax.scan(lambda x, _: (self.flow_2d(x), x), x0, None, length=self.T)[1]

    ## ============================================================================================

    def load_data(self, flow_args=None):
        if self.dim == 2:
            if flow_args is None or 'seed_points' in (flow_args or {}):
                with open(self.data_path, 'rb') as f:
                    data = pkl.load(f)

                lon, lat = data['lon'], data['lat']
                u, v = data['u'], data['v']
                speed = data['speed']

                print(f"lon shape: {lon.shape}")
                print(f"lat shape: {lat.shape}")
                print(f"u shape: {u.shape}")
                print(f"v shape: {v.shape}")

                unit_conv_mult = 1.94384 * 24 / 60

                u *= unit_conv_mult
                v *= unit_conv_mult
                speed *= unit_conv_mult

                u[np.isnan(speed)] = 0.0
                v[np.isnan(speed)] = 0.0
                speed[np.isnan(speed)] = 0.0

                mask = speed.ravel() > 0.0
                lon_valid = lon.ravel()[mask]
                lat_valid = lat.ravel()[mask]
                samples_full = jnp.vstack([lon_valid, lat_valid]).T

                # Handle both snapshot and time-series data
                if u.ndim == 3:
                    u_snap = u[0]
                    v_snap = v[0]
                    self.data_u_series = u
                    self.data_v_series = v
                else:
                    u_snap = u
                    v_snap = v
                    self.data_u_series = u[np.newaxis]
                    self.data_v_series = v[np.newaxis]

                self.data_lon = lon
                self.data_lat = lat
                self.data_u = u_snap
                self.data_v = v_snap

                self.grid = (jnp.array(lon[0, :]), jnp.array(lat[:, 0]))
                self.u_interp = RegularGridInterpolator(self.grid, jnp.array(u_snap.T), fill_value=0)
                self.v_interp = RegularGridInterpolator(self.grid, jnp.array(v_snap.T), fill_value=0)

                print('Computing Flow...')

                if flow_args and 'seed_points' in flow_args:
                    seed_points = flow_args['seed_points']
                else:
                    seed_points = samples_full[::80]

                verts = [self.compute_moving_centers(i) for i in seed_points]

            elif 'X' in flow_args:
                x = flow_args['X']
                y = flow_args['Y']
                u = flow_args['U']
                v = flow_args['V']

                print(f"X shape: {x.shape}")
                print(f"Y shape: {y.shape}")
                print(f"U shape: {u.shape}")
                print(f"V shape: {v.shape}")

                self.data_lon = x
                self.data_lat = y
                self.data_u = u
                self.data_v = v

                self.grid = (jnp.array(x[0, :]), jnp.array(y[:, 0]))
                self.u_interp = RegularGridInterpolator(self.grid, u.T, fill_value=0)
                self.v_interp = RegularGridInterpolator(self.grid, v.T, fill_value=0)

                verts = flow_args['centers']

            else:
                # Pre-computed centers only — no flow interpolators
                verts = flow_args['centers']

            self.vert_traj = jnp.swapaxes(jnp.array(verts), 0, 1)
            self.centers = np.array(verts)
            self.samples = jnp.ones(len(verts)) / len(verts)

            print(f"Samples shape: {self.samples.shape}")
            print(f"Centers shape: {self.vert_traj.shape}")
            print('Flow Computed.')

        elif self.dim == 3:
            self.load_mesh(self.mesh_path)

            print('Computing Flow...')

            self.vert_traj = []
            verts = self.points
            for _ in range(self.T):
                self.vert_traj.append(np.copy(verts))
                for _ in range(10):
                    verts = jax.vmap(lambda c: self.flow_tornado(c), in_axes=(0))(verts)

            self.vert_traj = np.array(self.vert_traj)
            centers = jnp.swapaxes(self.vert_traj, 0, 1)
            self.centers = centers
            self.samples = jnp.ones(len(centers)) / len(centers)

            print(f"Samples shape: {self.samples.shape}")
            print(f"Centers shape: {self.vert_traj.shape}")
            print('Flow Computed.')

    @staticmethod
    def RBF_kernel(x, xp, h):
        return jnp.exp(-jnp.sum((x-xp)**2)/h)

    def solve_flow(self):
        def _flow_tornado_step(v, dt):
            z_max = 0.01
            c_rad = 5
            omega = 5
            z_offset = 0.1
            k_att = 0.5
            k_vert = 1

            x, y, z = v[0], v[1], v[2]
            r_sq = x**2 + y**2

            S_z = jnp.abs(z) + z_offset
            dx_dt = S_z * (-(omega * y) - (k_att * x))
            dy_dt = S_z * ((omega * x) - (k_att * y))
            radial_focus = jnp.exp(-c_rad * r_sq)
            height_damping = jnp.maximum(0.0, z_max - z)
            dz_dt = k_vert * radial_focus * height_damping

            return dt * jnp.array([dx_dt, dy_dt, -dz_dt])

        if self.dim == 2:
            if hasattr(self, 'u_interp'):
                def f_dynamics(x, ctrl, dt):
                    return x + dt * (ctrl + jnp.array([self.u_interp(x)[0], self.v_interp(x)[0]]))
            else:
                def f_dynamics(x, ctrl, dt):
                    return x + 0.1 * ctrl
        elif self.dim == 3:
            def f_dynamics(x, ctrl, dt):
                return x + dt * ctrl + _flow_tornado_step(x, dt)

        args = {
            'h': self.h,
            'samples': self.samples,
            'sample_vals': self.samples,
            'x0': self.x0,
            'power': self.power,
            'dt': self.dt,
            'centers': self.vert_traj
        }

        print(f"Samples shape: {self.samples.shape}")
        print(f"Centers shape: {self.vert_traj.shape}")

        params = {'X': self.X, 'U': self.U}
        print('Initial Trajectory Shape:', self.X.shape)

        def loss(params, args):
            def mmd(params, args):
                info_sum = jnp.sum(self.compute_kernels(params['X'][0], args['centers']), axis=0) + 1e-8
                info_sum = info_sum / jnp.sum(info_sum, axis=0, keepdims=True)
                return jnp.linalg.norm(info_sum - args['sample_vals'])**2

            reg = jnp.mean((params['U'][:, 1:] - params['U'][:, :-1])**2)
            return mmd(params, args) + self.reg_weight * reg

        def eq_constr(params, args):
            def dyn_penalty(X, U):
                return X[1:] - vmap(f_dynamics, in_axes=(0, 0, None))(X[:-1], U[:-1], args['dt'])
            def init_cont_penalty(X):
                return args['x0'] - X[0]
            return jnp.hstack([
                vmap(dyn_penalty)(params['X'], params['U']).flatten(),
                vmap(init_cont_penalty)(params['X']).flatten(),
            ])

        def ineq_constr(params, args):
            return jnp.vstack([
                params['U'] - args['power'],
                -params['U'] - args['power'],
            ])

        solver = AugmentedLagrangeWrapper(jaxopt.LBFGS, params, loss, eq_constr, ineq_constr, args, c=self.solver_c, tol=1e-17)
        solver.solve(max_iter=500, eps=1e-8)

        self.trajectory = solver.solution['X'][0]
        self.power_use = np.array(solver.solution['U'])

        print(f"Max Eq Constraint: {jnp.max(jnp.abs(eq_constr(solver.solution, args)[:(len(self.trajectory)-1)]))}")
        print(f"Flow Trajectory Shape: {self.trajectory.shape}")
        print(f"Max Power Use: {np.max(np.abs(self.power_use))}")

        return self.trajectory

    def animate(self, save=True):
        if self.dim == 2:
            self.animate_2d(save=save)
        elif self.dim == 3:
            self.animate_3d(save=save)

    def animate_2d(self, save=True, filename='flow_emmd_2d.mp4'):
        info_term = self.compute_kernels(self.trajectory, self.vert_traj)
        info_sum = jnp.sum(info_term, axis=0)
        point_color = jnp.cumsum(info_term, axis=0) / jnp.sum(info_sum, axis=0, keepdims=True)

        if hasattr(self, 'data_lon'):
            file_path = f"../figures/{filename}"
            extent = [float(jnp.min(self.data_lon)), float(jnp.max(self.data_lon)),
                      float(jnp.min(self.data_lat)), float(jnp.max(self.data_lat))]
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor='#ddccaa')
            ax.add_feature(cfeature.OCEAN, facecolor='#aaddff')
            ax.coastlines(resolution='10m')
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.gridlines(draw_labels=False, linestyle='--', alpha=0.5)

            ax.streamplot(
                self.data_lon, self.data_lat, self.data_u, self.data_v,
                density=3, color='grey', transform=ccrs.PlateCarree()
            )

            trajectory_point, = ax.plot(
                self.trajectory[0, 0], self.trajectory[0, 1],
                'r', markersize=5, transform=ccrs.PlateCarree()
            )
            vert_scatter = ax.scatter(
                self.vert_traj[0, :, 0], self.vert_traj[0, :, 1],
                s=20, c=point_color[0], cmap='summer',
                transform=ccrs.PlateCarree(), vmin=0, vmax=0.01
            )

            images_data = []
            idx = 1
            save_frames = [int(self.T/3), int(2*self.T/3), self.T-1]
            frames_dir = '../figures/compare_2d/images/'
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)

            for t in range(self.T):
                trajectory_point.set_data([self.trajectory[:t, 0], self.trajectory[:t, 1]])
                vert_scatter.set_offsets(self.vert_traj[t])
                vert_scatter.set_array(point_color[t])
                ax.set_title(f'Time Step: {t+1}/{self.T}')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)

                if t in save_frames:
                    ax.set_title(None)
                    plt.savefig(f'{frames_dir}flow_emmd_{idx}.png', dpi=400, bbox_inches='tight')
                    print(f"Saved frame {idx} to {frames_dir}flow_emmd_{idx}.png")
                    idx += 1

                images_data.append(iio.imread(buf))

        else:
            file_path = filename
            fig, ax = plt.subplots(figsize=(10, 8))

            trajectory_point, = ax.plot(
                self.trajectory[0, 0], self.trajectory[0, 1],
                'r', markersize=5
            )
            vert_scatter = ax.scatter(
                self.vert_traj[0, :, 0], self.vert_traj[0, :, 1],
                s=20, c=point_color[0], cmap='summer', vmin=0, vmax=0.01
            )

            images_data = []

            for t in range(self.T):
                trajectory_point.set_data([self.trajectory[:t, 0], self.trajectory[:t, 1]])
                vert_scatter.set_offsets(self.vert_traj[t])
                vert_scatter.set_array(point_color[t])
                ax.set_title(f'Time Step: {t+1}/{self.T}')

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                images_data.append(iio.imread(buf))

        if save and images_data:
            print(f"Creating animation: {file_path} from {len(images_data)} frames...")
            iio.imwrite(file_path, images_data)
            print("Animation saved.")

        plt.close(fig)

    def animate_3d(self, save=True, filename='flow_emmd_3d.mp4'):
        print("Starting Matplotlib 3D Animation...")

        os.makedirs("../figures", exist_ok=True)
        save_path = os.path.join("../figures", filename)

        T = self.T
        vert_traj_np = np.array(self.vert_traj)
        trajectory_np = np.array(self.trajectory)

        print("Computing colors...")
        info_term = self.compute_kernels(self.trajectory, self.vert_traj)
        info_sum = jnp.sum(info_term, axis=0)
        point_colors = jnp.cumsum(info_term, axis=0) / jnp.sum(info_sum, axis=0, keepdims=True)
        point_colors_np = np.array(point_colors)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        all_points = vert_traj_np.reshape(-1, 3)
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        buffer = 0.1
        width = (z_max + buffer) - (z_min - buffer)
        ax.set_xlim(-width/2, width/2)
        ax.set_ylim(-width/2, width/2)
        ax.set_zlim(z_min - buffer, z_max + buffer)

        ax.view_init(elev=24, azim=-76, roll=0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(False)

        scat = ax.scatter(vert_traj_np[0, :, 0], vert_traj_np[0, :, 1], vert_traj_np[0, :, 2],
                          s=1.5, c=point_colors_np[0], cmap='winter', alpha=0.6)

        line, = ax.plot(trajectory_np[0:1, 0], trajectory_np[0:1, 1], trajectory_np[0:1, 2],
                        color='red', linewidth=2, linestyle='--')

        pos, = ax.plot(trajectory_np[0:1, 0], trajectory_np[0:1, 1], trajectory_np[0:1, 2],
                       color='red', marker='^', markersize=8,
                       markeredgecolor='k', markerfacecolor='red', markeredgewidth=1)

        frames_dir = "../figures/tornado/images/"
        idx = 1
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        save_frames = [10, 50, 80, 155]

        def update(t):
            nonlocal idx
            line.set_data(trajectory_np[:t+1, 0], trajectory_np[:t+1, 1])
            line.set_3d_properties(trajectory_np[:t+1, 2])
            pos.set_data([trajectory_np[t, 0]], [trajectory_np[t, 1]])
            pos.set_3d_properties([trajectory_np[t, 2]])
            scat._offsets3d = (vert_traj_np[t, :, 0], vert_traj_np[t, :, 1], vert_traj_np[t, :, 2])
            scat.set_array(point_colors_np[t])
            if t in save_frames:
                ax.set_title(None)
                ax.axis('off')
                plt.savefig(f'{frames_dir}flow_emmd_{idx}.png', dpi=400, bbox_inches='tight')
                print(f"Saved frame {idx} to {frames_dir}flow_emmd_{idx}.png")
                idx += 1
            return scat, line, pos

        if save:
            print(f"Saving animation to {save_path}...")
            ani = matplotlib.animation.FuncAnimation(fig, update, frames=T, interval=50, blit=False)
            ani.save(save_path, writer='ffmpeg', fps=30)
            print("Animation saved.")
        else:
            plt.show()

        plt.close(fig)

    def save_args(self, file_name="output.pkl"):
        output = {
            'trajectory': self.trajectory,
            'vert_traj': self.vert_traj,
            'h': self.h,
            'T': self.T,
            'dt': self.dt
        }
        with open(file_name, "wb") as f:
            pickle.dump(output, f)