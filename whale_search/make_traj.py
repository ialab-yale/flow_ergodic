import os, sys, io
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v3 as iio
import cmocean
from svgpath2mpl import parse_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from methods.flow_emmd import Flow_EMMD

def make_whale_marker():
    svg = (
        "m 28.31493,-36.258404 c -0.963982,-0.437038 -1.923684,-0.61758 -5.794484,-1.090075 "
        "-4.089735,-0.499219 -6.992566,-2.806542 -8.430365,-6.700899 -0.447673,-1.212548 "
        "-0.463734,-1.226989 -1.426577,-1.282699 -2.135735,-0.123574 -5.3106071,-0.507829 "
        "-7.1377361,-0.863881 -1.919877,-0.374125 -1.929862,-0.373688 -2.546851,0.111636 "
        "-1.420503,1.117368 -3.38496599,1.672925 -6.785875,1.919069 -1.042401,0.07544 "
        "-2.435425,0.289579 -3.09561,0.475853 -1.350677,0.381102 -1.715356,0.223538 "
        "-1.394219,-0.602386 0.270583,-0.695904 2.285974,-3.239885 3.264763,-4.121035 "
        "l 0.760217,-0.684381 -2.329338,-1.177999 c -4.3825609,-2.216362 -7.5065999,-4.812673 "
        "-9.6541909,-8.023357 -2.245522,-3.357092 -3.053014,-6.254883 -2.894796,-10.38836 "
        "0.109116,-2.850679 0.422767,-3.615443 1.613384,-3.933864 0.341987,-0.09146 "
        "3.796855,-0.184578 7.6774829,-0.206927 5.992555,-0.03451 7.25727,0.01543 "
        "8.393528,0.331442 1.94852101,0.541915 4.635271,1.91767 6.800691,3.482307 "
        "4.181684,3.0215 7.1906601,4.025568 12.0611851,4.024709 8.794362,-0.0016 "
        "17.679786,-5.396304 20.072278,-12.186814 0.648715,-1.841221 0.524445,-5.045528 "
        "-0.258836,-6.674077 -0.68746,-1.429324 -2.083708,-2.821712 -3.59323,-3.583294 "
        "-2.547251,-1.285137 -4.200263,-2.970691 -4.200263,-4.282951 0,-0.717214 "
        "0.23894,-0.73036 1.564567,-0.08608 1.166391,0.566885 1.365869,0.591881 "
        "4.338735,0.543683 l 3.116133,-0.05052 1.818102,0.880012 1.818103,0.880012 "
        "1.618042,-0.829491 1.618042,-0.829491 h 3.258554 c 2.90838,0 3.375418,-0.05391 "
        "4.346023,-0.50169 1.404215,-0.647818 1.6997,-0.636202 1.6997,0.06682 0,1.44583 "
        "-1.230779,2.671777 -4.268364,4.251605 -1.515708,0.78831 -2.518353,1.512519 "
        "-3.665678,2.647713 -1.939346,1.918841 -2.767594,3.539675 -3.533041,6.913964 "
        "-3.232253,14.248591 -10.060798,24.953977 -18.82586,29.514064 -0.99838,0.519414 "
        "-1.81455,1.014563 -1.813712,1.100332 8.38e-4,0.08577 0.355212,0.396823 "
        "0.787498,0.691233 1.435434,0.977607 3.152183,3.079306 4.877346,5.971002 "
        "1.457979,2.443845 1.696336,2.985481 1.642842,3.733157 -0.03442,0.48101 "
        "-0.16291,0.905116 -0.285545,0.942457 -0.122636,0.03734 -0.668326,-0.134015 "
        "-1.212645,-0.380791 z"
    )
    w = parse_path(svg)
    w.vertices -= w.vertices.mean(axis=0)
    w.vertices = -w.vertices
    return w


def run_animation(traj, vert_traj, dt, lon, lat, u, v, pt_color, labels=None, filename='whale_traj.mp4'):
    T = traj.shape[0]
    whale = make_whale_marker()

    speed = np.sqrt(u**2 + v**2)

    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-95.5, -91.25, 19, 23], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linestyle=':', zorder=3)
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, zorder=3)

    ax.streamplot(lon, lat, u, v, density=2, color='gray', linewidth=1, arrowsize=1,
                  transform=ccrs.PlateCarree(), zorder=2)

    ax.contourf(lon, lat, speed, 100, cmap=cmocean.cm.deep,
                transform=ccrs.PlateCarree(), zorder=1, extend='max', vmin=0, vmax=0.5)

    ax.plot(traj[:, 0], traj[:, 1], '--', color='red', alpha=0.6, linewidth=2,
            transform=ccrs.PlateCarree(), zorder=5)

    traj_line, = ax.plot([], [], 'r-', linewidth=2, transform=ccrs.PlateCarree(), zorder=5)
    traj_pt = ax.scatter([], [], c='red', marker='^', s=100,
                         transform=ccrs.PlateCarree(), zorder=6)

    w_scatter = ax.scatter(vert_traj[0, :, 0], vert_traj[0, :, 1], marker=whale,
                           s=900, c=pt_color[0], cmap='berlin', edgecolors='black',
                           linewidths=0.75, transform=ccrs.PlateCarree(), zorder=4,
                           vmin=0, vmax=7e-3)
    plt.colorbar(w_scatter, ax=ax, label='Whale Observation',
                 orientation='vertical', ticks=[], shrink=0.7, pad=0.15)

    images = []
    print("Generating frames...")
    for t in range(0, T, 2):
        traj_line.set_data(traj[:t, 0], traj[:t, 1])
        traj_pt.set_offsets([traj[t]])
        w_scatter.set_offsets(vert_traj[t])
        w_scatter.set_array(pt_color[t])
        if labels:
            ax.set_title(labels[t])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        images.append(iio.imread(buf))
        sys.stdout.write(f"\rFrame {t // 2 + 1}/{T // 2}")
        sys.stdout.flush()

    if images:
        print(f"\nSaving to {filename}...")
        iio.imwrite(filename, images, fps=5)
        print("Saved.")

    plt.close(fig)

if __name__ == '__main__':
    args = {
        'T': 120,
        'h': 0.1,
        'dt': 0.25,
        'max_dx': 5,
        'N_traj': 1,
        'power': 0.75 * 0.58 * 24 / 60,
        'dim': 2,
        'data_path': 'flow_data/flow_data.pkl',
        'extent': [-95, -91, 19, 23],
    }

    whale_locs = jnp.array([
        [-93,   20  ],
        [-93.1, 20.1],
        [-93.2, 19.9],
        [-92,   21  ],
        [-92.9, 21.1],
        [-92.7, 20.89]
    ])

    labels = []
    day = 0
    start = 3
    times = ['00:00 am', '06:00 am', '12:00 pm', '06:00 pm']
    for i in range(args['T']):
        start += 1
        t_idx = start % len(times)
        if t_idx == 0:
            day = i / len(times)
        elif i == 0:
            day = 0
        labels.append(f"October-{int(day + 1)}, {times[t_idx]}")

    flow = Flow_EMMD(args, x_0=np.array([-93, 19.2]))
    flow.reg_weight = 0.0
    flow.load_data(flow_args={'seed_points': whale_locs})

    lon = np.array(flow.data_lon[::10, ::10])
    lat = np.array(flow.data_lat[::10, ::10])
    u = np.array(flow.data_u[::10, ::10])
    v = np.array(flow.data_v[::10, ::10])

    flow.solve_flow()

    info_term = flow.compute_kernels(flow.trajectory, flow.vert_traj)
    info_sum = jnp.sum(info_term, axis=0)
    pt_color = np.array(jnp.cumsum(info_term, axis=0) / jnp.sum(info_sum, axis=0, keepdims=True))

    run_animation(np.array(flow.trajectory), np.array(flow.vert_traj),
                  flow.dt, lon, lat, u, v, pt_color, labels=labels)
