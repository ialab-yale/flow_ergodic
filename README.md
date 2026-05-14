# Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields

Code for the paper **"Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields"**. We present a trajectory optimization framework for ergodic coverage over time-varying distributions defined by dynamic particle flows — cattle herds, ocean currents, vortex fields — using Maximum Mean Discrepancy (MMD) minimized via augmented Lagrangian methods with JAX-accelerated gradients.

---

## Installation

```bash
conda create -n temp python=3.12
conda activate temp

# Core
pip install jax jaxlib jaxopt numpy matplotlib scipy

# Geometry (3D experiment)
pip install trimesh open3d

# Ocean / geo data
pip install cartopy cmocean cmcrameri imageio dill svgpath2mpl
```

---

## Experiments

### 1. Cattle Feeding

A herd of cows flows toward feeding sinks under repulsion and damping dynamics. The ergodic agent plans a trajectory that covers the herd's evolving spatial distribution.

**Generate the flow and save data:**
```bash
cd cattle_feeding
conda run -n temp python cattle_flow.py
```

**Run the ergodic planner:**
```bash
conda run -n temp python make_traj.py
```

https://github.com/user-attachments/assets/cattle_flow.mp4

https://github.com/user-attachments/assets/cattle_traj.mp4

---

### 2. Vortex Flow

Particles orbit a vortex center with wall repulsion. The ergodic agent tracks the rotating distribution.

**Generate the vortex flow:**
```bash
cd vortex_flow
conda run -n temp python flow_animate.py
```

**Run the ergodic planner:**
```bash
conda run -n temp python make_traj.py
```

https://github.com/user-attachments/assets/vortex_flow.mp4

https://github.com/user-attachments/assets/vortex_emmd.mp4

[![Vortex flow demo](vortex_flow/thumbnail.png)](vortex_flow/vortex_flow.mp4)


---

### 3. Whale Search (Gulf of Mexico)

Whale pods are advected through real HYCOM ocean current data. The agent plans an observation path that maximizes cumulative encounter probability over the mission horizon.

**Run the ergodic planner:**
```bash
cd whale_search
conda run -n temp python make_traj.py
```

https://github.com/user-attachments/assets/whale_traj.mp4

> Ocean current data (`whale_search/flow_data/flow_data.pkl`) must be present. The flow field is loaded from HYCOM Gulf of Mexico velocity snapshots.

---

## Methods

| File | Description |
|------|-------------|
| `methods/flow_emmd.py` | Core planner — MMD loss, augmented Lagrangian solver, 2D/3D dynamics |
| `methods/augmented_lagrange_wrapper.py` | Augmented Lagrangian wrapper around `jaxopt.LBFGS` |

**`Flow_EMMD`** takes a set of time-varying particle trajectories (the "information sources") and finds the agent path minimizing MMD between cumulative visitation and the empirical particle distribution. Constraints enforce dynamics and control limits.

```python
from methods.flow_emmd import Flow_EMMD

args = {'T': 120, 'h': 0.1, 'dt': 0.25, 'power': 0.5, 'dim': 2}
flow = Flow_EMMD(args, x_0=start_pos)
flow.load_data(flow_args=...)
flow.solve_flow()
```

---

## Project Structure

```
flow_ergodic/
├── methods/
│   ├── flow_emmd.py                  # Core planner
│   └── augmented_lagrange_wrapper.py # Optimizer
├── cattle_feeding/
│   ├── cattle_flow.py                # Herd simulation + animation
│   ├── make_traj.py                  # Ergodic planner for cattle
│   └── cattle_data.npz               # Saved flow data
├── vortex_flow/
│   ├── flow_animate.py               # Vortex simulation + animation
│   ├── make_traj.py                  # Ergodic planner for vortex
│   └── flow_data/
│       └── flow_data.npz             # Saved vortex data
└── whale_search/
    ├── make_traj.py                  # Ergodic planner for whale search
    └── flow_data/
        └── flow_data.pkl             # HYCOM Gulf of Mexico currents
```

---

## License

MIT — see [LICENSE](LICENSE).
