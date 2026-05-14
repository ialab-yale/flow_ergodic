# Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields

Code for the paper **"Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields"**. We present a trajectory planner for ergodic coverage over time-varying distributions defined by dynamic particle flows (e.g., cattle herds, ocean currents, vortex fields) using Maximum Mean Discrepancy (MMD).

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

![Whale search trajectory](whale_search/whale_traj.mp4)

> Ocean current data (`whale_search/flow_data/flow_data.pkl`) must be present. The flow field is loaded from HYCOM Gulf of Mexico velocity snapshots.

---

## License

MIT — see [LICENSE](LICENSE).
