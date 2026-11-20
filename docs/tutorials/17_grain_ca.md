# Tutorial 17: Grains and texture with the 3-D cellular automaton

**Goal.** Regrow grains in a laser track from a random substrate, then build two layers with a
rotated hatch and count the grains. About 2 minutes.

**You will learn**
* `TrackSection` from a melt-pool run, KGT tip kinetics;
* substrates, tracks and layers;
* grain statistics.

Units: SI (m, s, K).

## Step 1: the thermal input

The steady thermal history of one track (from the conduction melt pool) is shifted along every
scan vector.

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector
from tpmsfem.process import grain_ca as CA

th = ThermalProperties.from_si([20.0, 1290.0, 1350.0, 3000.0], [9.8, 26.0, 30.0, 30.0], [410.0, 650.0, 720.0, 720.0],
                               8440.0, 2.9e5, 1290.0, 1350.0)
mp = MeltPoolModel(Material("IN625", 208e3, 0.28, thermal=th), h=0.01, length=0.6, width=0.15, depth=0.08,
                   T0=25.0, emissivity=0.4, evaporation=Evaporation())
res = mp.run(Laser(180.0, 0.35, 0.05, depth=0.015), [ScanVector(0.1, 0.0, 0.5, 0.0, speed=800.0)], t_extra=2e-4)
sec = CA.TrackSection.from_meltpool(res, x_section=0.4, speed=800.0, x_start=0.1)
kin = CA.TipKinetics.kgt(c0=4.0, k=0.48, m=-10.5, D=3e-9, gibbs_thomson=3.65e-7)
for V in (0.01, 0.1, 0.5):
    print(f"KGT tip undercooling at {V} m/s: {kin.undercooling(V):.1f} K")
```

Output (from the test run):

```text
KGT tip undercooling at 0.01 m/s: 17.0 K
KGT tip undercooling at 0.1 m/s: 31.1 K
KGT tip undercooling at 0.5 m/s: 45.0 K
```

## Step 2: one track

```python
grid = CA.CAGrid(origin=(0.0, -70e-6, -45e-6), shape=(40, 140, 45), dx=1e-6)
grain, rot = CA.voronoi_substrate(grid, 8e-6, np.random.default_rng(0))       # 8 um substrate grains
ca = CA.GrainCA(grid, grain, rot, kin)
ev = CA.track_events(grid, sec, [CA.Track(start=(-0.6e-3, 0.0), end=(0.4e-3, 0.0), z_top=0.0, t0=0.0)])
out = ca.run(ev).result()
fz = out.melted
print(f"{fz.sum()} melted cells, {len(np.unique(out.grain[fz]))} grains in the pool")
```

Output (from the test run):

```text
160760 melted cells, 100 grains in the pool
```

## Step 3: two layers, 90° rotation

```python
grid = CA.CAGrid(origin=(0.0, 0.0, -45e-6), shape=(60, 60, 85), dx=1e-6)
grain, rot = CA.voronoi_substrate(grid, 8e-6, np.random.default_rng(1),
                                  mask=np.broadcast_to(grid.centres(2)[None, None, :] <= 0.0, grid.shape))
ca = CA.GrainCA(grid, grain, rot, kin, heat_flow_bias=0.0)
t = 0.0
for layer in range(2):
    z_top = (layer + 1) * 20e-6
    tracks = []
    for j, y in enumerate(np.arange(-60e-6, 121e-6, 60e-6)):
        a, b = (-0.6e-3, 0.66e-3) if j % 2 == 0 else (0.66e-3, -0.6e-3)
        p0, p1 = ((a, y), (b, y)) if layer % 2 == 0 else ((y, a), (y, b))
        tracks.append(CA.Track(p0, p1, z_top, t))
        t += 1.3e-3 / 0.8
    ca.run(CA.track_events(grid, sec, tracks))
out = ca.result()
top = out.grain[:, :, -15]
print(f"grains in the top layer: {len(np.unique(top[top >= 0]))}")
```

Output (from the test run):

```text
grains in the top layer: 11
```

* `heat_flow_bias` from 0 (dendritic, crystal axes) to 1 (cellular limit, heat flow).
* IPF colours: `solidification.ipf_color(out.orientations[ids], axis)`; texture fractions:
  `solidification.alignment`.
* Texture needs ≤ 1 µm cells and large blocks ([process_models.md](../process_models.md) §5.5).

## Next
[Tutorial 18: porosity and keyhole maps](18_porosity_keyhole_maps.md).
