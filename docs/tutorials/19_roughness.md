# Tutorial 19: Surface roughness of a lattice

**Goal.** Predict the as-built roughness against the build angle and map it over a gyroid
cell. About 20 seconds.

**You will learn**
* the three contributions: layer staircase, melt-pool rounding, adhered powder;
* area-weighted roughness of a TPMS surface.

## Step 1: roughness against build angle

θ = 0° is an up-skin, 90° a vertical wall, 180° a down-skin (overhang).

```python
import numpy as np
from tpmsfem.process.roughness import SurfaceModel, inclination, roughness_map

sm = SurfaceModel(layer=0.03, bead_radius=0.05, rho_side=40.0)   # mm; rho_side: adhered-powder density
for theta in (0, 45, 90, 135, 160):
    print(f"theta {theta:3d} deg: Sa = {sm.roughness(theta, size=(0.4, 0.4))['Sa']:5.1f} um")
```

Output (from the test run):

```text
theta   0 deg: Sa =   0.0 um
theta  45 deg: Sa =   1.9 um
theta  90 deg: Sa =   1.0 um
theta 135 deg: Sa =   3.9 um
theta 160 deg: Sa =   7.0 um
```

Calibrate `rho_side` on one measurement (e.g. a vertical wall); the angle dependence follows.

## Step 2: over a gyroid cell

```python
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import implicit_to_surface

v, f = implicit_to_surface(box((0, 0, 0), (4, 4, 4)) & TPMS("gyroid", cell_size=4.0, density=0.3), (0, 0, 0), (4, 4, 4), 0.2)
n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
area = 0.5 * np.linalg.norm(n, axis=1)
Sa, _ = roughness_map(n, sm, thetas=np.arange(0, 181, 30), size=(0.3, 0.3))
theta = inclination(n)                                   # build angle of every facet [deg]
down = theta > 90
print(f"area-weighted Sa {(Sa * area).sum() / area.sum():.1f} um; down-skin share of the area {100 * area[down].sum() / area.sum():.0f} %")
```

Output (from the test run):

```text
area-weighted Sa 3.8 um; down-skin share of the area 45 %
```

Down-skins dominate: powder sticks to overhangs and the melt sags into the powder below
(Tutorial 21 shows the keyhole breaking through).

## Next
[Tutorial 20: melt flow](20_free_surface_flow.md).
