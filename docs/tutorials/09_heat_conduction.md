# Tutorial 9: Transient heat conduction in a lattice

**Goal.** Cool a hot lattice cube on a cold base with surface losses, and follow its
temperature. About 10 seconds.

**You will learn**
* `ThermalProperties` from SI data, latent heat;
* `HeatModel`: fixed temperatures, convection, radiation;
* time stepping.

## Step 1: material

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import voxelize
from tpmsfem.thermal import HeatModel, geometric_steps

th = ThermalProperties.from_si(T=[20.0, 1600.0], k_W_mK=[7.0, 30.0], cp_J_kgK=[530.0, 750.0],
                               rho_kg_m3=4430.0, latent_J_kg=2.9e5, T_solidus=1604.0, T_liquidus=1660.0)
mat = Material("Ti64", E=110e3, nu=0.32, thermal=th)
print(f"k(800 C) = {th.conductivity(800.0):.1f} W/(m K)")
```

Output (from the test run):

```text
k(800 C) = 18.4 W/(m K)
```

## Step 2: a hot lattice on a cold base

```python
mesh = voxelize(box((0, 0, 0), (4, 4, 4)) & TPMS("gyroid", cell_size=4.0, density=0.3), (0, 0, 0), (4, 4, 4), 0.25)
hm = HeatModel(mesh, mat, T_initial=600.0)
hm.fix(np.flatnonzero(mesh.points[:, 2] < 1e-9), 20.0)          # base held at 20 C
hm.set_surface_losses(h=0.01, emissivity=0.3, T_amb=20.0)       # h in mW/(mm^2 K) = 10 W/(m^2 K)
t_prev = 0.0
for dt in geometric_steps(20.0, 0.2, 12):                       # 20 s in 12 growing steps
    hm.step(dt)
    if hm.time - t_prev > 2.0 or hm.time > 19.9:
        print(f"t = {hm.time:5.1f} s: T = {hm.T.min():5.0f} .. {hm.T.max():5.0f} C, mean {hm.T.mean():5.0f} C")
        t_prev = hm.time
```

Output (from the test run):

```text
t =   2.9 s: T =    20 ..   264 C, mean   176 C
t =   5.7 s: T =    20 ..   129 C, mean    88 C
t =   7.8 s: T =    20 ..    82 C, mean    58 C
t =  10.8 s: T =    20 ..    51 C, mean    39 C
t =  14.7 s: T =    20 ..    33 C, mean    28 C
t =  20.0 s: T =    20 ..    25 C, mean    23 C
```

`geometric_steps` grows the step: short at first, where temperatures change fast.

## Check yourself
* The top of the lattice stays hottest: the heat leaves through the base and the thin walls
  slow it down.
* Halve the steps and compare: the result should barely change.

## Next
[Tutorial 10: residual stress, inherent strain](10_inherent_strain.md).
