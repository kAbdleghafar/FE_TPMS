# Tutorial 5: Effective stiffness of a unit cell (homogenisation)

**Goal.** Compute the full 6 × 6 effective stiffness of one periodic TPMS cell, extract the
engineering constants, and compare densities and surfaces. About 1 minute.

**You will learn**
* periodic meshes;
* the stiffness matrix and its moduli;
* how stiffness scales with density (Gibson–Ashby).

## Step 1: one periodic cell

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.homogenize import homogenize
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS
from tpmsfem.mesh import voxelize

L = 2.0
cell = voxelize(TPMS("gyroid", cell_size=L, kind="sheet", density=0.3), (0, 0, 0), (L, L, L), 0.1,
                periodic=(True, True, True))
H = homogenize(tf.SolidModel(cell, TI6AL4V_LPBF.elastic()))
print((H.C / 1e3).round(2), "GPa")
```

Output (from the test run):

```text
[[18.33  7.79  7.79 -0.   -0.    0.  ]
 [ 7.79 18.33  7.79 -0.    0.    0.  ]
 [ 7.79  7.79 18.33  0.    0.   -0.  ]
 [-0.   -0.    0.    5.63 -0.    0.  ]
 [-0.    0.    0.   -0.    5.63 -0.  ]
 [ 0.    0.   -0.    0.   -0.    5.63]] GPa
```

Six unit macro-strains are applied with periodic fluctuations; each gives one column of C.

## Step 2: engineering constants

```python
print(f"E_xx {H.young('xx'):.0f} MPa, G_yz {H.shear('yz'):.0f} MPa, nu_xy {H.poisson('xx', 'yy'):.3f}")
print(f"Zener anisotropy {H.zener():.2f} (1 = isotropic), volume fraction {H.volume_fraction:.3f}")
```

Output (from the test run):

```text
E_xx 13683 MPa, G_yz 5626 MPa, nu_xy 0.298
Zener anisotropy 1.07 (1 = isotropic), volume fraction 0.318
```

## Step 3: density scaling

Gibson–Ashby: E/E_s = C ρⁿ, with n ≈ 1 for stretch-dominated and n ≈ 2 for
bending-dominated lattices.

```python
rows = []
for rho in (0.2, 0.3, 0.4):
    c = voxelize(TPMS("gyroid", cell_size=L, kind="sheet", density=rho), (0, 0, 0), (L,) * 3, 0.1,
                 periodic=(True, True, True))
    h = homogenize(tf.SolidModel(c, TI6AL4V_LPBF.elastic()))
    rows.append((h.volume_fraction, h.young("xx") / 110e3))
    print(f"density {h.volume_fraction:.3f}: E/E_s = {rows[-1][1]:.4f}")
rho_, e_ = np.array(rows).T
n, logC = np.polyfit(np.log(rho_), np.log(e_), 1)
print(f"Gibson-Ashby exponent n = {n:.2f}, C = {np.exp(logC):.2f}")
```

Output (from the test run):

```text
density 0.210: E/E_s = 0.0696
density 0.318: E/E_s = 0.1244
density 0.400: E/E_s = 0.1788
Gibson-Ashby exponent n = 1.46, C = 0.67
```

## Check yourself
* The gyroid sheet is nearly isotropic (Zener close to 1).
* Refine `h` until C changes less than your tolerance (GCI, Tutorial 4).

## Exercises
1. Repeat for `"schwarz_p"` and `"diamond"`; which is stiffest at 30 %?
2. Plates and shells: `components=(0, 1, 5)` with `periodic=(True, True, False)`.

## Next
[Tutorial 6: plasticity](06_plasticity.md).
