# Tutorial 4: A compression test and the effective modulus

**Goal.** Compress a lattice cube between platens, compute its effective Young's modulus,
look at the stresses, apply a surface load, and check mesh convergence. About 1 minute.

**You will learn**
* boundary conditions: fixed nodes, prescribed displacements, surface loads;
* the solvers;
* robust stress statistics and the grid convergence index (GCI).

## Step 1: the model

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize
from tpmsfem.post import results as R

def compression(h, size=4.0, strain=1e-3, density=0.35):
    lat = TPMS("gyroid", cell_size=4.0, kind="sheet", density=density)
    mesh = voxelize(lattice_cube(lat, size, plate=0.5), (0, 0, 0), (size,) * 3, h)
    md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
    z = mesh.points[:, 2]
    bottom, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > size - 1e-9)
    md.fix(bottom)
    md.fix(top, (0, 1), 0.0)
    md.fix(top, (2,), -strain * size)
    info = md.solve(solver="auto")
    E = -md.reactions()[top, 2].sum() / size**2 / strain
    return md, mesh, E, info

md, mesh, E, info = compression(0.2)
print(info)
print(f"E_eff = {E:.0f} MPa ({100 * E / 110e3:.1f} % of the solid)")
```

Output (from the test run):

```text
converged=True inc=1 newton=[0] krylov_its=357 solver=gmg time=0.49s
E_eff = 18996 MPa (17.3 % of the solid)
```

`solver="auto"` picks geometric multigrid (`gmg`) on voxel meshes; `amg`, `pcg` and `direct`
are the alternatives.

## Step 2: stresses

```python
r = R.save_vtu(md, "compression.vtu")
vm = r["vonMises_elem"]
print(R.summarize(vm, "von Mises [MPa]"))
print(f"99th percentile {np.percentile(vm, 99):.1f} MPa vs maximum {vm.max():.1f} MPa")
```

Output (from the test run):

```text
von Mises [MPa]: max 190.9 MPa, p99 157.7, p95 126.8, mean 48.49
99th percentile 157.7 MPa vs maximum 190.9 MPa
```

Voxel stair-steps inflate single-element maxima: report percentiles, or use tetrahedra
(Tutorial 3) for hot spots.

## Step 3: a surface load instead

```python
md2 = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
md2.fix(np.flatnonzero(mesh.points[:, 2] < 1e-9))
faces, face_type, _ = mesh.boundary_faces()
on_top = faces[mesh.points[faces].mean(axis=1)[:, 2] > 4.0 - 1e-9]
md2.add_surface_load(on_top, face_type, total_force=(50.0, 0.0, -400.0))   # N: shear + compression
md2.solve()
print("reaction sum [N]:", np.round(md2.reactions().sum(axis=0), 3))
print(f"top settles by {-md2.displacement()[np.unique(on_top), 2].mean() * 1e3:.2f} um")
```

Output (from the test run):

```text
reaction sum [N]: [-5.e+01 -1.e-03  4.e+02]
top settles by 6.58 um
```

The reactions balance the applied 50 N and −400 N.

## Step 4: mesh convergence

Three meshes refined by a constant ratio give the observed order and Roache's grid convergence
index (GCI). With voxels, also watch the meshed volume fraction: it moves with h, and the
stiffness follows it.

```python
from tpmsfem.bench.metrics import gci

rows = []
for n in (54, 36, 24):                                    # voxels across 4 mm: ratio 1.5, h divides the cube
    md_n, mesh_n, E_n, _ = compression(4.0 / n)
    rows.append((E_n, mesh_n.n_cells * (4.0 / n) ** 3 / 64.0))
    print(f"h = {4.0 / n:.3f} mm: E_eff {E_n:6.0f} MPa, meshed solid fraction {rows[-1][1]:.3f}")
print(gci(f1=rows[0][0], f2=rows[1][0], f3=rows[2][0], r=1.5))
```

Output (from the test run):

```text
h = 0.074 mm: E_eff  18790 MPa, meshed solid fraction 0.516
h = 0.111 mm: E_eff  18451 MPa, meshed solid fraction 0.494
h = 0.167 mm: E_eff  19504 MPa, meshed solid fraction 0.517
{'p': nan, 'f_exact': nan, 'gci_fine': nan, 'note': 'oscillatory or converged; GCI undefined'}
```

Here the result oscillates (the medium grid has the least solid), so the GCI is undefined. Then
report the spread of the grids as the uncertainty (here about ±3 %), refine further, or
compare E divided by the meshed solid fraction. A smooth geometry or a tetrahedral mesh
converges monotonically and gives a proper GCI (Tutorial 12 shows the output for one).

## Check yourself
* Doubling the strain doubles the force (linear elasticity).
* When the GCI is defined, it is the discretisation uncertainty of the fine grid in percent.
* The voxel size must divide the cube exactly, or no nodes lie on the loaded faces.
* Keep the coarsest grid fine enough that the walls stay connected (about 4 voxels across).

## Next
[Tutorial 5: homogenisation](05_homogenisation.md).
