# Tutorial 1: Getting started

**Goal.** Check the installation, choose the compute backend, learn the unit system, and run a
first complete calculation: the stiffness of a gyroid cube. About 1 minute.

**You will learn**
* how to pick NumPy (CPU) or CuPy (GPU);
* which units each part of TPMSsuite uses;
* the basic chain geometry → mesh → model → solve → results.

## Step 1: install and test

```bash
pip install -e ".[dev]"        # CPU: meshers, multigrid, numba, matplotlib, pytest
pip install -e ".[gpu]"        # optional: CuPy for an NVIDIA GPU
tpmsfem info                   # which backend and device are found
pytest -q                      # the test suite (~6 min)
```

## Step 2: choose the backend

Every heavy computation runs on an array backend: NumPy on the CPU or CuPy on a GPU, with the
same code. `"auto"` picks the GPU when CuPy and a CUDA device are present.

```python
import tpmsfem as tf

name = tf.set_backend("numpy")        # "cupy" on a CUDA machine, "auto" to pick
print("backend:", name)
print(tf.device_info())
```

Output (from the test run):

```text
backend: numpy
numpy 2.5.3 on CPU (x86_64)
```

You can also set it before Python starts: `TPMSFEM_BACKEND=cupy python my_script.py`.

## Step 3: know the units

| part | length | time | stress / pressure | temperature | other |
|---|---|---|---|---|---|
| geometry, meshing, FE, heat, conduction melt pool | mm | s | MPa | °C | force N, density t/mm³, power W |
| phase field, grain CA, free-surface flow, powder, keyhole laws | m | s | Pa | °C (K where stated) | SI throughout |

Thermal data in SI go in with `ThermalProperties.from_si`, which converts them.

## Step 4: a first calculation

A 4 mm gyroid sheet cube (35 % dense) with thin plates on top and bottom, compressed by 0.1 %.
The effective modulus is force / area / strain.

```python
import numpy as np
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

size, strain = 4.0, 1e-3
lattice = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.35)
mesh = voxelize(lattice_cube(lattice, size, plate=0.5), (0, 0, 0), (size,) * 3, h=0.25)
print(mesh.summary())

model = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())        # E = 110 GPa, nu = 0.32
z = mesh.points[:, 2]
bottom, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > size - 1e-9)
model.fix(bottom)                                          # clamp the bottom
model.fix(top, (0, 1), 0.0)                                # top: no sliding
model.fix(top, (2,), -strain * size)                       # top: pushed down 4 um
info = model.solve()
force = -model.reactions()[top, 2].sum()
print(f"converged: {info.converged}, effective modulus {force / size**2 / strain:.0f} MPa")
```

Output (from the test run):

```text
Mesh[hex8] 2,176 cells, 3,635 nodes, 10,905 dofs, bbox [0. 0. 0.] .. [4. 4. 4.], voxel h=0.25
converged: True, effective modulus 21784 MPa
```

## Step 5: save the results

```python
from tpmsfem.post import results as R

r = R.save_vtu(model, "first_model.vtu")                   # open in ParaView
print(R.summarize(r["vonMises_elem"], "von Mises [MPa]"))
```

Output (from the test run):

```text
von Mises [MPa]: max 159.5 MPa, p99 149.1, p95 131.2, mean 49.95
```

## Check yourself
* The effective modulus is about a fifth of the solid's 110 GPa. The 0.5 mm plates add to it;
  the lattice alone (Tutorial 5) gives about 12 % at 32 % density.
* Re-run with `h=0.2`: the modulus changes by a few percent. Tutorial 3 shows how to check
  mesh convergence properly.

## Next
[Tutorial 2: geometry](02_geometry.md).
