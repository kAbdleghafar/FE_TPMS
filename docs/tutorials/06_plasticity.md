# Tutorial 6: Compressing a lattice past yield

**Goal.** Load a lattice into the plastic range, follow the force–displacement curve, and
find where it yields. About 1 minute.

**You will learn**
* J2 plasticity with linear or tabulated hardening;
* incremental loading and convergence;
* plastic strain fields.

## Step 1: material and model

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import FlowTable, Material
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

mat = Material("Ti64", E=110e3, nu=0.32, yield_stress=900.0, hardening=1000.0)   # MPa
mesh = voxelize(lattice_cube(TPMS("gyroid", cell_size=4.0, density=0.35), 4.0, plate=0.5),
                (0, 0, 0), (4, 4, 4), 0.25)
md = tf.SolidModel(mesh, mat)
z = mesh.points[:, 2]
top = np.flatnonzero(z > 4 - 1e-9)
md.fix(np.flatnonzero(z < 1e-9))
md.fix(top, (0, 1), 0.0)
```

## Step 2: load in steps and record the curve

```python
curve = []
for strain in (0.002, 0.005, 0.01, 0.015, 0.02):
    md.fix(top, (2,), -strain * 4.0)
    info = md.solve(n_inc=2)
    F = -md.reactions()[top, 2].sum()
    peeq = md.plastic_strain()[1].mean(axis=1)
    curve.append((strain, F / 16.0))
    print(f"strain {100 * strain:4.1f} %: nominal stress {F / 16.0:6.1f} MPa, yielded {100 * (peeq > 0).mean():5.1f} % "
          f"of the elements, max PEEQ {100 * peeq.max():.2f} %, converged {info.converged}")
```

Output (from the test run):

```text
strain  0.2 %: nominal stress   42.3 MPa, yielded   0.0 % of the elements, max PEEQ 0.00 %, converged True
strain  0.5 %: nominal stress  105.7 MPa, yielded   1.0 % of the elements, max PEEQ 0.02 %, converged True
strain  1.0 %: nominal stress  157.3 MPa, yielded  21.5 % of the elements, max PEEQ 1.87 %, converged True
strain  1.5 %: nominal stress  169.7 MPa, yielded  30.1 % of the elements, max PEEQ 4.01 %, converged True
strain  2.0 %: nominal stress  175.9 MPa, yielded  34.6 % of the elements, max PEEQ 6.08 %, converged True
```

The nominal stress grows linearly until the walls start to yield, then bends over.

## Step 3: plot it

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

e, s = np.array(curve).T
plt.plot(100 * e, s, "o-")
plt.xlabel("nominal strain [%]")
plt.ylabel("nominal stress [MPa]")
plt.savefig("lattice_compression.png", dpi=100)
```

## Step 4: tabulated, temperature-dependent hardening

```python
flow = FlowTable.from_curves({20.0: ([0.0, 0.05, 0.2], [900.0, 960.0, 1050.0]),
                              500.0: ([0.0, 0.05, 0.2], [620.0, 660.0, 700.0])})
mat_t = Material("Ti64_table", E=110e3, nu=0.32, flow=flow)
for T in (20.0, 250.0, 500.0):                             # interpolated between the curves
    print(f"T = {T:5.0f} C: flow stress at eps_p = 0.05: {float(flow(0.05, T)):.0f} MPa")
```

Output (from the test run):

```text
T =    20 C: flow stress at eps_p = 0.05: 960 MPa
T =   250 C: flow stress at eps_p = 0.05: 816 MPa
T =   500 C: flow stress at eps_p = 0.05: 660 MPa
```

Add a `TemperatureTable` for E(T), ν(T) and the thermal expansion (Tutorial 11).

## Check yourself
* `info.converged` is True at every step; `solve` cuts back failed increments by itself.
* The first yield sits in the bending hinges of the inclined walls.

## Next
[Tutorial 7: where twins or HCP could start](07_stacking_faults_hcp.md).
