# Tutorial 10: AM residual stress with inherent strain

**Goal.** Build a lattice cube layer by layer with the fast inherent-strain method, remove it
from the plate, and compare the stresses before and after. About 20 seconds.

**You will learn**
* super-layers and element birth;
* calibrating the inherent strain;
* stress relaxation on removal.

## Step 1: build

Each super-layer is born strain-free and given an inherent strain ε* (the shrinkage the
process leaves). The part is welded to the plate during the build.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize
from tpmsfem.post import results as R
from tpmsfem.residual.am import AMProcess

mesh = voxelize(lattice_cube(TPMS("gyroid", cell_size=4.0, density=0.35), 4.0, plate=0.5), (0, 0, 0), (4, 4, 4), 0.25)
md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
am = AMProcess(md, layer_height=1.0, inherent_strain=(-0.004, -0.004, -0.002, 0, 0, 0))
am.build(verbose=False)
on_plate = R.result_fields(md)["vonMises_elem"][md.active_host]
print(f"{am.n_layers} layers; von Mises on the plate: median {np.median(on_plate):.0f}, p99 {np.percentile(on_plate, 99):.0f} MPa")
```

Output (from the test run):

```text
4 layers; von Mises on the plate: median 137, p99 722 MPa
```

## Step 2: remove from the plate

Cutting the part off with 3-2-1 support lets it spring back; part of the stress relaxes.

```python
am.remove_from_plate(verbose=False)
after = R.result_fields(md)["vonMises_elem"]
print(f"after removal: median {np.median(after):.0f}, p99 {np.percentile(after, 99):.0f} MPa")
```

Output (from the test run):

```text
after removal: median 60, p99 293 MPa
```

## Step 3: calibration

ε* is not known a priori: fit its scale to one measured distortion (for example a cantilever
deflection) with `residual.am.calibrate_scale`, then predict everything else. The AM-Bench
bridge in `benchmarks/literature/nist_ambench2018_*.py` shows the full workflow.

## Check yourself
* Stresses scale linearly with ε* (elastic material): doubling it doubles them.
* `mode="nodal"` matches the semantics of an Abaqus or CalculiX deck (Tutorial 12).

## Next
[Tutorial 11: thermo-mechanical build](11_thermomechanical_build.md).
