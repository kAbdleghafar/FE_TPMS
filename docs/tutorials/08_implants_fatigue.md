# Tutorial 8: An implant load case and its fatigue life

**Goal.** Load a lattice-core femoral stem as in the ISO 7206-4 fatigue test, then estimate
fatigue life with strain-life methods, including a mean (residual) stress. About 30 seconds.

**You will learn**
* parametric implants, standard load cases and contact-face loading;
* strain-life (Basquin–Coffin–Manson), Neuber, Morrow and Smith–Watson–Topper.

## Step 1: the stem and the ISO 7206-4 load

ISO 7206-4 pots the stem at a set level and tilts it; the head is loaded with a cyclic force.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import FemoralStem, TPMS
from tpmsfem.loadcases import gait_peak_hip_force, head_contact_faces, iso7206_4_stem
from tpmsfem.mesh import voxelize

stem = FemoralStem(design="lattice_core", lattice=TPMS("gyroid", cell_size=6.0, kind="sheet", density=0.5))
lo, hi = stem.bounds()
mesh = voxelize(stem.field(), lo, hi, 1.5)                 # coarse demo; use h <= wall/4 in practice
F, z_embed = iso7206_4_stem(stem.head_center, F=2300.0)    # force vector [N], potting level [mm]
print(mesh.summary())
print("ISO 7206-4 force [N]:", np.round(F, 1), f"potting at z = {z_embed:.1f} mm")
```

Output (from the test run):

```text
Mesh[hex8] 9,408 cells, 13,659 nodes, 40,977 dofs, bbox [-38.249 -13.5     0.5  ] .. [  9.751  13.5   153.5  ], voxel h=1.5
ISO 7206-4 force [N]: [ -399.4  -354.3 -2237.2] potting at z = 59.7 mm
```

## Step 2: solve

```python
md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
md.fix(mesh.nodes_where(lambda x, y, z: z <= z_embed + 1e-9))
faces, face_type = head_contact_faces(mesh, stem.head_center, stem.head_radius, F)
md.add_surface_load(faces, face_type, total_force=F)
md.solve()
u = md.displacement()[np.unique(faces)].mean(axis=0)
from tpmsfem.post import results as R
vm = R.result_fields(md)["vonMises_elem"]
print(f"head displacement along F {u @ F / np.linalg.norm(F):.4f} mm; von Mises p99 {np.percentile(vm, 99):.0f} MPa")
```

Output (from the test run):

```text
head displacement along F 0.7218 mm; von Mises p99 309 MPa
```

`gait_peak_hip_force(body_weight_N)` gives a physiological peak instead. The full study of
solid vs lattice-core stems is `examples/02_femoral_stem_iso7206.py`.

## Step 3: fatigue life (strain-life)

FE stresses are elastic. At a notch root, Neuber's rule converts the elastic stress amplitude
into an elastic-plastic strain amplitude on the cyclic curve; the strain-life curve then gives
the life. A tensile mean stress, e.g. an AM residual stress, shortens it.

```python
from tpmsfem.post.fatigue import IN625_LPBF_CONCLI2024 as FAT    # as-built LPBF IN625 parameters

sig_el = np.array([200.0, 400.0, 600.0])                   # elastic FE amplitudes [MPa]
s_a, e_a = FAT.neuber(sig_el)
for k in range(len(sig_el)):
    print(f"elastic {sig_el[k]:.0f} MPa -> notch {s_a[k]:.0f} MPa, strain amplitude {100 * e_a[k]:.3f} %")
for corr, sm in (("none", 0.0), ("morrow", 150.0), ("swt", 150.0)):
    N = FAT.life(e_a, sigma_m=sm, correction=corr)
    print(f"{corr:6s} (mean {sm:.0f} MPa): N_f = {np.array2string(N, precision=0)}")
```

Output (from the test run):

```text
elastic 200 MPa -> notch 200 MPa, strain amplitude 0.095 %
elastic 400 MPa -> notch 391 MPa, strain amplitude 0.195 %
elastic 600 MPa -> notch 513 MPa, strain amplitude 0.334 %
none   (mean 0 MPa): N_f = [1.e+12 7.e+09 1.e+08]
morrow (mean 150 MPa): N_f = [6.e+11 3.e+09 8.e+07]
swt    (mean 150 MPa): N_f = [4.e+09 1.e+07 3.e+05]
```

## Check yourself
* The life falls steeply with amplitude; a +150 MPa mean stress cuts it further, most with
  SWT.
* Use parameters for your own material and surface condition: as-built surfaces fatigue far
  worse than machined ones.

## Next
[Tutorial 9: heat conduction](09_heat_conduction.md).
