# TPMSsuite user manual

Contributors: Khaled Abdelghafar and Claude (Anthropic).

TPMSsuite is a Python package for triply periodic minimal surface (TPMS) structures. It covers
the whole chain:
* design of the geometry;
* meshing;
* mechanics: stiffness, strength, homogenised properties;
* heat transfer;
* additive-manufacturing (AM) residual stress;
* the laser process itself: melt pool, melt flow, scan paths, porosity, surface roughness;
* the resulting microstructure: solidification conditions, cells and segregation, grains and
  texture;
* benchmarking against commercial solvers and published experiments.

The Python package is imported as `tpmsfem`; the command-line tool is also called `tpmsfem`.

This manual shows **how to perform every calculation**. Each section explains:
* what is computed;
* the inputs and units;
* a minimal working example;
* what to check.

The code blocks build on each other, like cells of a notebook. A few reuse an object from an
earlier section, and say so. Every block is executed by `python tools/manual/check_manual.py`,
except those marked as not run (they need external solvers, measured data or long runs). The validation of each model against
measurements is documented in [process_models.md](process_models.md),
[literature_benchmarks.md](literature_benchmarks.md),
[freesurface_lbm.md](freesurface_lbm.md) and [porosity_maps.md](porosity_maps.md).

Contents:
1. [Installation, backends and units](#1-installation-backends-and-units)
2. [Geometry: TPMS lattices, grading, implants](#2-geometry-tpms-lattices-grading-implants)
3. [Meshing](#3-meshing)
4. [Linear static analysis](#4-linear-static-analysis)
5. [Homogenisation: effective stiffness of a unit cell](#5-homogenisation-effective-stiffness-of-a-unit-cell)
6. [Plasticity](#6-plasticity)
7. [Implants, load cases and fatigue](#7-implants-load-cases-and-fatigue)
8. [Transient heat conduction](#8-transient-heat-conduction)
9. [AM residual stress: inherent strain](#9-am-residual-stress-inherent-strain)
10. [AM residual stress: thermo-mechanical build](#10-am-residual-stress-thermo-mechanical-build)
11. [Export and code-to-code benchmarking](#11-export-and-code-to-code-benchmarking)
12. [Melt pool (conduction model)](#12-melt-pool-conduction-model)
13. [Scan paths for TPMS layers](#13-scan-paths-for-tpms-layers)
14. [Solidification conditions and spacing laws](#14-solidification-conditions-and-spacing-laws)
15. [Cells and segregation: phase field](#15-cells-and-segregation-phase-field)
16. [Grains and texture: 3-D cellular automaton](#16-grains-and-texture-3-d-cellular-automaton)
17. [Porosity: lack of fusion and keyholes](#17-porosity-lack-of-fusion-and-keyholes)
18. [Surface roughness](#18-surface-roughness)
19. [Melt flow with a free surface, powder beds](#19-melt-flow-with-a-free-surface-powder-beds)
20. [EBSD analysis](#20-ebsd-analysis)
21. [Benchmark and study scripts](#21-benchmark-and-study-scripts)
22. [Command line](#22-command-line)

---

## 1. Installation, backends and units

```bash
pip install -e ".[dev]"          # everything for the CPU: meshers, AMG, numba, matplotlib, pytest
pip install -e ".[gpu]"          # adds CuPy (CUDA 12) for the GPU backend
tpmsfem info                     # backend and device
pytest -q                        # test suite
```

**Backends.** The FE solver, the heat solver, the melt-pool models and the phase field run
on NumPy (CPU) or CuPy (GPU) with the same code. Choose the backend in either of two ways:
* with the environment variable `TPMSFEM_BACKEND=numpy|cupy|auto`;
* in Python, as below.

The phase field, the grain CA and the free-surface lattice kernel also have fused numba
kernels for the CPU (`pip install numba`).

```python
import tpmsfem as tf
tf.set_backend("numpy")          # "cupy" on a CUDA machine, "auto" picks the GPU if present
print(tf.device_info())
```

**Units.** The mechanical, thermal and conduction melt-pool parts use one consistent system:

| quantity | unit |
|---|---|
| length | mm |
| force | N |
| stress | MPa |
| time | s |
| mass density | t/mm³ |
| temperature | °C |
| energy | mJ |
| power | mW (laser power in W) |
| conductivity | mW/(mm K), numerically equal to W/(m K) |

Enter SI thermal data with `ThermalProperties.from_si`.

The microstructure and flow modules work in SI (m, s, K):
* the phase field (`process.phasefield`);
* the grain CA (`process.grain_ca`);
* the free-surface flow (`process.freesurface`);
* the powder generator (`process.powder`).

Each section states its units.

---

## 2. Geometry: TPMS lattices, grading, implants

Geometry is implicit: a `Field` is negative inside the solid. Fields combine with booleans:
`a | b` is the union, `a & b` the intersection, `a - b` the difference. They also support
`offset`, `shell`, `translate` and `rotate`.

**TPMS lattices.** The available surfaces are:
* gyroid;
* Schwarz-P;
* diamond;
* IWP;
* Neovius;
* Fischer–Koch S.

Each comes as a `sheet` (a thickened surface) or a `network` (one of the two solid phases).
The density is a relative density in (0, 1), a callable ρ(x, y, z) for a graded lattice, or
an explicit iso-level.

```python
import numpy as np
from tpmsfem.geometry import TPMS, SURFACES, box, cylinder, level_for_density, relative_density
from tpmsfem.geometry.tpms import sheet_with_thickness

print(sorted(SURFACES))
gyr = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.30)          # uniform sheet, 30 % dense
net = TPMS("diamond", cell_size=3.0, kind="network", density=0.25)       # strut-like network
graded = TPMS("schwarz_p", cell_size=4.0, kind="sheet",
              density=lambda x, y, z: 0.2 + 0.2 * np.clip(z / 20.0, 0.0, 1.0))   # graded along z
walls = sheet_with_thickness("gyroid", 4.0, thickness=0.3)              # prescribed wall thickness [mm]
c = level_for_density("gyroid", "sheet", 0.3)                            # iso-level for a density
print(f"level {c:.3f} -> density {relative_density('gyroid', 'sheet', c):.3f}")
part = (cylinder((0, 0, 0), (0, 0, 20), 8.0) & gyr) | box((-8, -8, 0), (8, 8, 1.0))   # lattice on a base
print(f"nominal density {gyr.nominal_relative_density():.3f}")
```

* **Graded wall thickness.** `sheet_with_thickness(surface, cell_size, thickness)` takes a
  scalar or a callable t(x, y, z). It is accurate to a few percent, where the plain `TPMS`
  sheet scales the level and misses thin walls by 10–20 %.
* **Orientation.** `TPMS(..., origin=..., rotation=R)` places and orients the lattice;
  `rotation_matrix(axis, angle_deg)` builds R.

**Implants.**
* `FemoralStem(design="solid" | "lattice_core" | ..., lattice=TPMS(...))` gives a parametric
  hip stem; `.field()`, `.bounds()`, `.head_center` and `.head_radius` describe it.
* `AcetabularCup(porous_thickness=..., lattice=...)` gives a cup with a porous shell.
* `lattice_cube(lattice, size, plate=...)` gives a test cube with platens.

---

## 3. Meshing

**Voxel hex8 meshes** are the default: robust for any TPMS, fast, and supported by the
geometric multigrid solver.

```python
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import voxelize, implicit_to_surface, write_stl, surface_area_volume

lat = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.3)
block = box((0, 0, 0), (8, 8, 8)) & lat
mesh = voxelize(block, (0, 0, 0), (8, 8, 8), h=0.2)        # h: voxel size [mm]
print(mesh.summary())
mesh.save("gyroid_block.vtu")                              # ParaView; also .inp, .msh ... via meshio
```

`voxelize` options:
* `subsample` (supersampled classification);
* `keep="largest"` (drop floating islands);
* `material_fields=[(id, field), ...]` (several materials);
* `periodic=(True, True, True)` for homogenisation (Section 5).

**Resolution.** Use at least 4 voxels across the thinnest wall. Check with a refinement
study: `tpmsfem.bench.metrics.gci` gives Roache's grid convergence index (Section 11).

**Surface meshes and tetrahedra.** Marching cubes gives an STL. Gmsh, TetGen or fTetWild
turn it into tet4 or tet10 meshes.

```python
verts, faces = implicit_to_surface(block, (0, 0, 0), (8, 8, 8), h=0.1)
write_stl("gyroid_block.stl", verts, faces)
area, volume = surface_area_volume(verts, faces)
print(f"surface {area:.1f} mm^2, volume {volume:.1f} mm^3 (relative density {volume / 8**3:.3f})")
```

<!-- not run: needs a few minutes and a clean surface for the tetrahedral mesher -->
```python
from tpmsfem.mesh.adapters import gmsh_tetmesh, tetgen_tetmesh, ftetwild_tetmesh, tet_quality
tet = gmsh_tetmesh(verts, faces, size=0.3, order=2)          # tet10; order=1 for tet4
print(tet.summary(), tet_quality(tet))
```

**Imported meshes.**
* `read_mesh(path)` reads anything meshio reads (.msh, .inp, .vtu, .med, ...).
* `voxelize_stl(path, h)` voxelises an STL, e.g. from CAD.

---

## 4. Linear static analysis

A `SolidModel` takes a mesh and a `Material` (or `{material_id: Material}`). You then:
* fix degrees of freedom (`fix(nodes, comps, value)`, with prescribed displacements);
* add loads: `add_nodal_force`, or `add_surface_load(faces, face_type, total_force= |
  traction= | pressure=)`;
* solve.

The solvers are:
* `gmg`: geometric multigrid PCG, for voxel meshes on CPU or GPU;
* `amg`: pyamg;
* `pcg`: Jacobi-preconditioned;
* `direct`: sparse LU;
* `auto`: chooses among them.

Example: a compression test of a gyroid cube between platens, and its effective modulus.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize
from tpmsfem.post import results as R

size, strain = 4.0, 1e-3
lat = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.35)
mesh = voxelize(lattice_cube(lat, size, plate=0.5), (0, 0, 0), (size,) * 3, 0.2)
md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())          # E = 110 GPa, nu = 0.32
z = mesh.points[:, 2]
bottom, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > size - 1e-9)
md.fix(bottom)                                            # clamp
md.fix(top, (0, 1), 0.0)
md.fix(top, (2,), -strain * size)                         # prescribed shortening
info = md.solve(solver="auto")
F = -md.reactions()[top, 2].sum()
print(info)
print(f"E_eff = {F / size**2 / strain:.0f} MPa")
r = R.save_vtu(md, "compression.vtu")                     # U, stresses, von Mises, principal ...
print(R.summarize(r["vonMises_elem"], "von Mises"))
```

A surface load instead of a prescribed displacement:

```python
md2 = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
md2.fix(bottom)
faces, face_type, _ = mesh.boundary_faces()
on_top = faces[mesh.points[faces].mean(axis=1)[:, 2] > size - 1e-9]
md2.add_surface_load(on_top, face_type, total_force=(50.0, 0.0, -400.0))   # N
md2.solve()
print("reaction sum", np.round(md2.reactions().sum(axis=0), 3))
```

**Outputs.**
* `displacement()` and `reactions()`;
* `stress()`, the integration-point Voigt stresses in MPa;
* `strain_energy()`;
* `post.results.result_fields(md)`, with nodal and element averages, von Mises and principal
  stresses.

`save_vtu` writes them for ParaView. Voxel stair-steps inflate single-element maxima, so
report percentiles (e.g. the 99th) or use a tetrahedral mesh for hot spots.

---

## 5. Homogenisation: effective stiffness of a unit cell

Mesh one unit cell with periodic axes. `homogenize` applies six unit macro-strains with
periodic fluctuations and returns the effective stiffness matrix C, together with the moduli
derived from it.

```python
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
print(f"E_xx {H.young('xx'):.0f} MPa, G_yz {H.shear('yz'):.0f} MPa, nu_xy {H.poisson('xx', 'yy'):.3f}, "
      f"Zener {H.zener():.2f}, volume fraction {H.volume_fraction:.3f}")
```

* **Plates and shells.** Use `components=(0, 1, 5)` with `periodic=(True, True, False)` for
  in-plane properties.
* **Convergence.** Refine h until C changes by less than your tolerance (Section 11, GCI).

---

## 6. Plasticity

Plasticity is J2 with isotropic hardening, either linear (`yield_stress`, `hardening`) or a
tabulated, temperature-dependent flow curve (`FlowTable`). The return mapping is exact and
uses a consistent tangent. For hex8 the B-bar formulation is switched on automatically.
Apply the load in increments.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import Material, FlowTable
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

mat = Material("Ti64", E=110e3, nu=0.32, yield_stress=900.0, hardening=1000.0)
mesh = voxelize(lattice_cube(TPMS("gyroid", cell_size=4.0, density=0.35), 4.0, plate=0.5),
                (0, 0, 0), (4, 4, 4), 0.25)
md = tf.SolidModel(mesh, mat)
z = mesh.points[:, 2]
top = np.flatnonzero(z > 4 - 1e-9)
md.fix(np.flatnonzero(z < 1e-9))
md.fix(top, (0, 1), 0.0)
md.fix(top, (2,), -0.06)                                   # 1.5 % compression
info = md.solve(n_inc=5)
eps_p, peeq = md.plastic_strain()
print(info)
print(f"max PEEQ {peeq.max():.4f}, force {-md.reactions()[top, 2].sum():.0f} N")
```

* **Tabulated hardening.** Use `FlowTable.from_curves({T: (eps_p, sigma), ...})` and pass it
  as `Material(..., flow=...)`. Add `TemperatureTable` for E(T), ν(T) and α(T).
* **Robustness.** `solve` cuts back failed increments and uses line search. Check
  `info.converged`.

---

## 7. Implants, load cases and fatigue

ISO 7206-4 stem fatigue loading of a lattice-core femoral stem is shown below.
`loadcases.gait_peak_hip_force` gives a gait-peak hip force.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, FemoralStem
from tpmsfem.loadcases import head_contact_faces, iso7206_4_stem
from tpmsfem.mesh import voxelize

stem = FemoralStem(design="lattice_core", lattice=TPMS("gyroid", cell_size=6.0, kind="sheet", density=0.5))
lo, hi = stem.bounds()
mesh = voxelize(stem.field(), lo, hi, 1.5)                 # coarse demo; use h <= wall/4 in practice
F, z_embed = iso7206_4_stem(stem.head_center, F=2300.0)    # force vector [N], potting level
md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
md.fix(mesh.nodes_where(lambda x, y, z: z <= z_embed + 1e-9))
faces, face_type = head_contact_faces(mesh, stem.head_center, stem.head_radius, F)
md.add_surface_load(faces, face_type, total_force=F)
md.solve()
u = md.displacement()[np.unique(faces)].mean(axis=0)
print(f"head displacement along F {u @ F / np.linalg.norm(F):.4f} mm")
```

`examples/02_femoral_stem_iso7206.py` runs the full comparison of the solid and lattice-core
stems, with stress percentiles and Abaqus/CalculiX decks. `examples/03_cup_am_residual_stress.py`
runs the porous cup.

### Fatigue (strain-life)

`post.fatigue.StrainLife` provides:
* the cyclic Ramberg–Osgood curve;
* Basquin–Coffin–Manson strain-life;
* Morrow or Smith–Watson–Topper mean-stress corrections (e.g. the AM residual stress as the
  mean stress);
* Neuber's rule for notch-root amplitudes from an elastic FE amplitude.

The parameters must match your material and surface condition. `IN625_LPBF_CONCLI2024`
holds published as-built LPBF IN625 values as a starting point.

```python
from tpmsfem.post.fatigue import IN625_LPBF_CONCLI2024 as IN625_FAT

sig_el = np.array([200.0, 400.0, 600.0])               # elastic FE stress amplitudes [MPa]
s_a, e_a = IN625_FAT.neuber(sig_el)                      # notch-root stress and strain amplitudes
for corr, sm in (("none", 0.0), ("morrow", 150.0), ("swt", 150.0)):
    N = IN625_FAT.life(e_a, sigma_m=sm, correction=corr)   # e.g. +150 MPa residual stress as the mean
    print(f"{corr:6s}: N_f = {np.array2string(N, precision=0)}")
```

---

## 8. Transient heat conduction

`HeatModel` solves transient conduction on the same meshes. It uses the enthalpy method,
with latent heat between solidus and liquidus. It handles:
* temperature-dependent k and c_p;
* convection and radiation on the free surfaces;
* element birth, for building layer by layer.

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import voxelize
from tpmsfem.thermal import HeatModel, geometric_steps

th = ThermalProperties.from_si(T=[20.0, 1600.0], k_W_mK=[7.0, 30.0], cp_J_kgK=[530.0, 750.0],
                               rho_kg_m3=4430.0, latent_J_kg=2.9e5, T_solidus=1604.0, T_liquidus=1660.0)
mat = Material("Ti64", E=110e3, nu=0.32, thermal=th)
mesh = voxelize(box((0, 0, 0), (4, 4, 4)) & TPMS("gyroid", cell_size=4.0, density=0.3),
                (0, 0, 0), (4, 4, 4), 0.25)
hm = HeatModel(mesh, mat, T_initial=600.0)
hm.fix(np.flatnonzero(mesh.points[:, 2] < 1e-9), 20.0)       # cold base
hm.set_surface_losses(h=0.01, emissivity=0.3, T_amb=20.0)   # h in mW/(mm^2 K) = 10 W/(m^2 K)
for dt in geometric_steps(20.0, 0.2, 12):                   # 20 s in 12 growing steps
    hm.step(dt)
print(f"after {hm.time:.0f} s: T = {hm.T.min():.0f} .. {hm.T.max():.0f} C")
```

---

## 9. AM residual stress: inherent strain

This is the fast layer-by-layer method. The part is cut into super-layers. Each layer is
born strain-free and given a calibrated inherent strain ε*. After the build, the part is
removed from the plate with 3-2-1 support and springs back. Calibrate ε* against one
measured distortion (`residual.am.calibrate_scale`).

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.post import results as R
from tpmsfem.mesh import voxelize
from tpmsfem.residual.am import AMProcess

mesh = voxelize(lattice_cube(TPMS("gyroid", cell_size=4.0, density=0.35), 4.0, plate=0.5),
                (0, 0, 0), (4, 4, 4), 0.25)
md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
am = AMProcess(md, layer_height=1.0, inherent_strain=(-0.004, -0.004, -0.002, 0, 0, 0))
am.build(verbose=False)
on_plate = R.result_fields(md)["vonMises_elem"][md.active_host]
am.remove_from_plate(verbose=False)
print(f"{am.n_layers} layers; von Mises p99 on the plate {np.percentile(on_plate, 99):.0f} MPa, "
      f"after removal {np.percentile(R.result_fields(md)['vonMises_elem'], 99):.0f} MPa")
```

Use `mode="nodal"` to match the semantics of an Abaqus or CalculiX deck (Section 11). With
plasticity, pass the elastic-plastic material and `n_inc > 1`.

---

## 10. AM residual stress: thermo-mechanical build

`ThermoMechanicalAM` couples `HeatModel` and `SolidModel` sequentially, one super-layer at a
time:
* the new layer is deposited hot;
* heat is conducted, convected and radiated;
* temperature-dependent elasto-plasticity is solved at selected steps;
* the build then cools down and is removed from the plate.

The material needs `thermal` properties and a `TemperatureTable`.

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.fem.materials import Material, TemperatureTable, ThermalProperties
from tpmsfem.geometry import box
from tpmsfem.mesh import voxelize
from tpmsfem.residual.thermomech import ThermoMechanicalAM
from tpmsfem.thermal import HeatModel

th = ThermalProperties.from_si([20.0, 1600.0], [7.0, 30.0], [530.0, 750.0], 4430.0, 2.9e5, 1604.0, 1660.0)
temp = TemperatureTable(T=[20.0, 800.0, 1600.0], E=[110e3, 70e3, 10e3], alpha=[8.6e-6, 10e-6, 11e-6],
                        T_ref=20.0, T_cap=1604.0)
mat = Material("Ti64", E=110e3, nu=0.32, yield_stress=900.0, hardening=1000.0, temperature=temp, thermal=th)
mesh = voxelize(box((0, 0, 0), (3, 3, 2)), (0, 0, 0), (3, 3, 2), 0.25)
md = tf.SolidModel(mesh, mat, T_initial=100.0)
hm = HeatModel(mesh, mat, T_initial=100.0)
hm.set_surface_losses(h=0.01, emissivity=0.3, T_amb=100.0)
tm = ThermoMechanicalAM(md, hm, layer_height=0.5, T_deposit=1650.0, layer_time=2.0, n_steps=3, T_plate=100.0)
tm.build(verbose=False)
tm.cool_down(T_room=20.0, n_steps=3)
tm.remove_from_plate(verbose=False)
s = md.stress().mean(axis=1)
print(f"{tm.n_layers} layers; residual sigma_xx {s[:, 0].min():.0f} .. {s[:, 0].max():.0f} MPa")
```

The full-scale application, the AM-Bench IN625 bridge, is
`benchmarks/literature/nist_ambench2018_tm.py`.

---

## 11. Export and code-to-code benchmarking

`io.abaqus.write_inp` writes the identical discrete problem as an Abaqus or CalculiX deck:
* nodes and elements;
* constraints;
* consistent nodal loads;
* eigenstrains;
* AM layer steps with element birth.

`io.calculix.run_ccx` runs CalculiX, and `bench.compare` compares the results field by field.

```python
from tpmsfem.io.abaqus import write_inp                  # md: the model of Section 10
deck_abq = write_inp(md, "residual_abaqus.inp", dialect="abaqus", title="TPMSsuite export")
deck_ccx = write_inp(md, "residual_ccx.inp", dialect="calculix")
print(deck_abq, deck_ccx)
```

<!-- not run: needs the CalculiX executable ccx (installed in CI) -->
```python
from tpmsfem.bench.compare import compare
from tpmsfem.io.calculix import frd_field, read_dat_stress, read_frd, run_ccx
r = run_ccx(deck_ccx, threads=4)
U_ref = frd_field(read_frd(r["frd"]), "DISP", md.n_nodes)
S_ref = read_dat_stress(r["dat"], md.ne, md.nq).mean(axis=1)
print(compare(md.displacement(), md.stress().mean(axis=1), U_ref, S_ref, active=md.active_host))
```

**Mesh convergence.** `gci` takes three results on grids refined by a ratio r. It returns
the observed order, the extrapolated value and the grid convergence index.

```python
from tpmsfem.bench.metrics import gci
print(gci(f1=10_310.0, f2=10_520.0, f3=10_960.0, r=1.5))   # fine, medium, coarse
```

`examples/04_benchmark_vs_calculix.py` runs four cases (lattice, stem, AM cup, tet10 sphere)
against CalculiX. [benchmarking.md](benchmarking.md) describes the Abaqus workflow.

---

## 12. Melt pool (conduction model)

`MeltPoolModel` is an explicit enthalpy finite-volume solver on a voxel grid (units mm, s,
W). Its features are:
* a Gaussian or top-hat laser with an absorption depth;
* any scan path;
* an optional powder layer;
* latent heat, evaporation (`Evaporation`), radiation and convection;
* anisotropic liquid conductivity, to mimic convection.

It records the peak temperature, the melting and solidification times, and G, R and the
cooling rate at solidification.

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector, raster
from tpmsfem.process.meltpool import fusion_zone, pool_length

th = ThermalProperties.from_si(T=[20.0, 1290.0, 1350.0, 3000.0], k_W_mK=[9.8, 26.0, 30.0, 30.0],
                               cp_J_kgK=[410.0, 650.0, 720.0, 720.0], rho_kg_m3=8440.0, latent_J_kg=2.9e5,
                               T_solidus=1290.0, T_liquidus=1350.0)
IN625 = Material("IN625", E=208e3, nu=0.28, thermal=th)
mp = MeltPoolModel(IN625, h=0.01, length=0.6, width=0.15, depth=0.08, T0=25.0, emissivity=0.4,
                   evaporation=Evaporation())                    # half-symmetric about y = 0
res = mp.run(Laser(power=180.0, absorptivity=0.35, radius=0.05, depth=0.015),
             [ScanVector(0.1, 0.0, 0.5, 0.0, speed=800.0)], t_extra=2e-4)
w, d = fusion_zone(res, x_section=0.4)
print(f"melt pool width {1e3 * w:.0f} um, depth {1e3 * d:.0f} um, peak {res.T_peak.max():.0f} C, "
      f"{res.steps} steps")
```

* **Calibration.** Fit `absorptivity` and `depth` to one measured cross-section; NIST AM-Bench
  case B is shown in `benchmarks/literature/nist_ambench2018_tracks.py`.
* **Hatches.** `raster(x0, x1, y0, y1, hatch, speed)` builds a bidirectional hatch. Set
  `powder=` for a powder layer and `symmetric_y=False` for multiple tracks.
* **Part geometry.** `set_part_geometry(field)` makes the substrate the part built so far
  (powder elsewhere), e.g. one layer of a TPMS lattice.
* **Preheat.** `set_temperature(T)` sets a preheat field.

---

## 13. Scan paths for TPMS layers

`slice_scan_path` slices an implicit part at height z. It returns contour and hatch vectors
(with laser-off jumps), ready for `MeltPoolModel.run` and the grain CA.

```python
from tpmsfem.geometry.tpms import sheet_with_thickness
from tpmsfem.process.scanpath import slice_scan_path

gyroid = sheet_with_thickness("gyroid", 4.0, lambda x, y, z: 0.2 + 0.1 * x)   # graded walls [mm]
path, stats = slice_scan_path(gyroid, z=1.3, bounds=(0.0, 2.0, 0.0, 2.0), hatch=0.085, speed=750.0,
                              contour_offset=0.035, angle_deg=67.0)
print(stats)
```

---

## 14. Solidification conditions and spacing laws

From a melt-pool result:
* **G and R.** The thermal gradient G and the solidification velocity R at the last liquidus
  crossing, with the cooling rate and their statistics.
* **Spacing.** Primary spacing laws, Kurz–Fisher and Hunt.
* **Columnar-to-equiaxed transition.** Hunt's criterion.
* **Grains.** A reduced grain-growth model (epitaxy, competitive selection) with IPF colours.
* **Export.** The thermal history in ExaCA format.

```python
from tpmsfem.process import solidification as S

ok, G, R, cr = S.gr_fields(res)                       # res: the melt pool of Section 12; SI units
print({k: v["p50"] for k, v in S.summary(res).items() if isinstance(v, dict) and "p50" in v})
sc = S.SolidificationConstants(freezing_range=60.0, D_liquid=3e-9, gibbs_thomson=3.65e-7, partition=0.48)
lam = S.primary_spacing(G[ok], R[ok], sc, model="hunt")
print(f"Hunt spacing median {np.median(lam):.2f} um; equiaxed fraction "
      f"{(S.hunt_cet(G[ok], R[ok], sc) == 2).mean():.3f}")
gr = S.grow_grains(res, substrate_grain_size=0.03, seed=0)
print(S.grain_stats(gr, res, x_section=0.4))
n = S.export_exaca(res, "exaca_track.csv")
```

---

## 15. Cells and segregation: phase field

`DirectionalPF` is the 2-D quantitative phase field of a dilute binary alloy in directional
solidification (Karma–Echebarria, anti-trapping current, frozen temperature). It runs at the
melt pool's G and R (SI units) and resolves:
* the cells or dendrites;
* their spacing;
* the tip compositions;
* the solute segregation.

It runs on NumPy/numba or CuPy.

```python
import numpy as np
from tpmsfem.process import phasefield as PF

alloy = PF.IN625_NB                                   # IN625 as Ni-4 wt% Nb (Ghosh et al.)
fast = alloy.at_velocity(0.05, V_D=1.0)               # solute trapping, continuous growth model
print(f"k {alloy.k:.2f} -> {fast.k:.3f}, freezing range {alloy.freezing_range:.1f} -> {fast.freezing_range:.1f} K")
pf = PF.DirectionalPF(fast, G=1e7, V=0.05, width_m=1.2e-6, height_m=0.6e-6, W0_m=10e-9, seed=0)
r = pf.run(1.0e-6 / 0.05, adapt_dt=True)              # 1 um of growth (~20 s on 4 cores)
lam, n = r.spacing()
tips = r.tips()
print(f"{n:.0f} cells, spacing {lam * 1e6:.2f} um, tip c_s* = {np.median([t['cs'] for t in tips]) * fast.c_inf:.2f} wt%"
      if tips else "planar front still breaking up")
print(r.segregation())
```

* **Other alloys.** Use `DiluteAlloy.from_slope(c_inf, k, m, D=..., gibbs_thomson=...)`.
* **Interface width.** Choose `W0` so that D/V ≥ 5–10 W0; check convergence by halving W0.
  The cost scales as W0⁻⁵.
* **Long runs.** Use `adapt_dt=True` (1.7× faster). See
  `benchmarks/literature/pf_path_trapping.py` for checkpointing.
* **Checks.** `mullins_sekerka_rate`, `constitutional_velocity` and
  `absolute_stability_velocity` verify the model and the regime. The path study is in
  [process_models.md](process_models.md) §5.3.

---

## 16. Grains and texture: 3-D cellular automaton

`process.grain_ca` grows grains in 3-D. It uses SI units.
* **Envelopes.** Each grain's octahedral dendrite envelopes advance along its own <100> axes,
  at the KGT tip velocity.
* **Decentring.** Decentred octahedra keep the growth along the crystal axes, not the grid.
* **Capture.** Neighbouring liquid cells are captured when an envelope reaches them.
* **Nucleation and epitaxy.** Nucleation in the undercooled liquid is optional; unmelted
  cells seed the melt (epitaxy).
* **Thermal input.** The steady thermal history of one track (`TrackSection`, from a
  `MeltPoolModel` run) is shifted along every scan vector of any strategy (`track_events`).
  Pool ends are rounded where the laser switches on or off.
* **Growth mode.** `heat_flow_bias=α` turns the growth axis towards the heat flow: 0 is
  dendritic, 1 the cellular limit.

```python
import numpy as np
from tpmsfem.process import grain_ca as CA

sec = CA.TrackSection.from_meltpool(res, x_section=0.4, speed=800.0, x_start=0.1)   # from Section 12
kin = CA.TipKinetics.kgt(c0=4.0, k=0.48, m=-10.5, D=3e-9, gibbs_thomson=3.65e-7)
print(f"KGT tip undercooling at 0.1 m/s: {kin.undercooling(0.1):.1f} K")
grid = CA.CAGrid(origin=(0.0, -70e-6, -45e-6), shape=(40, 140, 45), dx=1e-6)     # 40 x 140 x 45 um
grain, rot = CA.voronoi_substrate(grid, 8e-6, np.random.default_rng(0))        # random 8 um substrate
ca = CA.GrainCA(grid, grain, rot, kin)
ev = CA.track_events(grid, sec, [CA.Track(start=(-0.6e-3, 0.0), end=(0.4e-3, 0.0), z_top=0.0, t0=0.0)])
out = ca.run(ev).result()
fz = out.melted
print(f"{fz.sum()} melted cells, {len(np.unique(out.grain[fz]))} grains in the pool, {out.steps} steps")
```

**Many tracks and layers.** Call `run` once per layer with that layer's tracks; the state
persists. Here two layers of 20 µm with 60 µm hatch are built, rotated by 90°:

```python
grid = CA.CAGrid(origin=(0.0, 0.0, -45e-6), shape=(80, 80, 85), dx=1e-6)
grain, rot = CA.voronoi_substrate(grid, 8e-6, np.random.default_rng(1),
                                  mask=np.broadcast_to(grid.centres(2)[None, None, :] <= 0.0, grid.shape))
ca = CA.GrainCA(grid, grain, rot, kin, heat_flow_bias=0.0)
t = 0.0
for layer in range(2):
    z_top = (layer + 1) * 20e-6
    tracks = []
    for j, y in enumerate(np.arange(-60e-6, 141e-6, 60e-6)):
        a, b = (-0.6e-3, 0.68e-3) if j % 2 == 0 else (0.68e-3, -0.6e-3)   # bidirectional, overrunning
        p0, p1 = ((a, y), (b, y)) if layer % 2 == 0 else ((y, a), (y, b))
        tracks.append(CA.Track(p0, p1, z_top, t))
        t += 1.3e-3 / 0.8
    ca.run(CA.track_events(grid, sec, tracks))
out = ca.result()
top = out.grain[:, :, -15]
print(f"grains in the top layer: {len(np.unique(top[top >= 0]))}")
```

* **Analysis.** `solidification.ipf_color(out.orientations[ids], axis)` gives IPF colours;
  `solidification.alignment` gives texture fractions.
* **Applications.** See `benchmarks/literature/ca_multilayer.py` (blocks, texture, grain
  width) and `cantor_gyroid_ca.py` (scan paths clipped to a TPMS lattice).
* **Resolution.** Texture needs ≤ 1 µm cells and large blocks; grain width is robust at
  2 µm. See [process_models.md](process_models.md) §5.4–§5.5 and §7.4.

---

## 17. Porosity: lack of fusion and keyholes

From melt-pool width W and depth D (D measured from the top of the powder layer), hatch h
and layer L:
* **Lack of fusion.** The criterion of Tang et al. (2017), (h/W)² + (L/D)² ≤ 1 for full
  fusion, and the geometric unfused fraction.
* **Keyholes.** The keyhole number Ke of Gan et al. (2021): conduction below ~1.4, keyhole
  above ~6, pores likely above ~16–30.

```python
from tpmsfem.process.defects import keyhole_number, lof_criterion, lof_porosity, volumetric_energy_density

W, D, hatch, layer = 97.0, 54.0, 85.0, 30.0           # um (W, D from a melt-pool run with powder)
print(f"criterion {lof_criterion(W, D, hatch, layer):.2f}, unfused {100 * lof_porosity(W, D, hatch, layer, cv=0.1, samples=4):.2f} %")
Ke = keyhole_number(P=200.0, v=0.75, absorptivity=0.35, radius=35e-6, rho=7980.0, cp=600.0,
                    T_l=1340.0, T0=80.0, alpha=5e-6)   # SI
print(f"Ke = {Ke:.1f}; VED = {volumetric_energy_density(200.0, 0.75, 85e-6, 30e-6) / 1e9:.0f} J/mm^3")
```

Full process maps over power and speed come from `benchmarks/literature/porosity_maps.py`.
Validation against Locatelli et al. (2026) is in [porosity_maps.md](porosity_maps.md).

---

## 18. Surface roughness

`SurfaceModel` predicts the as-built roughness against the build angle θ (0° up-skin, 90°
vertical, 180° down-skin). It combines three effects:
* the layer staircase;
* melt-pool rounding;
* adhered powder particles (calibrate `rho_side` on one measurement).

`roughness_map` maps it onto any surface, e.g. a TPMS triangulation.

```python
import numpy as np
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import implicit_to_surface
from tpmsfem.process.roughness import SurfaceModel, roughness_map

sm = SurfaceModel(layer=0.03, bead_radius=0.05, rho_side=40.0)
for theta in (0, 45, 90, 135):
    print(theta, f"Sa = {sm.roughness(theta, size=(0.4, 0.4))['Sa']:.1f} um")
v, f = implicit_to_surface(box((0, 0, 0), (4, 4, 4)) & TPMS("gyroid", cell_size=4.0, density=0.3), (0, 0, 0), (4, 4, 4), 0.2)
n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
area = 0.5 * np.linalg.norm(n, axis=1)
Sa, _ = roughness_map(n, sm, thetas=np.arange(0, 181, 30), size=(0.3, 0.3))
print(f"area-weighted Sa of the gyroid cell: {(Sa * area).sum() / area.sum():.1f} um")
```

---

## 19. Melt flow with a free surface, powder beds

`FreeSurfaceMeltPool` is a free-surface lattice Boltzmann melt-flow solver (SI units). Its
physics:
* D3Q19 TRT flow, Körner free surface, Darcy damping in the mushy zone;
* an enthalpy heat solver on the same cells;
* Marangoni stress (constant dσ/dT or `SahooSurfaceTension`), recoil pressure, evaporation
  and radiation;
* a powder bed from `process.powder`.

With `energy_transport="lbm"`, heat moves exactly with the mass; any remainder is booked as
`energy_residual`. Production runs need 2.5–6 µm cells, i.e. a GPU. The example is a
coarse demonstration.

```python
import numpy as np
from tpmsfem.process import Evaporation, Laser, ScanVector
from tpmsfem.process.freesurface import FreeSurfaceMeltPool, SahooSurfaceTension
from tpmsfem.process.powder import rain_powder, voxelise

dx = 10e-6
nx, ny, nz = 40, 20, 14
metal = np.zeros((nx, ny, nz))
metal[1:-1, 1:-1, 1:7] = 1.0                                      # plate, 60 um thick
bed = rain_powder(extent=(dx, (nx - 1) * dx, dx, (ny - 1) * dx), floor=7 * dx, layer=40e-6, seed=0)
print(f"{len(bed)} particles, packing {bed.packing_density:.2f}")
metal = np.maximum(metal, voxelise(bed, dx, metal.shape))          # fill levels
fs = FreeSurfaceMeltPool(IN625, dx, metal, dt=2e-7, viscosity=5e-3, sigma=1.8,
                         dsigma_dT=SahooSurfaceTension(a_s=0.01, A=3.8e-4), evaporation=Evaporation(),
                         energy_transport="lbm")                    # IN625 from Section 12
out = fs.run(Laser(180.0, 0.35, 50e-6, depth=5e-6), [ScanVector(80e-6, ny * dx / 2, 300e-6, ny * dx / 2, 0.8)],
             t_extra=2e-5)
print(f"u_max {out.u_max:.2f} m/s, mass error {out.mass_error:.1e}, energy residual {out.energy_residual:.1%}, "
      f"{out.steps} steps at {out.mlups:.2f} MLUPS")
```

* **Case B.** `tools/lbm/freesurface_caseB.py`.
* **Track on powder.** `tools/lbm/freesurface_powder_track.py`.
* **Animation.** `tools/lbm/freesurface_animation.py`.
* **Status and limits.** See [freesurface_lbm.md](freesurface_lbm.md).

---

## 20. EBSD analysis

`process.ebsd` reads and analyses EBSD maps of cubic phases:
* `read_ctf` reads Oxford .ctf files;
* `euler_to_matrix` converts Bunge angles;
* `misorientation` uses the 24 cubic symmetry operators;
* `segment_grains` segments grains at a threshold angle;
* `kam` gives the kernel average misorientation;
* `grain_shapes` gives grain sizes and aspect ratios.

<!-- not run: needs a measured map, e.g. NIST AMB2018-02 (fetched by the benchmark script) -->
```python
from tpmsfem.process import ebsd as E
m = E.read_ctf("High_resolution_Map_Data_3.ctf")
g = m.orientations()                                  # (ny, nx, 3, 3)
valid = m.phase > 0
labels = E.segment_grains(g, valid, threshold=10.0, min_size=10)
shapes = E.grain_shapes(labels, m.step)
print(labels.max() + 1, "grains")
```

`benchmarks/literature/nist_ambench2018_tracks_ebsd.py` and `nist_ambench2018_tracks_ca.py`
compare predicted grains with this map.

---

## 21. Benchmark and study scripts

| calculation | script | documentation |
|---|---|---|
| effective stiffness, Gibson–Ashby fit, GCI | `examples/01_tpms_effective_stiffness.py` | README |
| femoral stem, ISO 7206-4 | `examples/02_femoral_stem_iso7206.py` | README |
| porous cup, AM residual stress | `examples/03_cup_am_residual_stress.py` | README |
| code-to-code vs CalculiX/Abaqus | `examples/04_benchmark_vs_calculix.py` | benchmarking.md |
| melt pool, G–R, grains, ExaCA export | `examples/05_lpbf_meltpool_microstructure.py` | process_models.md §1–§5 |
| phase-field cells | `examples/06_phasefield_cells.py` | process_models.md §5.3 |
| layer animation | `examples/07_layer_animation.py` | process_models.md |
| published benchmarks (lattices, implants, AM-Bench) | `benchmarks/literature/*.py` | literature_benchmarks.md |
| AM-Bench thermo-mechanical bridge | `benchmarks/literature/nist_ambench2018_tm.py` | literature_benchmarks.md |
| phase field up the melt pool with trapping | `benchmarks/literature/pf_path_trapping.py` | process_models.md §5.3 |
| grain CA vs EBSD; IN625 blocks | `nist_ambench2018_tracks_ca.py`, `ca_multilayer.py` | process_models.md §5.4–§5.5 |
| CoCrFeMnNi gyroid: solidification, heat accumulation, residual strain, grains | `cantor_gyroid_*.py` | process_models.md §7 |
| porosity maps | `porosity_maps.py` | porosity_maps.md |
| free-surface melt flow | `tools/lbm/*.py` | freesurface_lbm.md |

Long studies store their results in `benchmarks/literature/results/` and can be resumed.
`tools/report/make_report.js` builds the Word progress report from the stored figures.

---

## 22. Command line

```bash
tpmsfem info                                              # backend and device
tpmsfem voxelize lattice.vtu --surface gyroid --kind sheet --cell 3 --density 0.3 --size 6 --h 0.1
tpmsfem compare ours.npz --abaqus abq_prefix              # compare with an Abaqus export
```
