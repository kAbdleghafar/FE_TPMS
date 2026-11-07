# tpmsfem

**Open-source, GPU-accelerated finite element analysis of TPMS lattice implants:
stress/strain under physiological loads and additive-manufacturing residual stress,
with interfaces to open-source meshers and a code-to-code benchmarking workflow
against commercial solvers.**

```
implicit TPMS geometry ──► voxel hex8 mesher ─────────────┐
 (gyroid, diamond, P, IWP,   (GPU classification)          │
  graded density, implants) ─► marching cubes ─► Gmsh /     ├─► matrix-free FE solver ─► VTU (ParaView)
                                 TetGen / fTetWild (tet4/10)│    NumPy | CuPy (CUDA)   ─► Abaqus / CalculiX deck
 any .msh/.inp/.vtu/.med via meshio ───────────────────────┘    elastic | J2 plastic   ─► benchmark metrics
                                                                 element birth + eigenstrain (LPBF)
```

| | |
|---|---|
| Geometry | Gyroid, Schwarz-P, diamond, IWP, Neovius, FKS; sheet or network; density-calibrated and functionally graded; composable SDF booleans; parametric femoral stem (lattice core / porous coating) and acetabular cup |
| Meshing | Built-in voxel (hex8) mesher with island removal and GPU classification; marching cubes → **Gmsh**, **TetGen**, **fTetWild** (tet4/tet10); anything **meshio** reads |
| Elements | hex8 (full integration or **B-bar**), tet4, tet10 |
| Physics | Small-strain elasticity; J2 plasticity with linear or **tabulated, temperature-dependent** isotropic hardening (exact multilinear radial return + consistent tangent, annealing); thermal strains with E(T), ν(T), α(T); eigenstrains (inherent strain); element birth (strain-free activation, also at a deposition temperature); **transient heat conduction** (k(T), c_p(T), latent heat by an enthalpy method, convection + radiation on evolving free surfaces) |
| Residual stress | LPBF **inherent-strain layer-by-layer** build, or a **sequentially coupled thermo-mechanical** layer-lumped build (heat + temperature-dependent elasto-plasticity), + build-plate removal (3-2-1 support) → residual stress and distortion |
| Process scale | **Laser scan → melt pool → solidification → surface**: explicit enthalpy finite-volume melt-pool solver on voxel grids (Gaussian laser, scan paths and hatching, powder layer, latent heat, evaporation, NumPy/CuPy), G–R maps, cell spacing (Kurz–Fisher/Hunt), Hunt CET, epitaxial competitive grain growth with IPF maps, **ExaCA** export; **2-D quantitative phase field** of cells/dendrites and Nb segregation at the melt-pool G, R (Karma–Echebarria anti-trapping model, numba/CuPy); as-built roughness (staircase + melt-pool rounding + adhered powder) vs build angle, mapped onto TPMS surfaces |
| Fatigue | Strain-life post-processing: cyclic Ramberg–Osgood, Basquin–Coffin–Manson with Morrow / SWT mean-stress correction, Neuber notch rule |
| Solver | Matrix-free element-by-element operator (one batched GEMM per chunk on voxel meshes); **geometric multigrid** PCG for voxel meshes (matrix-free Galerkin hierarchy, mesh-independent iteration counts, CPU **or GPU**); AMG (pyamg) or sparse direct for tet meshes; Newton–Raphson with an elastic predictor and automatic cut-back for plasticity |
| Benchmarking | Exact Abaqus/CalculiX deck export (same nodes, elements, BCs, **consistent nodal loads**, eigenstrains, layer steps), CalculiX runner + readers, Abaqus ODB→CSV script, field metrics, Roache GCI |

## Status: what is verified

All numbers are reproducible with `pytest` and the scripts in `examples/`.

**Code vs. mathematics** (`tests/test_verification.py`)

| test | result |
|---|---|
| Patch test, distorted hex8 (standard and B-bar), tet4, tet10 | exact to round-off |
| Lamé thick sphere under pressure, Gmsh tet10 | max radial-displacement error 0.53 % |
| Hill elastic–perfectly-plastic thick sphere (J2, 5 increments) | outer displacement error 0.07 % |
| Uniaxial J2 with linear hardening, then elastic unloading | exact (1e-6) |
| Bilayer strip with eigenstrain (curvature) | 0.78 % |
| Clamped / free body with eigenstrain | exact |
| Cantilever, voxel hex8 (h = L/40 → L/80) | 4.3 % → 1.6 % (shear locking of linear hexes, converging) |

**Code vs. code** (`examples/04_benchmark_vs_calculix.py`, CalculiX 2.21, identical decks)

| case | element | dofs | U rel-L2 | S rel-L2 | von Mises p99 |
|---|---|---:|---:|---:|---:|
| gyroid sheet lattice, compression + shear | hex8 | 127,941 | 8.6e-07 | 5.2e-08 | identical |
| lattice-core femoral stem, ISO 7206-4 load | hex8 | 132,930 | 1.4e-06 | 5.1e-08 | identical |
| porous cup, **LPBF layer-by-layer + plate removal** | hex8 | 52,008 | 1.7e-06 | 7.3e-08 | identical |
| thick sphere under pressure | tet10 | 46,329 | 1.2e-06 | 3.7e-08 | identical |

(1e-6 is the precision floor of CalculiX' `.frd` output.) The Abaqus decks for the same
cases are written next to the CalculiX ones; see [docs/benchmarking.md](docs/benchmarking.md).

**Not yet verified:** the CuPy (GPU) path has not been executed by the authors yet (the
development container had no GPU). The code uses only NumPy-compatible CuPy APIs; please
run `pytest` with `TPMSFEM_BACKEND=cupy` and report results. Abaqus decks have been
validated through CalculiX but not run in Abaqus itself.

### Against published results

[docs/literature_benchmarks.md](docs/literature_benchmarks.md) re-implements published results from
four independent groups and compares them:

* **Van Viet, Zaki & El-Rich (Khalifa U.), PLoS One 2025** (graded gyroid femur/cranial implants):
  all four stiffnesses stated in the paper are reproduced within **-1.1 % to +3.2 %**.
* **Ahmed, Barsoum & Abu Al-Rub (Khalifa U.), Metals 2022** (IN718 TPMS lattices, SLM residual stress):
  * Periodic-cell moduli match the literature values in their Table 3 (IWP-S and IWP-L within
    about 5-8 %, P-S within 4-12 %).
  * Their 3x3x3 Ansys moduli equal our elastic-plastic secant modulus at 0.7-0.9 % strain.
  * The residual-stress modulus loss is reproduced in magnitude, but not its topology ranking.
* **NIST AM-Bench 2018** (IN625 LPBF bridge). The **thermo-mechanical** build uses NIST MaCTO
  flow curves of the same build and the published layer time, with **no calibration**:
  * ridge deflection after the leg cut within **3.3 %** of the measurement. With NIST AMB2022
    tension data from another build it is +8.5 %; with a generic DED property table, −40 %.
    The deflection follows the as-built yield strength, while the strain pattern does not
    change.
  * contour-method σ_xx r = 0.71 in the part and 0.87 in the plate;
  * synchrotron ε_xx / ε_zz r = 0.80 / 0.92;
  * neutron ε_xx r = 0.88 in the body. The thin legs are still not resolved (r = 0.22),
    although their tensile bias is gone.
  * The elastic inherent-strain model, calibrated on the deflection, gives r = 0.77 / 0.85,
    0.75 / 0.89 and 0.81 / 0.13 for the same quantities.
* **NIST AM-Bench 2018-02** (laser tracks on IN625): calibrated on NIST's case B, the melt-pool
  model predicts the widths of cases A and C within 5 % and the depths within −17 % to +11 %.
  Surface cooling rates are 3–5× above the thermography ([docs/process_models.md](docs/process_models.md)).
  Regrown from NIST's *measured* substrate grains (case-B EBSD map), the simulated melt pool
  reproduces 26–28 % of the measured grain orientations point by point (random: 0.5 %), and,
  along the simulated gradient, the measured grain count. A 3-D back-trace through the
  simulated melt pool explains the rest. Grains that start within 0.5 µm of the measured
  plane match 51 % of the time. Those starting ≥ 1.5 µm behind it, on substrate NIST did not
  measure, match only 6–8 %.
* **Ghosh et al. 2017 / 2018** (NIST, phase-field cells in Ni–Nb / IN625): code to code, the phase field
  reproduces their primary spacing (0.604 µm on their fit at 3e5 K/s, 0.221 µm against 0.23 µm at 1e6 K/s)
  and tip compositions (3.78 / 6.81 wt% Nb against 3.65 / 6.87). Driven by the case-B melt pool, it gives
  0.57 µm cells at the fusion boundary falling to 0.30 µm higher up, inside the 0.2–1.0 µm measured by SEM;
  Hunt's law trends the other way. Verified against Mullins–Sekerka theory; quantitative while D/V ≳ 10 W0.
* **Kasperovich et al. 2021** (LPBF IN718 roughness vs build angle): the surface model reproduces the
  upskin, vertical and downskin Sa after fitting three powder-adhesion parameters. It then predicts
  26 µm area-weighted Sa for a gyroid implant cell, dominated by its 30 % downskin area.
* **Naghavi, ..., Liu (UCL), Bioengineering 2022** (Ti6Al4V gyroid/diamond bone scaffolds):
  * Design trends match their Abaqus FE within ±4 % after one common factor; the absolute values
    are 15-21 % lower, which points to the solid modulus used in their FE.
  * We are 10-20 % below their experiments, consistent with as-built densification.

## Install

```bash
pip install -e ".[dev]"            # CPU: numpy/scipy + meshers, AMG, pytest
pip install cupy-cuda12x           # GPU (CUDA 12); then TPMSFEM_BACKEND=cupy
sudo apt install calculix-ccx      # optional: open-source reference solver
```

Gmsh's wheel needs `libglu1-mesa` (and X libs) on minimal Linux images.

## Quick start

```python
import numpy as np
import tpmsfem as tf
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.post import save_vtu

lat  = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.3)       # or density=lambda x,y,z: ...
mesh = voxelize(lattice_cube(lat, 8.0), (0, 0, 0), (8, 8, 8), h=0.1)  # GPU-classified voxels
md   = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())

z = mesh.points[:, 2]
md.fix(np.flatnonzero(z < 1e-9))                                      # clamp bottom
md.fix(np.flatnonzero(z > 8 - 1e-9), (2,), -0.008)                    # 0.1 % compression
info = md.solve()                                                     # pcg on GPU, amg/direct on CPU
save_vtu(md, "gyroid.vtu")
```

Residual stress from LPBF:

```python
from tpmsfem.residual.am import AMProcess
am = AMProcess(md, layer_height=0.5, inherent_strain=(-4e-3, -4e-3, -2e-3, 0, 0, 0))  # calibrate!
am.build()               # layer-by-layer, welded to the plate
am.remove_from_plate()   # spring-back, final residual stress
```

## Examples

| script | what it does |
|---|---|
| `examples/01_tpms_effective_stiffness.py` | E_eff vs. density for gyroid/diamond/P sheets, Gibson–Ashby fit, mesh-convergence GCI |
| `examples/02_femoral_stem_iso7206.py` | solid vs. lattice-core stem under ISO 7206-4-style load: stiffness (stress shielding), stress percentiles, Abaqus/CalculiX decks |
| `examples/03_cup_am_residual_stress.py` | porous cup built rim-down: layer history, residual stress on/off plate, rim out-of-roundness & flatness; `--plastic` for J2 |
| `examples/04_benchmark_vs_calculix.py` | the code-to-code benchmark above; writes Abaqus decks + our results for commercial comparison |
| `examples/05_lpbf_meltpool_microstructure.py` | AM-Bench case-B laser track: melt pool, G–R map, CET, epitaxial grains, ExaCA export, 5-track hatch |
| `examples/06_phasefield_cells.py` | 2-D phase-field cells in IN625 (Ni–4Nb) at a case-B melt-pool G and R: spacing and Nb segregation |
| `benchmarks/perf_scaling.py` | matrix-free throughput and solve time vs. size, per backend (CPU/GPU) |

## Documentation

* [docs/theory.md](docs/theory.md): formulation, elements, plasticity, inherent strain, element-birth semantics
* [docs/process_models.md](docs/process_models.md): melt pool, solidification microstructure (grains, phase-field cells), surface roughness
* [docs/benchmarking.md](docs/benchmarking.md): V&V plan, running Abaqus/Ansys/CalculiX comparisons, metrics, pitfalls found
* [docs/roadmap.md](docs/roadmap.md): what's next (GPU multigrid, thermo-mechanical AM, contact/press-fit, fatigue)

## Limitations (read before using results for design decisions)

* Small strain, quasi-static, isotropic materials. The material presets are typical
  literature values. Replace them with data from your as-built material.
* Inherent strains must be **calibrated** (e.g. cantilever/bridge distortion tests); the defaults are illustrative.
* Voxel meshes have stair-stepped surfaces: single-element stress maxima are mesh-dependent.
  Report percentiles, and use tet10 meshes of the smoothed surface for hot-spot or fatigue assessment.
  Use at least 4 voxels across the thinnest wall and check with a GCI study.
* Geometric multigrid applies to voxel meshes only. Tet meshes use AMG on the CPU, or Jacobi-PCG on the GPU.

License: Apache-2.0
