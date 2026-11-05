# Verification, validation and benchmarking plan

The goal is to show that tpmsfem gives the same answer as established commercial codes on
**the same discrete problem**, and that the discrete problem converges to the physics.
The structure follows ASME V&V 40-2018 and the FDA guidance on computational modelling
credibility for medical devices (2023):

| level | question | how | status |
|---|---|---|---|
| Code verification | Are the equations solved correctly? | patch tests, closed-form solutions (`tests/test_verification.py`) | done |
| Code-to-code | Same answer as independent solvers? | identical decks → CalculiX / Abaqus / Ansys (this page) | CalculiX done; Abaqus/Ansys decks ready |
| Calculation verification | Is the mesh fine enough? | GCI study per model (`bench.metrics.gci`, example 01) | tooling done |
| Validation | Does the model match experiments? | lattice compression tests, DIC, LPBF distortion (CMM/3D scan), XRD/contour residual stress | your lab |

## 1. Make the problems identical

Most "code A vs code B" disagreements in lattice/AM papers come from defining different
problems, not from solver error. `tpmsfem.io.abaqus.write_inp` therefore exports:

* the same nodes and elements (1-based, Abaqus node ordering),
* **one `*BOUNDARY` line per constrained dof**, not a geometric selection,
* **consistent nodal loads as `*CLOAD`**, not a pressure or coupling that each code integrates differently,
* eigenstrains as orthotropic expansion coefficients with a nodal temperature 0 → 1,
* the AM process as `*MODEL CHANGE` steps, written in a form both codes interpret identically (see §4).

Element formulation must also match (docs/theory.md §3):

| your reference | use in tpmsfem |
|---|---|
| Abaqus C3D8 (default, selectively reduced) | `SolidModel(..., bbar=True)` |
| CalculiX C3D8 | `bbar=False` (default for elastic) |
| Ansys SOLID185 (default KEYOPT(2)=0 = B-bar) | `bbar=True` |
| C3D10 / SOLID187 | tet10 |

## 2. Run the reference

**CalculiX** (automated): `python examples/04_benchmark_vs_calculix.py`

**Abaqus**
```bash
abaqus job=stem_abaqus input=results/04/stem_abaqus.inp cpus=8 interactive
abaqus python tools/abaqus_odb_export.py stem_abaqus.odb stem_abq
python -m tpmsfem.bench.compare results/04/stem_tpmsfem.npz --abaqus stem_abq --json stem_vs_abaqus.json
```
For hex8 cases rerun example 04 with `--bbar`, so that the tpmsfem side matches Abaqus' C3D8.

**Ansys Mechanical / MAPDL.** Import the `.inp` mesh through *External Model* (Workbench), or
convert it with meshio. Re-apply the constrained node sets and nodal forces from the deck
(they are listed explicitly). A native APDL writer is on the roadmap. For the AM process, use
Ansys' own inherent-strain workflow and compare final distortion and stress statistics, not
step-by-step fields.

## 3. Metrics

`tpmsfem.bench.compare.compare` reports:

* `U_rel_l2`, `U_rel_max`: nodal displacement
* `S_rel_l2`: element-mean stress tensor (independent of integration-point numbering)
* `vonMises_percentiles`: relative difference of the p50/p95/p99/max von Mises.
  Use p95–p99 for lattices: stair-step corners create singular single-element maxima.
* wall-clock time and dofs (and GPU memory via `benchmarks/perf_scaling.py`)

Acceptance used here: identical discrete problem → `U_rel_l2 < 1e-5`, `S_rel_l2 < 1e-5`
(output-precision limited). Different element formulations or meshers → report the difference,
together with each code's GCI.

## 4. Pitfalls we hit (and how the exporter handles them)

1. **Integration-point numbering.** CalculiX numbers C3D8 Gauss points lexicographically;
   tpmsfem numbers them like the element nodes. The reader permutes them. Element means sidestep this.
2. **Element birth in CalculiX** is only allowed in non-linear steps. The exporter adds an
   unreachable `*PLASTIC` card (yield 1e12) to elastic materials, which keeps the response linear.
3. **CalculiX makes born elements strain-free at the *end* of their birth step**, so any load
   applied in that step is absorbed. The exporter writes two steps per layer: birth, then inherent strain.
4. **CalculiX extrapolates temperatures onto newly activated nodes.** The exporter pins the new
   nodes to T = 0 in the birth step.
5. **Partial temperature at birth.** A newborn element whose bottom nodes are already at T = 1
   gets the *element-mean* thermal strain absorbed by CalculiX; reproduce this with
   `AMProcess(..., mode="nodal", birth_reference="element_mean")`. The physically consistent
   default (`"qp"`) differs by tens of % in the newest layers. How Abaqus treats this case must be
   checked once on your installation with `cup_am_abaqus.inp`. If it disagrees, compare against both variants.
6. **Voxel grids must be commensurate** with the geometry when boundary node sets are selected by
   coordinate, or the "top" node set can silently be empty. Examples snap `h`.
7. **Under-resolved TPMS walls** (< 3 voxels) make the stiffness depend strongly on `h`, and
   islands can disconnect. Always report the voxels-across-wall count and a GCI.

## 5. Suggested benchmark matrix for a paper

| case | geometry | physics | references |
|---|---|---|---|
| B1 | gyroid / diamond / IWP sheet unit cells, ρ = 0.2–0.5 | linear elastic, E_eff | analytical bounds, Abaqus, compression tests |
| B2 | lattice-core femoral stem | ISO 7206-4 load, elastic | Abaqus C3D8 / C3D10, strain gauges |
| B3 | porous acetabular cup | press-fit pressure, elastic | Abaqus, DIC |
| B4 | AM cantilever (calibration) | inherent strain, layer-wise | Abaqus AM / Ansys Additive / Simufact, CMM |
| B5 | porous cup as-built | inherent strain + J2, removal | Abaqus, 3D-scan distortion, XRD/contour |
| B6 | scaling | 1e5 → 1e8 dofs, CPU vs GPU | wall time, memory, energy per solve |
