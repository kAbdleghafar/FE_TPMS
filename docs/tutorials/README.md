# TPMSsuite tutorials

Step-by-step tutorials, one for each calculation TPMSsuite can do. Each one:
* states its goal and what you will learn;
* builds the calculation in small steps, explaining every input;
* shows the output of a real run, recorded under each code block (`Output (from the test
  run):`). Your numbers should match closely on the NumPy backend;
* ends with checks, limits, exercises and links to the studies and validation.

The tutorials are standalone: each starts from scratch, in its own namespace, and runs in
seconds to a few minutes on a laptop CPU. All of them are executed by
`python tools/manual/check_manual.py docs/tutorials/*.md`; `--update-output` refreshes the
recorded outputs. For a compact reference of every function, see the [user manual](../manual.md).

| # | tutorial | what you compute | time |
|---|---|---|---|
| 1 | [Getting started](01_getting_started.md) | backends, units, a first stiffness calculation | 1 min |
| 2 | [Geometry](02_geometry.md) | TPMS sheets and networks, density, grading, booleans, implants | 10 s |
| 3 | [Meshing](03_meshing.md) | voxel meshes, resolution, STL surfaces, tetrahedra | 30 s |
| 4 | [A compression test](04_linear_static.md) | effective modulus, stresses, surface loads, mesh convergence | 1 min |
| 5 | [Homogenisation](05_homogenisation.md) | 6 × 6 stiffness of a cell, moduli, Gibson–Ashby exponent | 1 min |
| 6 | [Plasticity](06_plasticity.md) | yield of a lattice, force–displacement curve, tabulated hardening | 1 min |
| 7 | [Stacking faults and HCP](07_stacking_faults_hcp.md) | where twins/HCP could start, how much HCP forms | 1 min |
| 8 | [Implants and fatigue](08_implants_fatigue.md) | ISO 7206-4 stem, strain-life with mean stress | 30 s |
| 9 | [Heat conduction](09_heat_conduction.md) | transient cooling of a lattice | 10 s |
| 10 | [Inherent strain](10_inherent_strain.md) | AM residual stress, layer by layer, removal | 20 s |
| 11 | [Thermo-mechanical build](11_thermomechanical_build.md) | hot deposition, cooling, temperature-dependent stresses | 1 min |
| 12 | [Export and benchmarking](12_export_benchmark.md) | Abaqus/CalculiX decks, field comparison, GCI | 20 s |
| 13 | [Melt pool](13_melt_pool.md) | single tracks, power sweep, powder, calibration | 2 min |
| 14 | [Scan paths](14_scan_paths.md) | contours and hatches of a TPMS layer, track ends, ramps, turns | 10 s |
| 15 | [Solidification](15_solidification.md) | G, R, cell spacing, CET, grains, ExaCA export | 1 min |
| 16 | [Phase field](16_phase_field.md) | cells and segregation in 2-D and 3-D, synthetic EDS | 3 min |
| 17 | [Grain CA](17_grain_ca.md) | grains and texture over tracks and layers | 2 min |
| 18 | [Porosity and keyhole maps](18_porosity_keyhole_maps.md) | lack of fusion, keyhole depth/stability, constant-Ke power | 10 s |
| 19 | [Roughness](19_roughness.md) | Sa against build angle, over a gyroid | 20 s |
| 20 | [Free-surface melt flow](20_free_surface_flow.md) | a track over powder with Marangoni, recoil, evaporation | 2 min |
| 21 | [Keyholes](21_keyholes.md) | ray tracing, bubbles and pores, a keyhole run, closed keyholes | 2 min |
| 22 | [EBSD](22_ebsd.md) | grains, shapes and KAM from an orientation map | 5 s |
| 23 | [Command line](23_command_line.md) | `tpmsfem info / voxelize / compare` | – |

**Suggested paths.**
* *Lattice mechanics:* 1 → 2 → 3 → 4 → 5 → 6 → 8.
* *Residual stress:* 9 → 10 → 11 → 12.
* *Process and microstructure:* 13 → 14 → 15 → 16 → 17 → 22.
* *Defects and keyholes:* 13 → 18 → 20 → 21, with 14 for scan paths and 19 for surfaces.
* *Deformation mechanisms:* 6 → 7.
