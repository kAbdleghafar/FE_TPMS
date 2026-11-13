# Roadmap

Ordered by impact on the project goals (speed on real implant models, residual-stress
fidelity, benchmark credibility).

## Near term

1. **Run and tune the GPU path.** Run `pytest` and `benchmarks/perf_scaling.py` with
   `TPMSFEM_BACKEND=cupy`. Fuse gather → GEMM → scatter into custom CUDA kernels
   (`cupy.RawKernel`) if the throughput is below ~200 M elem/s. Try mixed precision
   (fp32 operator inside fp64 CG) on consumer GPUs.
2. ~~Geometric multigrid for voxel meshes~~ **done** (`fem/multigrid.py`). Next: build the
   Galerkin products in float32 on GPUs, and use multigrid for tet meshes via
   voxel-to-tet transfer or aggregation.
3. **Native Ansys APDL writer** (SOLID185/187, D/F commands, MP/TB), mirroring the Abaqus exporter.
4. **Hot-spot accuracy.** Map voxel solutions to tet10 meshes of the smoothed surface
   (submodelling), or use a smoothed-boundary voxel formulation.
5. **Anisotropic calibration** helper for inherent strain (three unit runs + least squares),
   plus an example with published cantilever data.

## Medium term

6. ~~Thermo-mechanical layer-by-layer AM~~ **first version done**:
   * heat conduction (`thermal/heat.py`);
   * tabulated temperature-dependent J2 plasticity;
   * a layer-lumped sequentially coupled build (`residual/thermomech.py`);
   * the AM-Bench 2018 run (docs/literature_benchmarks.md §3).

   Next:
   * a moving heat source on a submodel, calibrated on the AMB2018 thermography
     (doi:10.18434/M31935; radiant temperature, emissivity correction, valid range 550–1050 °C);
   * combined isotropic–kinematic (Chaboche) hardening once hysteresis loops are available;
   * a GPU heat solver;
   * the Barsoum TPMS ranking with the thermal model.
7. **Process-scale models** (`process/`): melt pool, solidification and roughness **first version done**
   (docs/process_models.md). Next:
   * a free-surface melt-pool model (keyhole, Marangoni): **own GPU-ready solver started**
     (`process/freesurface.py`, docs/freesurface_lbm.md): free-surface lattice Boltzmann with melting,
     Marangoni, recoil, evaporation; verified, AM-Bench case B run on CPU at 10 µm. Next: powder bed,
     a fine window that follows the laser (as the 3-level refinement of Lei et al. 2025), fused CUDA
     kernels, height-function curvature with consistency checks, then multi-track layers on a GPU.
     (The earlier OpenLB feasibility check: docs/lbm_feasibility.md);
   * running ExaCA on the exported thermal histories;
   * grains: validated against the NIST case-B EBSD map, in 2-D (26–28 % point-wise, grain
     count from gradient-following epitaxy) and in 3-D (agreement 51 % for in-plane grain
     origins, decaying to 6–8 % beyond 1.5 µm). Next: serial-section substrate data, the track
     3/5/10 maps and cell spacing from SEM, once NIST's files download again;
   * cells and Nb segregation: 2-D quantitative phase field **done** (docs/process_models.md §5.3; code to
     code with Ghosh et al. 2017, consistent with the measured IN625 cell spacing; under-resolved above
     R ≈ 0.05 m/s at W0 = 10 nm, +38 % spacing at 5 nm). Next: 3-D runs on the GPU,
     a thinner interface (or a finite-interface-dissipation model) for R > 0.05 m/s, velocity-dependent
     partitioning, and Laves/NbC from the segregated Nb (CALPHAD), feeding the heat-treatment model;
   * lattice layers: part-aware melt-pool substrate and clipped scan paths **done**; CoCrFeMnNi gyroid study
     (docs/process_models.md §7). Multi-layer heat accumulation as preheat **done** (§7.1). Next: full-height
     builds and layer-time studies on the GPU, a CALPHAD-based multicomponent Mn/Ni partitioning, Mn-loss tracking, and diamond/IWP comparisons;
   * a GPU run of multi-layer patches;
   * roughness-aware effective wall thickness for TPMS stiffness and fatigue.
8. **Stress-relief / HIP heat treatment** (creep-based relaxation of the residual stress).
9. **Contact**: press-fit of the cup into bone (interference fit), stem–bone interface, as
   mortar or penalty contact with an augmented Lagrangian on the GPU.
10. **Bone models**: CT-based heterogeneous bone (HU → E mapping) and bone-remodelling
   (strain-energy-density) loops for stress-shielding studies.
11. **Fatigue post-processing**: strain-life (Basquin–Coffin–Manson, Morrow/SWT, Neuber, cyclic Ramberg–Osgood) **done** (`post/fatigue.py`, as-built LPBF IN625 parameters of Concli et al. 2024). Next: multiaxial criteria (Sines, Crossland, Fatemi–Socie) on FE stress histories, residual stress as mean stress, and ISO 7206-4 / -6 reporting.
12. **Multi-GPU** domain decomposition (NCCL) for > 1e8 dofs.

## Long term

12. Homogenisation-based multiscale (FE²) for graded lattices.
13. Gradient-based lattice-density optimisation (adjoint is cheap with the matrix-free operator).
14. Geometric non-linearity (lattice buckling under compression).
