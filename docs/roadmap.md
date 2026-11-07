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
7. **Stress-relief / HIP heat treatment** (creep-based relaxation of the residual stress).
8. **Contact**: press-fit of the cup into bone (interference fit), stem–bone interface, as
   mortar or penalty contact with an augmented Lagrangian on the GPU.
9. **Bone models**: CT-based heterogeneous bone (HU → E mapping) and bone-remodelling
   (strain-energy-density) loops for stress-shielding studies.
10. **Fatigue post-processing**: strain-life (Basquin–Coffin–Manson, Morrow/SWT, Neuber, cyclic Ramberg–Osgood) **done** (`post/fatigue.py`, as-built LPBF IN625 parameters of Concli et al. 2024). Next: multiaxial criteria (Sines, Crossland, Fatemi–Socie) on FE stress histories, residual stress as mean stress, and ISO 7206-4 / -6 reporting.
11. **Multi-GPU** domain decomposition (NCCL) for > 1e8 dofs.

## Long term

12. Homogenisation-based multiscale (FE²) for graded lattices.
13. Gradient-based lattice-density optimisation (adjoint is cheap with the matrix-free operator).
14. Geometric non-linearity (lattice buckling under compression).
