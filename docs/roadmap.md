# Roadmap

Ordered by impact on the project goals (speed on real implant models, residual-stress
fidelity, benchmark credibility).

## Near term

1. **Run and tune the GPU path.** Run `pytest` and `benchmarks/perf_scaling.py` with
   `TPMSFEM_BACKEND=cupy`. Fuse gather → GEMM → scatter into custom CUDA kernels
   (`cupy.RawKernel`) if the throughput is below ~200 M elem/s. Try mixed precision
   (fp32 operator inside fp64 CG) on consumer GPUs.
2. **Geometric multigrid for voxel meshes.** 2×2×2 coarsening with Galerkin coarse
   operators (matrix-free on fine levels, assembled on coarse levels), Chebyshev–Jacobi
   smoothing. This gives iteration counts independent of mesh size, and it is what
   giga-voxel FE relies on (cf. Aage et al., *Nature* 2017).
3. **Native Ansys APDL writer** (SOLID185/187, D/F commands, MP/TB), mirroring the Abaqus exporter.
4. **Hot-spot accuracy.** Map voxel solutions to tet10 meshes of the smoothed surface
   (submodelling), or use a smoothed-boundary voxel formulation.
5. **Anisotropic calibration** helper for inherent strain (three unit runs + least squares),
   plus an example with published cantilever data.

## Medium term

6. **Thermo-mechanical layer-by-layer AM** (transient heat conduction per super-layer +
   thermo-elasto-plasticity) as a higher-fidelity alternative to inherent strain. The
   Barsoum benchmark (docs/literature_benchmarks.md §2.4) shows a uniform inherent strain
   cannot rank TPMS topologies by residual-stress sensitivity; this is therefore the top
   physics item.
7. **Stress-relief / HIP heat treatment** (creep-based relaxation of the residual stress).
8. **Contact**: press-fit of the cup into bone (interference fit), stem–bone interface, as
   mortar or penalty contact with an augmented Lagrangian on the GPU.
9. **Bone models**: CT-based heterogeneous bone (HU → E mapping) and bone-remodelling
   (strain-energy-density) loops for stress-shielding studies.
10. **Fatigue post-processing**: Goodman/Sines/Crossland with as-built LPBF S-N data; ISO 7206-4 / -6 reporting.
11. **Multi-GPU** domain decomposition (NCCL) for > 1e8 dofs.

## Long term

12. Homogenisation-based multiscale (FE²) for graded lattices.
13. Gradient-based lattice-density optimisation (adjoint is cheap with the matrix-free operator).
14. Geometric non-linearity (lattice buckling under compression).
