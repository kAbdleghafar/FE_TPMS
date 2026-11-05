# Theory and implementation notes

Units are consistent (mm, N, MPa). Symmetric tensors are stored as 6-vectors
`[xx, yy, zz, yz, xz, xy]` with tensor (not engineering) shear.

## 1. Discrete equilibrium

Find nodal displacements `u` such that, on all free dofs,

    R(u) = f_ext - f_int(u) = 0,     f_int = Σ_e Σ_q w_eq B_eqᵀ σ_eq

with integration-point stress

    σ = C : (ε(u) - ε_ref - ε* - ε_p)

* `ε_ref` is the strain an element had when it was born (element birth, §4),
* `ε* = s · ε̂*` is an eigenstrain (thermal or inherent strain) with direction `ε̂*`
  per element and magnitude `s` per integration point,
* `ε_p` is the J2 plastic strain.

Newton–Raphson with the consistent tangent solves `R = 0`. Loads, prescribed
displacements and eigenstrain magnitudes are *targets* that `SolidModel.solve(n_inc)`
ramps linearly from their previously applied values, as an Abaqus step does.
Convergence: `‖R_free‖ ≤ tol · max(‖f_ext‖, ‖f_int‖, ‖R₀‖)`, where `R₀` is the initial residual
of the increment. Without `R₀`, a body released from its supports, whose `f_int → 0`, could never converge.

## 2. Matrix-free operator (CPU/GPU)

`K v` is never assembled for the `pcg` solver:

1. gather element dofs `v_e = v[dofs]` (chunked),
2. displacement gradients at integration points,
3. stress increment (elastic or consistent tangent),
4. element forces, then a scatter-add (`bincount`, atomic on GPU).

On **voxel meshes** every element is the same cube, so steps 2 and 4 become dense GEMMs with
one shared gradient matrix `M` (24 × 72). For elasticity the whole element operator collapses to
`f_e = λ_e v_e K_λ + μ_e v_e K_μ`, with two 24 × 24 reference matrices computed once.
For tets, per-element gradients are stored (`ne × nq × nn × 3`) and batched `matmul` is used.
Memory per hex8 element (elastic) is about 200 bytes, plus state arrays for plasticity and element birth.

Preconditioners: Jacobi (diagonal computed matrix-free) on any backend; on the CPU,
smoothed-aggregation AMG (pyamg, rigid-body near-null space) or sparse LU. For plasticity
the AMG/LU of the *elastic* matrix preconditions CG on the *tangent* operator, so there is
no re-assembly per Newton iteration.

## 3. Elements

| tpmsfem | quadrature | Abaqus | CalculiX |
|---|---|---|---|
| `hex8` | 2×2×2 Gauss | — | C3D8 (verified identical) |
| `hex8`, `bbar=True` | 2×2×2 + mean dilatation | C3D8 (selectively reduced) | — |
| `tet4` | 1 point | C3D4 | C3D4 |
| `tet10` | 4 points | C3D10 | C3D10 (verified identical) |

**B-bar** in gradient form: the volumetric strain at each point is replaced by the
element average `θ̄ = Σ w θ_q / Σ w`. The internal force then uses the modified stress
`σ̃_q = dev σ_q + p̄ I`, where `p̄` is the weighted mean pressure. This follows from
`δε̄_q = dev δε_q + ⅓ δθ̄ I` and makes the tangent symmetric. It is on by default for plastic hex8
(J2 flow is isochoric, so full integration would lock).

## 4. Element birth

`activate(cells, strain_free=True)` sets `ε_ref = ε(u_now) − ε*_present` at every integration
point of the new elements, so they are born stress-free on the current deformed configuration,
as a new powder layer is on a part that has already distorted. New nodes start at `u = 0`
(nominal, recoater-levelled position).

`eig_birth="element_mean"` subtracts the *element-average* eigenstrain present at birth
instead of the point-wise value. That is what CalculiX 2.21 does when a temperature field is
already non-zero on part of a newborn element. We verified this to 1e-7 against ccx (see benchmarking.md).

## 5. J2 plasticity

Radial return with linear isotropic hardening `σ_y(α) = σ_y0 + H α`:

    trial  s_tr = dev C:(ε − ε_p,n − …),  q_tr = √(3/2)|s_tr|
    f = q_tr − σ_y(α_n) > 0  ⇒  Δα = f / (3μ + H),  n = s_tr/|s_tr|
    σ = σ_tr − 2μ √(3/2) Δα n,   ε_p += √(3/2) Δα n

Consistent tangent `C_ep = κ I⊗I + 2μθ I_dev − 2μθ̄ n⊗n` with
`θ = 1 − 3μΔα/q_tr`, `θ̄ = 3μ/(3μ+H) − 3μΔα/q_tr` (Simo & Hughes 1998, Box 3.2).
Per integration point only `θ, θ̄, n` are stored; the tangent is applied matrix-free.

## 6. Inherent-strain AM model

1. The part is sliced into super-layers along +z (build direction).
2. For each layer: birth (strain-free), then apply the inherent strain `ε*` to the layer, then
   solve, with the plate interface welded (all dofs fixed).
3. Removal: release the weld and add a statically determinate 3-2-1 support (A: xyz; B: two
   components; C: z) chosen from the longest chord of the plate footprint. This adds no stress.
4. Optional J2 plasticity (`--plastic`) captures yielding during the build. It also makes the
   response non-linear in `ε*`.

`mode="element"`: `ε*` is uniform in each element (the classical method).
`mode="nodal"`: `ε*` follows a nodal activation field interpolated to integration points,
which is how a commercial code applies it via "temperature" and orthotropic expansion.
Use this mode for code-to-code comparisons of the layer process.

**Calibration.** For the elastic method, every response is linear in `ε*`. A single unit
simulation of a calibration geometry (e.g. a cantilever cut from the plate) therefore gives
the scale factor directly: `residual.am.calibrate_scale`. Anisotropic calibration (εxx, εyy, εzz)
needs three unit runs and a least-squares fit.

## 7. Post-processing

Integration-point stresses are extrapolated to nodes (`E = N_q⁻¹` for hex8; a linear fit for
tet10) and averaged over the elements sharing a node. That is comparable to commercial
"averaged nodal" output, but **voxel corner singularities** make single-node maxima meaningless,
so report percentiles. Element averages are integration-point-order independent and are what
the benchmarks compare.
