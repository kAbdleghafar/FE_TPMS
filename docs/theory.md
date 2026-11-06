# Theory and implementation notes

Units are consistent (mm, N, MPa). Symmetric tensors are stored as 6-vectors
`[xx, yy, zz, yz, xz, xy]` with tensor (not engineering) shear.

## 1. Discrete equilibrium

Find nodal displacements `u` such that, on all free dofs,

    R(u) = f_ext - f_int(u) = 0,     f_int = Σ_e Σ_q w_eq B_eqᵀ σ_eq

with integration-point stress

    σ = C(T) : (ε(u) - ε_ref - ε* - ε_th(T) - ε_p)

* `ε_ref` is the strain an element had when it was born (element birth, §4),
* `ε* = s · ε̂*` is an eigenstrain (thermal or inherent strain) with direction `ε̂*`
  per element and magnitude `s` per integration point,
* `ε_th(T)` is the thermal strain of a temperature field (§5.1),
* `ε_p` is the J2 plastic strain.

Newton–Raphson with the consistent tangent solves `R = 0`. Loads, prescribed
displacements and eigenstrain magnitudes are *targets* that `SolidModel.solve(n_inc)`
ramps linearly from their previously applied values, as an Abaqus step does.
Convergence: `‖R_free‖ ≤ tol · max(‖f_ext‖, ‖f_int‖, ‖R₀‖)`, where `R₀` is the initial residual
of the increment. Without `R₀`, a body released from its supports, whose `f_int → 0`, could never converge.

**Robustness in plastic and thermal steps:**
* **Inexact Newton.** Iterative linear solves use Eisenstat–Walker forcing terms
  `η_k = 0.9 (‖R_k‖/‖R_k−1‖)²`, clipped to `[lin_rtol, 0.1]`. They are never tighter than
  `0.5 tol·ref/‖R_k‖`, so the linear solver does not over-solve. The elastic predictor of a
  plastic step is solved to 1e-3.
* **Backtracking line search.** It works on `‖R‖` with steps 1, ½, ¼ and keeps the best.
  It stops the elastic/plastic active-set cycling that otherwise stalls Newton at about 1e-2.
  An example is a hot layer contracting on reheated, softened material. In the AM-Bench
  build, the stalled increment converges in 7 iterations instead of cutting back.
* **Stagnation cut-back.** An increment whose residual has not halved within `stall_window`
  iterations is cut back at once.
* **Multigrid refresh.** With temperature-dependent moduli, the level-0 smoother is refreshed
  (Jacobi diagonal and Chebyshev spectral bound) when `μ` has drifted by more than 2 % since
  setup. A stale bound makes the Chebyshev smoother amplify the stiffened modes: a test that
  cools a block from 900 °C stalls without the refresh and converges in a few iterations with it.

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

**Geometric multigrid** (`fem/multigrid.py`, default for voxel meshes): PCG preconditioned
by a V-cycle on nested voxel grids.
* **Coarsening:** each level merges 2 × 2 × 2 voxels. The prolongation P is trilinear
  interpolation between the nested node lattices.
* **Coarse operators:** exact Galerkin products `A_{l+1} = Pᵀ A_l P`, assembled *element by
  element*. A fine voxel at child position c of its parent contributes `P_cᵀ K_e P_c`, with
  P_c one of eight constant 24 × 24 matrices. On the finest level this is just `λ_e A_c + μ_e B_c`.
  So coarse levels are also stored as per-element 24 × 24 matrices and applied matrix-free
  (batched GEMV). There is no assembled matrix except on the coarsest level, which uses
  sparse LU, or pyamg if it is still large.
* **Supports:** Dirichlet dofs are masked in the element matrices before the Galerkin
  product, so the coarse levels see the supports exactly.
* **Smoother:** a degree-2 Chebyshev–Jacobi smoother over [λ_max/8, λ_max], with λ_max from
  power iteration. It is symmetric, so the V-cycle is a valid CG preconditioner.
* **Plasticity:** the hierarchy is built from the elastic stiffness. The finest level applies
  the true consistent tangent.

Measured on a gyroid lattice, 4 CPU cores:
* iteration counts stay at 20–29 from 0.19 M to 1.17 M dofs;
* 1.17 M dofs solve in 32 s, against 102 s for AMG and 186 s for Jacobi-PCG (1,855 iterations);
* a layer-by-layer AM build is 3.8× faster than with AMG.

Other preconditioners: Jacobi (any backend); on the CPU, smoothed-aggregation AMG (pyamg,
rigid-body near-null space) or sparse LU, for non-voxel (tet) meshes. For plasticity the
AMG/LU of the *elastic* matrix preconditions CG on the tangent operator.

**Plastic voxel models** use the shared elastic 24 × 24 GEMM kernel for every element whose
integration points are all elastic. Only yielding elements take the integration-point
kernel, 7.7× faster when nothing has yielded.

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

### 5.1 Tabulated and temperature-dependent hardening

`FlowTable` gives a multilinear curve `σ_y(α)` (points `(α_k, σ_k)`, flat or `tail_slope`
beyond the last point, as Abaqus `*PLASTIC`), optionally for several temperatures (linear
interpolation in T, held constant outside the table). The consistency condition

    g(Δα) = q_tr − 3μ Δα − σ_y(α_n + Δα) = 0

is piecewise linear. It is solved **exactly**, without local iterations. The solver evaluates
`g` at every breakpoint beyond `α_n`, takes the first one where `g ≤ 0`, and solves the linear
equation on that segment (or on the tail). The consistent tangent is the one above, with `H`
the slope of the active segment. A one-point table with a tail slope reproduces linear
hardening to round-off (tested).

**Temperature.** The temperature field is nodal or per integration point and is ramped over
the increments of a step like any other load. Total formulation:

    σ = C(T) : (ε − ε_ref − ε* − ε_th(T) − ε_p),   ε_th = [α(T)(T − T_ref)] − [same at T_0]

* `α(T)` is the *mean* coefficient from `T_ref` (Abaqus `*EXPANSION, ZERO=`).
  `TemperatureTable.mean_from_instantaneous` converts tangent coefficients.
* `ε_th` is frozen above `T_cap` (the solidus): molten material carries no stress.
* `E(T)` and `ν(T)` are taken at the element-mean temperature, so the shared voxel GEMM and
  the multigrid hierarchy stay valid.
* `σ_y(T, α)` and `ε_th` are taken at the integration points.
* Birth absorbs the current thermal strain: an element born at its deposition temperature is
  stress-free there.
* `anneal_temperature` resets the equivalent plastic strain `α` above it (Abaqus
  `*ANNEAL TEMPERATURE`). The plastic strain tensor is kept, so the stress does not jump.

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

## 6.1 Heat conduction and the thermo-mechanical AM model

`thermal.HeatModel` solves `Σ_m C_m dH_m(T)/dt + K(T) T + q_s(T) = Q` on the same mesh.
* **Capacity.** Lumped nodal capacities with the volumetric enthalpy `H_m(T) = ρ∫c_p dT` plus
  the latent heat, released linearly between solidus and liquidus. Energy, latent heat
  included, is conserved exactly whatever the step (tested to 1e-9).
* **Conductivity.** `k(T)` per element at the element-mean temperature.
* **Surface losses.** Convection and radiation on the free faces of the *active* elements,
  with an optional face filter, for example to make powder-covered faces adiabatic.
* **Other features.** Prescribed temperatures, and element birth that gives the new nodes a
  deposition temperature.
* **Time integration.** Backward Euler. Each step uses Newton iterations whose update is applied
  to the nodal enthalpy and mapped back through the monotone `T(h)` (Nedjar 2002). Plain Newton
  or secant iterations in `T` oscillate across the solidus/liquidus kinks; this one does not.
  Linear systems use pyamg-preconditioned CG.
* **Verification.** The Fourier-series solution of a slab is reproduced, as is lumped-capacitance
  cooling (with a face filter).

`residual.thermomech.ThermoMechanicalAM` runs the layer-lumped ("flash") process, the
part-scale approach of commercial AM simulators:
1. Each super-layer is born at `T_deposit`: thermally on its new nodes, and mechanically
   stress-free.
2. It cools for the super-layer print time in geometrically growing heat steps.
3. After each heat step (or every `mech_every` steps) the temperature-dependent elasto-plastic
   model is re-equilibrated.
4. `cool_down` brings the part to room temperature. `remove_from_plate` puts it on the 3-2-1
   support.

## 7. Post-processing

Integration-point stresses are extrapolated to nodes (`E = N_q⁻¹` for hex8; a linear fit for
tet10) and averaged over the elements sharing a node. That is comparable to commercial
"averaged nodal" output, but **voxel corner singularities** make single-node maxima meaningless,
so report percentiles. Element averages are integration-point-order independent and are what
the benchmarks compare.
