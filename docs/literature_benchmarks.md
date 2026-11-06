# Benchmarks against published results (Barsoum; El-Rich)

This page compares tpmsfem with finite element results published by the Khalifa
University groups of **Imad Barsoum** and **Marwan El-Rich**. Every published
number below is transcribed from the paper, with its location; the transcriptions
are in `benchmarks/literature/data/*.json`. Every tpmsfem number is reproduced by
`benchmarks/literature/run_all.sh`. Raw results are written to `results/literature/`.

These are **independent re-implementations from the paper text**, unlike the CalculiX
comparison, which uses identical decks and agrees to 1e-7. They use:

* a different element type (voxel hex8, vs. Abaqus C3D10 and Ansys tets);
* our own geometry generation;
* our reading of every detail the paper does not state. Each such assumption is listed.

**Numerical uncertainty.** Each tpmsfem value comes from three voxel grids:

* If the three values converge, the value is Richardson-extrapolated. Its uncertainty is
  the grid-convergence index (GCI; Celik et al., *J. Fluids Eng.* 130 (2008) 078001,
  generalised for non-constant refinement ratios).
* If they oscillate, or the changes do not shrink with refinement, the finest value is
  reported. Its uncertainty is the spread of the three.

Voxelisation shifts the volume fraction by a few tenths of a percent between grids. For the
Barsoum lattices, which have a target density, each grid value is therefore scaled back to
the target density with the topology's own Gibson–Ashby exponent before extrapolation.

## Summary

| paper | quantity | agreement |
|---|---|---|
| Van Viet, Zaki & **El-Rich** 2025 | E_y of graded gyroid femur and cranial cells (all 4 values stated in the text) | **−1.1 % … +3.2 %** |
| Ahmed, **Barsoum** & Abu Al-Rub 2022 | periodic-cell E vs. the literature fits their Table 3 lists (Lee et al.; Al-Ketan et al.) | IWP-S **−8 … +0.3 %**, IWP-L **−5 … +5 %**, P-S **−12 … −4 %**; G-S +20 … +48 % against an *experimental* reference |
| Ahmed, Barsoum & Abu Al-Rub 2022 | 3×3×3 specimen E vs. their Ansys Table 3 fits | our elastic moduli are higher: +14 … +40 % (P-S), +41 … +70 % (IWP-S), +32 … +37 % (IWP-L), +81 … +100 % (G-S). For P-S and IWP-S the paper's values equal our elastic-plastic secant modulus at 0.7–0.9 % strain (§2.3) |
| Ahmed, Barsoum & Abu Al-Rub 2022 | modulus reduction by SLM residual stress | effect and magnitude (5–17 %) reproduced; **topology ranking not reproduced** with a uniform inherent strain (§2.4) |

---

## 1. Van Viet, Zaki & El-Rich, PLoS One 20 (2025) e0332104

*"Tuneable multidirectional mechanical attributes of novel sectionally nonlinearly
functionally graded femur and cranial bone implants with triply periodic minimal
surfaces"*, doi:10.1371/journal.pone.0332104, open access (PMC12419660).

**What they did.** These are single gyroid cells whose sheet-thickness parameter `t` is
graded along z:
* Cranial cell (Eq. 1): thick at both ends, thin in the middle.
* Femur cell (Eq. 2): trabecular grading in the lower half, cortical in the upper half.
* Thickness limits from Table 1: t_max = 1.5, t_min = 0.2, t_minc = t_maxt = 0.5.

The material is Ti with E = 104 GPa and ν = 0.3. They used Abaqus C3D10 elements, about
45,000 per cell, and homogenised with periodic boundary conditions. For loading along x or y,
the faces normal to z are traction-free.

**What we did.** Same cells, voxel hex8 at 40/60/90 voxels per cell. We used
`tpmsfem.fem.homogenize` with periodicity in x and y, free z faces, and in-plane loading
(ε_xx, ε_yy, γ_xy), giving E_y = 1/S_yy.

**Assumptions (not stated in the paper):**
* One gyroid period per cell, with the gyroid phase starting at z = 0.
* Sheet solid where |f| ≤ t(z). This is consistent with Table 1: t_max equals max|f| of each
  surface (1.5 for gyroid, 3 for primitive, 5 for IWP), i.e. the "fully dense" end.

| case | vol. fraction | E_y on 3 grids (n=40/60/90) [GPa] | E_y best [GPa] | uncertainty | published [GPa] | diff |
|---|---:|---|---:|---:|---:|---:|
| cranial g0=2 | 0.417 | 25.72 / 25.78 / 25.83 | 26.11 | GCI 1.3 % | 25.3 | +3.2 % |
| cranial g0=3 | 0.348 | 20.13 / 19.62 / 19.69 | 19.69 | spread 2.6 % | - | - |
| cranial g0=4 | 0.307 | 15.69 / 16.57 / 16.35 | 16.35 | spread 5.4 % | - | - |
| cranial g0=5 | 0.280 | 13.58 / 14.14 / 14.28 | 14.34 | GCI 0.5 % | - | - |
| cranial g0=7 | 0.245 | 10.92 / 11.03 / 11.57 | 11.57 | spread 5.6 % | 11.5 | +0.6 % |
| femur gc=0.5 gt=2 | 0.486 | 34.78 / 34.68 / 34.85 | 34.85 | spread 0.5 % | 35.2 | -1.0 % |
| femur gc=1 gt=2 | 0.428 | 27.21 / 27.48 / 27.53 | 27.54 | GCI 0.1 % | - | - |
| femur gc=3 gt=2 | 0.343 | 18.04 / 17.94 / 17.99 | 17.99 | spread 0.6 % | - | - |
| femur gc=7 gt=2 | 0.304 | 13.29 / 13.76 / 13.84 | 13.85 | GCI 0.1 % | 14.0 | -1.1 % |

All four values stated in the text are reproduced within −1.1 % to +3.2 %. That is well
inside the paper's own experiment-vs-FE scatter (up to 15.6 %) and comparable to its
homogenisation-vs-reaction-force check (0.1 %) plus the voxel-vs-C3D10 discretisation
difference. Stiffness falls with the gradient index as the paper describes. E_x and E_y
differ by 2–6 % because the gyroid phase breaks x/y equivalence in a z-graded cell; the
paper reports y.

Not reproduced: Table 2 (cortical part of the femur cell). The paper does not define how
the cortical part was isolated or which axis is "longitudinal". The wave-velocity and Hill
yield-surface results need the Ti plasticity data of their ref. 46, which the paper does not
list.

---

## 2. Ahmed, Barsoum & Abu Al-Rub, Metals 12 (2022) 1344

*"Numerical investigation on the effect of residual stresses on the effective mechanical
properties of 3D-printed TPMS lattices"*, doi:10.3390/met12081344, CC BY.

**What they did.**
* **Lattices:** IN718 (Table 1: E0 = 165 GPa, ν = 0.3, σ_y = 648 MPa). Four types: P-S, G-S
  and IWP-S sheets, and the IWP-L ligament lattice. Relative densities 10, 20 and 30 %, as
  3 × 3 × 3 cells of 10 mm.
* **Process simulation:** sequentially coupled thermo-mechanical SLM simulation in Ansys
  Workbench (HEX8, 0.4 mm lumped layers), followed by plate removal.
* **Mechanical test:** compression and tension along z on 0.25 mm tets, with a bilinear
  elastic-plastic material. The bottom face has a z-symmetry condition. The top is driven
  through a remote point to ±10 % strain.
* **Modulus:** slope of the elastic part of the stress–strain curve, giving the Gibson–Ashby
  fits of Table 3. The same table also lists periodic-cell literature fits.

**Our models.**
* Voxel hex8, density-calibrated level sets.
* IWP-L is the f > c side of the I-WP surface. That is the body-centred "I" labyrinth
  (ligaments along ⟨111⟩), chosen *a priori* because the paper calls it bending-dominated
  (n ≈ 2.4).
* P-S, IWP-S and IWP-L are mirror-symmetric about the specimen mid-planes and run as quarter
  models. The chiral gyroid runs as a full model.

### 2.1 Periodic unit cell vs. the literature rows of Table 3

| topology | rho | E/E0 on 3 grids (density-corrected) | E/E0 best | uncertainty | published | diff |
|---|---:|---|---:|---:|---:|---:|
| P-S | 0.10 | 0.0148 / 0.0155 / 0.0157 | 0.0158 | GCI 0.4 % | 0.0164 | -4.0 % |
| P-S | 0.20 | 0.0417 / 0.0421 / 0.0430 | 0.0430 | spread 3.0 % | 0.0487 | -11.7 % |
| P-S | 0.30 | 0.0878 / 0.0852 / 0.0862 | 0.0862 | spread 3.0 % | 0.0921 | -6.5 % |
| G-S | 0.10 | 0.0311 / 0.0308 / 0.0315 | 0.0315 | spread 2.2 % | 0.0213 | +48.1 % |
| G-S | 0.20 | 0.0672 / 0.0693 / 0.0697 | 0.0697 | GCI 0.1 % | 0.0553 | +26.0 % |
| G-S | 0.30 | 0.1161 / 0.1139 / 0.1162 | 0.1162 | spread 2.0 % | 0.0968 | +20.0 % |
| IWP-S | 0.10 | 0.0250 / 0.0262 / 0.0262 | 0.0262 | GCI 0.0 % | 0.0262 | +0.3 % |
| IWP-S | 0.20 | 0.0727 / 0.0718 / 0.0710 | 0.0672 | GCI 6.8 % | 0.0730 | -8.0 % |
| IWP-S | 0.30 | 0.1289 / 0.1322 / 0.1319 | 0.1319 | spread 2.5 % | 0.1330 | -0.8 % |
| IWP-L | 0.10 | 0.0027 / 0.0028 / 0.0029 | 0.0031 | GCI 6.7 % | 0.0029 | +5.0 % |
| IWP-L | 0.20 | 0.0172 / 0.0173 / 0.0179 | 0.0179 | spread 3.9 % | 0.0183 | -2.2 % |
| IWP-L | 0.30 | 0.0497 / 0.0501 / 0.0506 | 0.0506 | spread 1.8 % | 0.0533 | -5.0 % |

| topology | C1 (ours) | n (ours) | C1, n published |
|---|---:|---:|---|
| P-S | 0.533 | 1.54 | 0.61, 1.57 (Lee et al. 2017 (periodic unit cell)) |
| G-S | 0.478 | 1.18 | 0.51, 1.38 (Al-Ketan et al. 2018 [23]) |
| IWP-S | 0.739 | 1.46 | 0.79, 1.48 (Al-Ketan et al. 2018 [26]) |
| IWP-L | 1.084 | 2.55 | 1.28, 2.64 (Al-Ketan et al. 2018 [26]) |

* **P-S, IWP-S and IWP-L** agree with the periodic-cell values quoted in Table 3 to within a
  few percent; IWP-S at 20 % is −8 %. The P-S exponent (1.54 vs 1.57) and the IWP exponents
  (1.46 vs 1.48; 2.55 vs 2.64) agree.
* **G-S** is 20–48 % stiffer than its Table 3 reference. That reference (Al-Ketan et al.
  2018, *Adv. Eng. Mater.* 20, 1800029) reports **experiments** on microlattices made by
  two-photon lithography. As-fabricated defects make measured moduli lower than ideal-geometry
  FE, so this row is not a like-for-like benchmark.

### 2.2 3 × 3 × 3 specimen, paper set-up

Bottom face u_z = 0. The top face is displaced rigidly in z (remote point, Ansys' default
rigid behaviour). "Published" is the mean of the Table 3 tension and compression fits,
evaluated at ρ.

| topology | rho | E/E0 on 3 grids (density-corrected) | E/E0 best | uncertainty | published | diff |
|---|---:|---|---:|---:|---:|---:|
| P-S | 0.10 | 0.0129 / 0.0135 / 0.0134 | 0.0134 | spread 4.9 % | 0.0118 | +14.0 % |
| P-S | 0.20 | 0.0405 / 0.0399 / 0.0388 | 0.0388 | spread 4.4 % | 0.0324 | +19.7 % |
| P-S | 0.30 | 0.0809 / 0.0819 / 0.0818 | 0.0818 | spread 1.3 % | 0.0586 | +39.6 % |
| G-S | 0.10 | 0.0255 / 0.0257 / 0.0265 | 0.0265 | spread 3.9 % | 0.0133 | +99.7 % |
| G-S | 0.20 | 0.0622 / 0.0607 / 0.0612 | 0.0612 | spread 2.3 % | 0.0336 | +82.2 % |
| G-S | 0.30 | 0.1032 / 0.1063 / 0.1045 | 0.1045 | spread 3.0 % | 0.0578 | +80.9 % |
| IWP-S | 0.10 | 0.0243 / 0.0249 / 0.0253 | 0.0262 | GCI 4.3 % | 0.0186 | +41.0 % |
| IWP-S | 0.20 | 0.0737 / 0.0709 / 0.0696 | 0.0685 | GCI 2.1 % | 0.0457 | +50.0 % |
| IWP-S | 0.30 | 0.1340 / 0.1318 / 0.1315 | 0.1315 | GCI 0.0 % | 0.0773 | +70.1 % |
| IWP-L | 0.10 | 0.0036 / 0.0036 / 0.0036 | 0.0036 | spread 2.1 % | 0.0027 | +32.6 % |
| IWP-L | 0.20 | 0.0185 / 0.0199 / 0.0193 | 0.0193 | spread 7.0 % | 0.0147 | +31.8 % |
| IWP-L | 0.30 | 0.0521 / 0.0532 / 0.0535 | 0.0535 | GCI 0.2 % | 0.0391 | +37.0 % |

| topology | C1 (ours) | n (ours) | C1, n published |
|---|---:|---:|---|
| P-S | 0.565 | 1.63 | tension 0.34, 1.45; compression 0.34, 1.47 |
| G-S | 0.463 | 1.25 | tension 0.29, 1.33; compression 0.29, 1.35 |
| IWP-S | 0.746 | 1.46 | tension 0.36, 1.27; compression 0.38, 1.33 |
| IWP-L | 1.006 | 2.45 | tension 0.72, 2.41; compression 0.72, 2.43 |

The gyroid has no mirror symmetry, so it is a full model. Its grids are limited by CPU memory: the finest
10 % grid has only about 2 voxels across the 0.32 mm wall, so treat G-S 10 % as indicative.

### 2.3 Why the 3 × 3 × 3 moduli differ from the paper

Our 3 × 3 × 3 moduli are close to our periodic ones: P-S 85–95 %, IWP-S 100–102 % and IWP-L
106–116 %. The paper's 3 × 3 × 3 moduli are 58–72 % of the periodic literature values for P-S and
IWP-S, 42–50 % of our periodic G-S, and 73–93 % for IWP-L. We checked three candidate causes:

1. **Unstated boundary condition** (`ahmed_barsoum_2022_sensitivity.py`). A deformable
   instead of rigid top remote point lowers E by 6.9 % (P-S) and 3.5 % (IWP-S).
2. **Unstated lattice phase.** Shifting the TPMS by a quarter cell relative to the specimen
   faces changes E by +12.6 % (P-S) and −7.3 % (IWP-S). A half-cell shift maps the P and I-WP
   sheets onto themselves.
3. **Slope extraction from the elastic-plastic curve** (`ahmed_barsoum_2022_secant.py`).
   We ran the elastic-plastic z-compression of the 3 × 3 × 3 specimen to 1 % strain. The material
   is bilinear IN718 (σ_y = 648 MPa from Table 1, tangent modulus about 2 GPa assumed), on
   quarter models with 24/28/40 voxels per cell for 30/20/10 %. Local yielding starts between
   0.25 % and 0.5 % macroscopic strain. By 1 %, 50–65 % of the elements have yielded and the
   secant modulus is 50–70 % of the initial one. The table compares the published modulus with
   our converged elastic one, and gives the strain at which our secant modulus has fallen to the
   same ratio:

   | topology | rho | ours (converged, elastic) | published | published / ours | our secant / initial at 0.5 % / 1 % | strain where they match |
   |---|---:|---:|---:|---:|---|---:|
   | IWP-S | 0.10 | 0.0262 | 0.0186 | 0.71 | 0.968 / 0.642 | 0.90 % |
   | IWP-S | 0.20 | 0.0685 | 0.0457 | 0.67 | 0.879 / 0.514 | 0.79 % |
   | IWP-S | 0.30 | 0.1315 | 0.0773 | 0.59 | 0.855 / 0.490 | 0.87 % |
   | P-S | 0.10 | 0.0134 | 0.0118 | 0.88 | 0.987 / 0.701 | 0.69 % |
   | P-S | 0.20 | 0.0388 | 0.0324 | 0.84 | 0.990 / 0.686 | 0.75 % |
   | P-S | 0.30 | 0.0818 | 0.0586 | 0.72 | 0.955 / 0.613 | 0.85 % |

   All six published/ours ratios correspond to a secant modulus taken at **0.7–0.9 % strain**.
   That is a plausible extraction window for a slope read off a curve that runs to 10 %
   strain in auto-stepped increments. It also explains why IWP-S, which yields earlier,
   shows the larger gap.

Points 1 and 2 bound the purely modelling-choice uncertainty at about ±10–15 %. Point 3
accounts for the rest of the P-S and IWP-S differences, including their growth with density
and the larger IWP-S gap.

The paper itself supports this reading. Its convergence study (Fig. 4) puts the 3 × 3 × 3 P-S
*yield strength* within 2 % of the periodic value of Lee et al. Yet its 3 × 3 × 3 P-S
*modulus* is 34 % below Lee et al.'s periodic modulus. In an elastic analysis a 3 × 3 × 3
specimen is close to the periodic value, as ours is. A slope that already includes early
local yielding is low in exactly this way.

G-S (published/ours ≈ 0.50–0.55) and IWP-L (≈ 0.75, with the *same* exponent n ≈ 2.45 vs
2.41–2.43) were not part of the secant study.

**Conclusion.**
* The periodic-cell elasticity matches the literature values the paper lists.
* The paper's 3 × 3 × 3 moduli are consistent with ours when the modulus is read as a secant
  at 0.7–0.9 % strain of the elastic-plastic curve.
* The paper does not state the extraction window, the remote-point behaviour or the cell
  phase. From the text alone this benchmark cannot be closed tighter than about ±15 %. Those
  three settings, or the Ansys project, would close it.
* For anyone reusing these Gibson–Ashby fits: they describe a secant-type modulus, not the
  initial elastic modulus.

### 2.4 Residual-stress effect (trend check)

`ahmed_barsoum_2022_rs.py` checks the paper's headline effect with tpmsfem's AM model. This
is **not** a reproduction of their numbers. They used a thermo-mechanical simulation;
tpmsfem currently uses the inherent-strain method, so it has no thermal history.

Set-up: P-S and IWP-S at 30 %, quarter models with 0.42 mm voxels (about the paper's 0.4 mm
lumped AM layer).
* **Build:** layer-by-layer in 0.83 mm super-layers, welded to a rigid plate. One common
  illustrative inherent strain ε* = (−0.005, −0.005, −0.0025), the same for both topologies.
  Bilinear IN718 (J2).
* **Release and test:** cut the in-plane weld, then load with a rigid top face to 0.05 % strain
  in tension and in compression. The apparent modulus is compared with the same test on a
  residual-stress-free lattice.

| topology (rho = 30 %) | E/E0 no RS | E/E0 with RS, tension | compression | reduction ours (tension / compression) | reduction published (tension / compression) | RS von Mises p50 / p95 | yielded during build |
|---|---:|---:|---:|---:|---:|---:|---:|
| P-S | 0.0773 | 0.0699 | 0.0734 | 9.5 % / 5.1 % | 16.8 % / 16.6 % | 403 / 620 MPa | 54 % |
| IWP-S | 0.1404 | 0.1205 | 0.1289 | 14.2 % / 8.2 % | 6.1 % / 4.1 % | 360 / 619 MPa | 60 % |

The published reductions are computed from the Table 3 fits with and without residual
stress at ρ = 30 %.

**Reproduced:**
* Residual stresses lower the apparent build-direction modulus, by a similar order of
  magnitude (5–17 %).
* The reduction is at least as large in tension as in compression. The paper has P-S
  about equal (16.8 vs 16.6 %) and IWP-S larger in tension (6.0 vs 3.9 %); ours is larger in
  tension for both.
* Residual stresses reach yield: 95th-percentile von Mises stress ≈ 620 MPa against
  σ_y = 648 MPa. More than half of the elements yield during the build.

**Not reproduced: the topology ranking.** The paper finds P-S far more sensitive than IWP-S
(17 % vs 4–6 %). With one common inherent strain we find the opposite (5–10 % vs 8–14 %). The
inherent-strain method gives the same incompatibility strain to every topology, so the
residual-stress level is set only by geometric constraint. In the paper's thermal model it
also depends on the heat flow through each topology: wall thickness, connectivity to the
plate, heat accumulation in thicker P-S sections.

Matching the ranking therefore needs either inherent strains calibrated per topology (from
thermal simulations or measured distortion), or the thermo-mechanical layer model on the
roadmap. For TPMS implants, where the topology is the design variable, that is the main
lesson of this benchmark.

---

## Reproducing

```bash
./benchmarks/literature/run_all.sh                       # all of the above (resumable)
python benchmarks/literature/ahmed_barsoum_2022.py --report
python benchmarks/literature/van_viet_elrich_2025.py --report-only
```

CPU timings in this repository's runs (4-core container, no GPU, mostly single-threaded AMG):
* El-Rich: about 40 min.
* Barsoum periodic: about 45 min.
* Barsoum 3 × 3 × 3: about 2.5 h. The full gyroid models dominate.
* Residual-stress study: about 30 min per topology. Most of that is the plastic layer-by-layer build.
* Secant study: about 10 min per case.

Raw results are copied to `benchmarks/literature/results/`. On a GPU, the matrix-free solver
allows the finer grids that would tighten the gyroid and 10 % uncertainties.

## Solver issues found and fixed while running these benchmarks

* **Convergence reference.** Perfectly homogeneous periodic cells have an exactly zero
  residual. The convergence test now uses the non-cancelling element force magnitude as its
  reference (cf. Abaqus' time-averaged force).
* **Boundary-condition jump.** With plasticity, evaluating the residual with a prescribed-
  displacement increment imposed fakes yielding in the boundary layer, and Newton stalled
  (IWP-S and P-S lattices). Increments now enter through a consistent K·Δu predictor with the
  elastic tangent. Newton then converges in 0–4 iterations.
* **Robustness.** Non-converged increments are automatically halved (cut-back).
* **Preconditioner cache.** Changing only prescribed values no longer rebuilds it.
