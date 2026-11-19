# Process-scale models: laser scan, melt pool, solidification, surface roughness

`tpmsfem.process` adds melt-pool-scale models of laser powder-bed fusion to the part-scale
thermo-mechanical AM model. They form one chain:

    scan path + laser  ->  melt pool (meltpool.py)  ->  G, R, cooling rate, melting times
                                         |                        |
                                         v                        v
                         bead size -> surface roughness   solidification microstructure
                                     (roughness.py)       grains (solidification.py, ExaCA export)
                                                          cells, Nb segregation (phasefield.py)

Everything runs on the NumPy/CuPy backend, so the same code is GPU-ready. The figures and
numbers below come from a 4-core CPU.

## 1. Melt pool (`process/meltpool.py`)

**Physics.** An explicit finite-volume enthalpy method on a structured voxel grid,
`dH/dt = div(k grad T) + q_laser - q_surface`. It includes:

* **Enthalpy.** Sensible and latent heat (solidus to liquidus), inverted to `T` by table
  look-up, so energy is conserved to round-off (tested to 1e-14).
* **Conductivity.** `k(T)` at faces as a harmonic mean, plus an optional anisotropic
  enhancement factor in the liquid. This is the standard conduction-model stand-in for
  Marangoni convection, which is not resolved.
* **Powder.** A powder layer with reduced conductivity, densified when it melts.
* **Laser.** A Gaussian (or top-hat) surface source of absorbed power `A·P`, normalised exactly
  on the grid. An optional absorption depth spreads it into the material.
* **Scan paths.** Scan vectors and raster hatching with jumps.
* **Surface losses.** Radiation, convection, and evaporation (Clausius–Clapeyron vapour
  pressure with a Hertz–Knudsen–Langmuir mass flux and back-condensation). Evaporation caps
  the surface temperature near boiling, at about 3000 to 3500 °C here.
* **Symmetry.** A symmetry plane under a single track halves the cost.
* **Recorded per cell.** Peak temperature, number of melting events, last melting and
  solidification times, and, at solidification, cooling rate, `G = |grad T|`, its direction
  and `R = cooling rate / G`. Probes record surface histories, as a thermal camera does.

**Verification.**
* **Exact solution.** A moving Gaussian source on a semi-infinite body, the Eagar–Tsai
  integral, with constant properties. With 10 µm cells the solver agrees within 1 % at all
  checked points. The analytic routine itself reproduces Rosenthal's point-source solution
  to 6 digits.
* **Energy balance.** With latent heat and surface losses, the enthalpy balance closes to
  1e-9 or better.

**Validation, NIST AM-Bench 2018-02** (single tracks on bare IN625, NIST AMMT testbed): see §4.

## 2. Solidification microstructure (`process/solidification.py`)

* **G–R map.** Thermal gradient and solidification velocity at the last liquidus crossing of
  every melted cell, the cooling rate `G·R`, and the morphology indicator `G/R`.
* **Cell and dendrite spacing.** Kurz–Fisher (`4.3 (ΔT0 D Γ / k)^¼ G^-½ R^-¼`), Hunt, or a
  user-fitted cooling-rate law.
* **Columnar-to-equiaxed transition.** Hunt's (1984) criterion with a tip-undercooling growth
  law.
* **Grains.** A reduced cellular-automaton model of epitaxial, competitive columnar growth.
  * Melted cells re-solidify in time order.
  * Each one takes the grain of an already solid neighbour. The choice is weighted by how
    well that grain's nearest <100> axis aligns with the local growth direction (grad T at
    solidification), so well-oriented grains crowd out the others.
  * Equiaxed grains nucleate where Hunt predicts equiaxed growth.
  * Outputs: grain maps, inverse-pole-figure colours and texture fractions.
* **ExaCA export.** `x,y,z,tm,tl,cr` CSV (SI units) for ORNL's open-source GPU
  cellular-automaton code. Use it when the full decentred-octahedron kinetics matter.
* **Cells and dendrites, resolved** (`process/phasefield.py`, §5.3). A 2-D quantitative
  phase-field model grows the cell array itself at a melt-pool (G, R). It gives the spacing,
  the cell or dendrite morphology and the Nb segregation between cells, which the spacing
  laws above only estimate.

**Constants.** The alloy constants (Gibbs–Thomson, diffusivity, partition coefficient,
nucleation density, growth law) are inputs. The defaults are generic Ni–Nb values and are
**not calibrated for IN625**.

## 3. Surface roughness (`process/roughness.py`)

The as-built topography of a surface tilted by θ from the build direction is the
superposition of three parts:
* **Layer staircase.** Peak-to-valley `t |cos θ|`, so `Ra = t |cos θ|/4` for sharp steps.
* **Melt-pool rounding.** Morphological closing with the bead radius, taken from the
  melt-pool model.
* **Partially fused powder.** Spheres from the powder size distribution, stuck on as a
  Poisson process. Their density rises on down-facing surfaces and falls on up-facing ones.

Roughness parameters follow ISO 4287 and 25178 (Ra, Rq, Rz; Sa, Sq, Sz). `roughness_map` puts
the model onto any triangulated surface (for example a TPMS implant) by facet inclination.

**Validation, Kasperovich et al. 2021** (LPBF IN718, 30 µm layers): see §5.

## 4. AM-Bench 2018-02 single tracks (melt pool)

NIST laser tracks on bare IN625 plate, AMMT testbed: 1070 nm, D4σ = 170 µm (Gaussian 1/e²
radius 85 µm), true powers 137.9 / 179.2 / 179.2 W. Measurements are from the
[AMB2018-02 description](https://www.nist.gov/ambench/amb2018-02-description) and its
melt-pool-length, cross-section and cooling-rate result pages.

**Model set-up.**
* 5 µm cells, half-symmetric domain 1.9 × 0.3 × 0.2 mm, 1.5 mm track.
* Evaporation on. The liquid is not enhanced: enhancing it in depth would have needed a
  factor of about 7.

**Calibration on case B**, NIST's calibration case. Two parameters are fitted to its width
and depth:
* absorptivity A = 0.323;
* absorption depth δ = 15.3 µm (exponential), the usual effective treatment of the shallow
  vapour depression at about 1.6 MW/cm² peak intensity.

Without δ, absorptivity alone (0.206) matches the width but gives only 22 µm depth. Cases A
and C are predictions:

| case | P, v | quantity | measured (NIST AMMT) | tpmsfem | diff |
|---|---|---|---:|---:|---:|
| A (prediction) | 137.9 W, 400 mm/s | melt-pool width | 147.9 ± 3.7 µm | 141.7 µm | −4 % |
| A (prediction) | 137.9 W, 400 mm/s | melt-pool depth | 42.5 ± 1.7 µm | 47.2 µm | +11 % |
| A (prediction) | 137.9 W, 400 mm/s | melt-pool length (solidus) | 300 ± 12 µm | 357 µm | +19 % |
| B (calibration) | 179.2 W, 800 mm/s | width / depth (fitted) | 123.5 / 36.0 µm | 123.3 / 36.0 µm | 0 % |
| B (calibration) | 179.2 W, 800 mm/s | melt-pool length (solidus) | 359 ± 20 µm | 414 µm | +15 % |
| C (prediction) | 179.2 W, 1200 mm/s | melt-pool width | 106.0 ± 1.4 µm | 100.9 µm | −5 % |
| C (prediction) | 179.2 W, 1200 mm/s | melt-pool depth | 29.6 ± 0.6 µm | 24.7 µm | −17 % |
| C (prediction) | 179.2 W, 1200 mm/s | melt-pool length (solidus) | 370 ± 10 µm | 344 µm | −7 % |
| A / B / C | | surface cooling rate 1290→1190 °C | 1.2 / 1.1 / 1.9 ×10⁶ K/s | 3.3 / 5.9 / 8.9 ×10⁶ K/s | 3–5× too high |

![track B](img/meltpool_track_B.png)

**Reading the results.**
* **Widths** are predicted within 5 %.
* **Depths** are within −17 % to +11 %, but the speed trend is too steep: too deep at
  400 mm/s, too shallow at 1200 mm/s. One fixed absorption depth cannot follow a vapour
  depression that deepens at low speed. A speed- or intensity-dependent δ, or a
  free-surface model, would.
* **Lengths** are within −7 % to +19 %. The two NIST machines differ by about 2× in measured
  length (CBM 659–782 µm), and NIST left the cause open.
* **Cooling rates** are 3–5× higher than NIST's thermography. The camera's spatial and
  temporal resolution and the emissivity assumption smear the steep solidification-front
  gradient, which likely contributes. NIST also reports that the two machines' cooling rates
  "differ drastically". The model, however, also omits fluid flow in the melt pool, which reshapes the temperature
  field near the solidification front. The predicted rates are therefore not validated.
* **Cost.** One case-B run takes 2.5 minutes on 4 CPU cores (2,528 explicit steps, 0.96 M cells).

## 5. Solidification and grains (track B)

`examples/05_lpbf_meltpool_microstructure.py` (summary in `results/process/summary.json`):

| quantity (re-solidified cells of track B, 5–50–95th percentiles) | value |
|---|---|
| thermal gradient G | 5.1e+05 / 7.3e+06 / 2.5e+07 K/m |
| solidification velocity R | 0.017 / 0.086 / 0.28 m/s (scan speed 0.8 m/s) |
| cooling rate G·R at the liquidus | 7.5e+04 / 5.2e+05 / 1.6e+06 K/s |
| primary spacing, Kurz–Fisher (generic constants) | 1.16 / 1.82 / 5.89 µm |
| Hunt CET (CMSX-4 constants): columnar / mixed / equiaxed | 43 / 36 / 22 % of the fusion zone |
| grains in the fusion zone (30 µm substrate grains, purely epitaxial) | 110; <100> within 15° of the build direction in 16 % of the volume (random: about 10 %) |
| ExaCA export | 18,296 cells (`results/process/exaca_track_B.csv`) |
| 5-track hatch on 20 µm powder | up to 3 melting events per cell; 638 grains in the fused region |

* **Cooling at the liquidus.** The cooling rate at the liquidus crossing is about 10× lower
  than the surface rate between 1290 and 1190 °C: latent heat slows cooling in the mushy zone.
* **Texture.** Competitive growth gives a mild <100> texture along the build direction.
* **First layer.** The grains of the first layer are inherited epitaxially from the plate.

![5-track hatch](img/meltpool_hatch.png)

### 5.1 Grains against NIST's EBSD map (case B)

**Data.** NIST's example EBSD map of a case-B track: a transverse cross-section, Oxford
`.ctf` with 1218 × 890 points at 0.14 µm
([AMB2018-02_CaseB_EBSDdata.zip](https://s3.amazonaws.com/nist-midas/1889/AMB2018-02_CaseB_EBSDdata.zip)).
It is analysed with `tpmsfem.process.ebsd`: CTF reader, Bunge orientations, cubic
disorientation, grain segmentation at 10° and kernel misorientation. The script is
`benchmarks/literature/nist_ambench2018_tracks_ebsd.py`.

**Test.** The melt pool grows epitaxially from the plate's grains, so the measured
substrate fixes the initial condition. The measured orientations *inside* the model's fusion
line are hidden and regrown from the substrate:
* in the order of the simulated solidification times;
* along the simulated growth direction (grad T at solidification) of the steady-state
  case-B section;
* on the 0.28 µm EBSD grid (2 × 2 binning).

The prediction is compared with the measurement pixel by pixel.

**Placement.** The fusion line is the calibrated model's, which matches NIST's case-B mean
size. Its centre is put where the elongated grains change their lean (x = 82.4 µm). Depth is
measured from the local surface.

| regrowth of the melt pool (3116 µm², 39,749 px) | pixels within 10° of measured | grains (10°) | area-weighted grain size |
|---|---:|---:|---:|
| measured | — | 92 | 10.1 µm |
| tpmsfem, epitaxy along the simulated thermal gradient (trace back to the substrate) | **27.5 %** | 83 | 12.3 µm |
| tpmsfem, upstream-neighbour rule (`grow_grains(rule="upstream")`) | 26.6 % | 50 | 13.1 µm |
| tpmsfem, competitive growth (`grow_grains`, default) | 26.4 % | 45 | 14.5 µm |
| tpmsfem, dendrite-arm capture rule | 21.0 % | 42 | 14.5 µm |
| baseline: copy the nearest fusion-line grain | 25.8 % | 51 | 13.0 µm |
| baseline: copy the grain straight below | 19.8 % | — | — |
| random orientations | 0.5 % | — | — |

![EBSD regrowth](img/ebsd_caseB_regrowth.png)

**Reading the results.**
* **The fan of columnar grains comes from epitaxy.** It converges toward the top centre, and
  so does the simulated one. About a quarter of the melt pool, point by point, is the plate
  grain directly below or beside it, extended along the thermal gradient.
* **The gradient direction helps a little.** Following the simulated gradient is slightly
  better than following the fusion-line normal (27.5 % vs 25.8 %), and it reproduces the
  grain count: 83 against 92 measured.
* **Competition adds nothing here.** Selecting grains by <100> alignment does not improve
  point-wise agreement over this short, 36 µm growth distance. It merges grains (45 against
  92). An earlier, slightly smaller fusion-zone definition gave the competitive rule a texture
  match (9.6 % against 10.9 % <100> within 15° of the build direction). That match vanished
  with the smoother definition, so it is not claimed.
* **The fusion line checks out independently.** The model's fusion line follows the
  measured boundary between columnar and equiaxed grains.
* **The test has a ceiling.**
  * It is a 2-D test: grains can enter the section from substrate above or below the plane.
  * The fusion line is the model's, not a measured one.
  * The unmelted plate is fine-grained (median equivalent diameter 4.2 µm, many annealing
    twins).
  * The model resolves the thermal field at 5 µm.

### 5.2 3-D test: where the section's grains come from

**What the 2-D test assumed.** In a moving melt pool the growth direction also points
forward along the scan, so a grain seen in the transverse section may have started on the
fusion boundary *behind* it. The 2-D regrowth assumed every grain started in the measured
plane.

**Method.** `benchmarks/literature/nist_ambench2018_tracks_ebsd3d.py` traces each melt-pool
pixel of the section backwards through the simulated **3-D** steady-state growth-direction
field of case B, until it leaves the melt pool. That point is where its grain started.

**Origins.** All 39,749 traces reach the fusion boundary, after a median path of
11 µm.
* **Distance behind the section.** The median origin lies 0.8 µm behind
  the section, the 90th percentile 3.6 µm.
* **Where.** Grains near the top centre start furthest back; along the sides and bottom they
  start in the plane.
* **In-plane share.** 64 % of the pixels have an origin within
  ±1.4 µm of the plane, half the measured median substrate grain size of
  2.7 µm.

**Parameter-free check.** Give every pixel the measured in-plane substrate orientation at its
origin's (y, z), then bin the agreement by the origin's distance from the plane:

| origin behind the section | share of melt-pool pixels | within 10° of measured |
|---|---:|---:|
| 0–0.5 µm | 39.5 % | 51.2 % |
| 0.5–1 µm | 15.6 % | 30.8 % |
| 1–1.5 µm | 11.1 % | 11.8 % |
| 1.5–2 µm | 8.3 % | 7.4 % |
| 2–3 µm | 11.3 % | 7.8 % |
| 3–4 µm | 6.9 % | 7.2 % |
| 4–6 µm | 7.2 % | 6.3 % |
| random orientations | | 0.6 % |

* **Agreement follows the origin's distance.** Half of the pixels whose grains start within
  0.5 µm of the measured plane are predicted correctly. Agreement falls to a residual 6–8 %
  once the origin lies 1.5 µm or more behind it. The residual exceeds random because large
  substrate grains reach a few µm out of the plane.
* **The prediction is confirmed.** The point-wise failures of the 2-D test are explained by
  grains from substrate NIST did not measure, and their share is predicted by the 3-D thermal
  field.

**Hybrid prediction.** Use the measured orientations for in-plane origins, and a
statistically equivalent 3-D substrate elsewhere: Voronoi grains whose orientations are drawn
from the measured substrate, 5 realisations.
* **All pixels:** 26.7 ± 0.6 % within 10°.
* **In-plane origins:** 41 %.
* **Upstream origins:** 1.5 %, close to random.
* **Ceiling.** The 2-D result (27.5 %) is close to the best that can be expected from a single
  section. Doing better needs the substrate in 3-D, for example from serial sections.

**Grain count.** The hybrid predicts 236 grains in the section with
2.7 µm stand-in substrate grains, against
92 measured. With the area-weighted size
(7.9 µm) it predicts 121. The
unmeasured 3-D substrate grain size sets the count of the upstream-origin grains.

![3-D origin of the melt-pool grains](img/ebsd_caseB_3d.png)

The remaining NIST track files (SEM and EBSD maps of tracks 3, 5 and 10, the 10 optical
cross-sections of mds2-3830) did not download: they returned server errors, HTTP 500, on
2026-11-07.

### 5.3 Cells and Nb segregation: 2-D quantitative phase field

**Why a phase field.** The grain model (§2, §5.1–5.2) places grains but not the cells inside
them.
* In LPBF IN625 each grain solidifies as an array of sub-micron cells.
* Nb is rejected into the liquid between the cells. The Laves phase and NbC that form there
  control the heat-treatment response, corrosion and fatigue.
* The spacing laws (Kurz–Fisher, Hunt) only give an estimate.

`process/phasefield.py` grows the cell array itself at a melt-pool (G, R).

**Model.** The quantitative thin-interface phase field of a dilute binary alloy in
directional solidification (Karma 2001; Echebarria, Folch, Karma & Plapp 2004).
* **Variables.** A phase field `φ` (+1 solid, −1 liquid) and the supersaturation `U`.
* **Temperature.** Frozen: `T = T0 + G (z − V t)`, with the melt-pool G and R as inputs.
* **Thin-interface terms.** An anti-trapping current, plus `λ = a1 W0/d0` and
  `τ0 = a2 λ W0²/D` (a1 = 0.8839, a2 = 0.6267). These remove the spurious interface effects
  (solute trapping, a jump in chemical potential, kinetic undercooling) to second order in the
  interface width `W0`.
* **Anisotropy.** Four-fold, `1 + ε4 cos 4(ψ − ψ0)`; the crystal can be tilted from the
  gradient.
* **Diffusion.** One-sided: no diffusion in the solid.

**Numerics.**
* **Grid.** Explicit conservative finite volumes with `dx = 0.8 W0`, periodic across the growth
  direction.
* **Moving window.** The window follows the front. The rows that leave it are kept as the
  solidified Nb map.
* **Kernel.** A fused numba kernel runs at 7 ns per cell and step on 4 CPU cores, 18× the NumPy
  array path; the two agree to 1e-15. The array code also runs on CuPy GPUs.
* **Alloy.** IN625 is quasi-binary Ni–4 wt% Nb, as in Ghosh et al. (2018): k = 0.48,
  m = −10.5 K/wt%, D = 3e-9 m²/s, Γ = 3.65e-7 K m, ε4 = 0.02, so ΔT0 = 45.5 K and d0 = 8.0 nm
  (`phasefield.IN625_NB`).

**Measurements.**
* **Spacing.** The width divided by the number of Nb-rich cell boundaries. Boundaries are counted,
  as on a micrograph, on lines 25–50 % of the tip height below the leading tip. A boundary
  counts when it stands 0.2 c∞ above the neighbouring cell cores.
  * A relative threshold failed: the strongly enriched liquid droplets set the scale and the
    solid boundaries dropped below it.
  * Counting tips failed too: at high G, the active tips of a coarsening array are further
    apart in height than any fixed band.
* **Tip compositions.** c_s* is the solid composition just behind each active tip, c_l* the
  liquid peak just ahead of it.

**Verification** (`tests/test_phasefield.py`, `benchmarks/literature/ghosh_2017_phasefield.py`).
* **Exactness.** Solute is conserved to 1e-14 in a closed box.
* **Stability limits.** A planar front stays planar below the constitutional limit
  `V_cs = D G/ΔT0`, sitting at the solidus. It breaks into cells above.
* **Growth rates.** Small front perturbations follow the Mullins–Sekerka dispersion relation
  (one-sided, frozen temperature, non-quasi-static). With D/V = 20 W0 the growth rates agree
  within 5 % across the unstable band, at W0 = 2.5 d0 and 1.25 d0.
* **At AM speeds** (Ni–5Nb, G = 1e7 K/m, ε4 = 0), the ratio of phase-field to Mullins–Sekerka
  growth rate is:

| V | W0 = 10 nm | W0 = 5 nm |
|---|---|---|
| 0.03 m/s (D/V = 10 and 20 W0) | 0.88–0.94 | 0.88–0.96 |
| 0.1 m/s (D/V = 3 and 6 W0) | 0.57–0.76 | 0.70–0.83 |

  The thin-interface model stays quantitative where D/V ≳ 10 W0, i.e. up to about 0.03 m/s at
  W0 = 10 nm. Faster growth needs a thinner interface, and the cost rises as W0^-4. Ghosh et al.
  (2017) reach the same conclusion. They find the anti-trapping current "not adequate to
  eliminate solute-trapping at these high solidification rates".

**Code-to-code: Ghosh, Ma, Ofori-Opoku & Guyer, MSMSE 25 (2017) 065002.** Their setup is the
same model class:
* Ni–5 wt% Nb, k = 0.48, m = −10.5 K/wt%, D = 3e-9 m²/s, Γ = 3.65e-7 K m, ε4 = 0.03;
* W0 = 10 nm, dx = 8 nm, 2-D, frozen temperature, G = 1e7 K/m.

| V | cooling rate G V | tpmsfem spacing | Ghosh 2017 | c_s* tpmsfem / Ghosh | c_l* tpmsfem / Ghosh |
|---|---|---|---|---|---|
| 0.03 m/s | 3e5 K/s | 0.604 µm | 0.604 µm (fit), 0.71 µm (run quoted) | 3.78 / 3.65 wt% | 6.81 / 6.87 wt% |
| 0.05 m/s | 5e5 K/s | 0.402 µm | 0.452 µm (fit) | 3.85 wt% | 6.68 wt% |
| 0.1 m/s | 1e6 K/s | 0.221 µm (W0 = 10 nm), **0.304 µm (W0 = 5 nm)** | 0.23 µm (run quoted), 0.304 µm (fit) | 3.94 / 4.04 wt% | 6.42 / 6.94 wt% |
| 0.2 m/s | 2e6 K/s | 0.250 µm | 0.205 µm (fit) | 4.37 wt% | 6.46 wt% |

Their fit is `PDAS = 800 µm (G V)^-0.57`. Hunt's law gives 0.56–0.89 µm and Kurz–Fisher's
1.2–2.0 µm over the same range.

![phase field vs Ghosh et al. 2017](img/pf_ghosh2017.png)

**Application: AM-Bench case B against the measured IN625 cells**
(`benchmarks/literature/nist_ambench2018_tracks_pf.py`).
* **Inputs.** The calibrated case-B melt pool gives G and R at four heights up the centreline
  of the x = 0.7 mm section. R is `v |n_x|`, which is smoother on 5 µm voxels than cooling
  rate / G.
* **Run.** The phase field runs at W0 = 10 nm, 4 µm wide, over 12 µm of growth.
* **Measurement.** Ghosh, Ma, Levine, Ricker, Stoudt, Heigel & Guyer, JOM 70 (2018) 1011,
  measured an average primary spacing of about 0.2–1.0 µm by SEM. The track was on an IN625
  plate at 195 W / 800 mm/s, the nominal setting nearest case B. The cells are coarse at the
  pool bottom and fine towards its rear. Their own phase field gives 0.20–0.43 µm.

| height above the fusion boundary | G | R | D/V | spacing, phase field | Hunt | c_s* | c_l* |
|---|---|---|---|---|---|---|---|
| 0 µm (lowest melted row) | 2.4e7 K/m | 0.027 m/s | 11 W0 | 0.57 µm | 0.56 µm | 3.23 wt% | 5.83 wt% |
| 5 µm | 1.25e7 K/m | 0.074 m/s | 4.1 W0 | 0.37 µm | 0.60 µm | 3.34 wt% | 5.49 wt% |
| 10 µm | 6.6e6 K/m | 0.092 m/s | 3.3 W0 | 0.35 µm | 0.79 µm | 3.35 wt% | 5.40 wt% |
| 15 µm | 3.5e6 K/m | 0.133 m/s | 2.3 W0 | 0.30 µm | 0.98 µm | 3.44 wt% | 5.29 wt% |

![phase field, AM-Bench case B](img/pf_ambench_caseB.png)

**Reading the results.**
* **Code to code.** At the shared W0 = 10 nm the phase field reproduces Ghosh et al.'s cell
  spacing within their own scatter: their quoted run at 3e5 K/s lies 18 % above their fit.
* **Convergence at 0.1 m/s.** Halving W0 to 5 nm raises the spacing from 0.221 to 0.304 µm
  (+38 %), onto Ghosh's fit. At D/V = 3 W0 a 10 nm interface therefore under-predicts the
  spacing by about a quarter. The close match to Ghosh's 0.23 µm quoted run is most likely two
  codes sharing the same under-resolution. The 5 nm result is not proven converged either
  (Mullins–Sekerka rates still 17–30 % low at D/V = 6 W0). At 0.03 m/s the tip
  compositions agree within 4 % (solid) and 1 % (liquid). The effective partition
  coefficient c_s*/c_l* rises with V, from 0.55 to 0.68 against k = 0.48.
* **Against the measurement.** All four case-B spacings lie inside the measured 0.2–1.0 µm.
  * They fall from 0.57 µm at the fusion boundary to 0.30 µm higher up, as the micrographs
    show.
  * Hunt's law trends the other way (0.56 → 0.98 µm), because G falls faster than R rises.
  * The measurement is a single range from one micrograph, so this is a consistency check,
    not a point-wise validation.
* **The Nb segregation is of the right size.** At the fusion boundary the cell cores hold
  3.2 wt% Nb and the liquid at the tips 5.8 wt% (Ghosh 2018 phase field: 3.7 / 5.1 wt%).
  Deeper in the mushy zone, intercellular droplets reach 9 wt% in the 2 µm window. Ghosh 2017
  report 15.8 wt% in a 40 µm domain. The value keeps rising with depth because the binary model
  has no eutectic or Laves reaction to cap it.
* **Limits.**
  * Only the fusion-boundary point satisfies D/V ≥ 10 W0; the three points above it have
    D/V = 2.3–4 W0. The 0.1 m/s convergence test suggested they might be 30 % low. The
    convergence series below, on this path, does not confirm that: at 10 µm the spacing changes
    by less than one window step (8 %) between W0 = 10, 6.5 and 4.1 nm.
  * The model is 2-D, and 2-D cell arrays are coarser than 3-D ones.
  * The temperature field is frozen (no latent heat at this scale).
  * IN625 is treated as a binary.

**The whole path, with solute trapping** (`benchmarks/literature/pf_path_trapping.py`).
* **Path.** The same centreline as above, extended to the surface (7 points, 0–30 µm above the
  fusion boundary). R rises from 0.027 to 0.29 m/s, so D/V falls from 110 to 10 nm.
* **Trapping.** The continuous growth model (Aziz & Kaplan) as velocity-corrected alloy
  parameters (`DiluteAlloy.at_velocity`, V_D = 1 m/s): k rises from 0.49 to 0.60, and the
  freezing range falls from 43 to 29 K.
* **Resolution.** The interface width follows the diffusion length, W0 = min(10 nm, D/5R),
  i.e. D/V ≥ 5 W0 everywhere (2.1 nm at the surface). The full criterion D/V ≥ 10 W0 would
  cost ~300 CPU-hours here; the cost scales as W0⁻⁵.
* **Convergence check.** At 10 µm, W0 = 10, 6.5 and 4.1 nm (D/V = 3.3, 5 and 8 W0).
* **Trapping on and off.** One run without trapping at the same W0, 20 µm up.
* **Windows and cost.** 2.4 × 1.2 µm windows (1.6 × 0.8 µm at the top two points), 6 µm of
  growth, the last 40 % averaged. An adaptive stable time step (×1.7 faster) and checkpoints
  make the runs feasible: about 6 h each for the top two points on two cores.

| height above the fusion boundary | R | k(V) | W0 | spacing | c_s* (cores) | c_l* (tip liquid) | c_s*/c_l* | 95th pct. of the solid |
|---|---|---|---|---|---|---|---|---|
| 0 µm | 0.027 m/s | 0.49 | 10 nm | 0.57 µm | 3.27 wt% | 5.77 wt% | 0.57 | 8.1 wt% |
| 5 µm | 0.074 m/s | 0.52 | 8.1 nm | 0.37 µm | 3.46 wt% | 5.52 wt% | 0.63 | 7.1 wt% |
| 10 µm | 0.092 m/s | 0.52 | 10 / 6.5 / 4.1 nm | 0.35 / 0.32 / 0.33 µm | 3.47–3.50 wt% | 5.24–5.72 wt% | 0.61–0.66 | 6.2–7.2 wt% |
| 15 µm | 0.133 m/s | 0.54 | 4.5 nm | 0.28 µm | 3.62 wt% | 5.51 wt% | 0.66 | 6.6 wt% |
| 20 µm | 0.173 m/s | 0.56 | 3.5 nm | 0.30 µm | 3.72 wt% | 5.50 wt% | 0.68 | 6.0 wt% |
| 20 µm, no trapping | 0.173 m/s | 0.48 | 3.5 nm | 0.28 µm | 3.60 wt% | 5.79 wt% | 0.62 | 6.7 wt% |
| 25 µm | 0.257 m/s | 0.59 | 2.3 nm | 0.30 µm | 3.90 wt% | 5.56 wt% | 0.70 | 5.6 wt% |
| 30 µm | 0.287 m/s | 0.60 | 2.1 nm | 0.31 µm | 3.90 wt% | 5.55 wt% | 0.70 | 6.0 wt% |

The 95th percentile is that of the Nb content of the solid that left the window: a measure of
the intercellular enrichment.

![phase field up the pool with trapping](img/pf_path_trapping.png)

* **The cells refine from the fusion boundary, then level off.** The spacing goes from
  0.57 µm at the boundary to 0.37 µm 5 µm above it, and stays at 0.28–0.31 µm from 10 µm up
  to the surface.
  * This lies within the measured 0.2–1.0 µm and within Ghosh et al.'s phase field
    (0.20–0.43 µm). The micrographs show the same coarse-to-fine progression.
  * Hunt's law trends the other way (0.56 → 2.9 µm), because G falls 100-fold while R rises
    only 10-fold.
* **The resolution is adequate.** At 10 µm, W0 = 10, 6.5 and 4.1 nm give 0.35, 0.32 and
  0.33 µm, within one window step. So D/V = 5 W0 is converged for the spacing here, and so are
  the W0 = 10 nm values in the table above.
* **Segregation weakens up the pool.**
  * The cell cores carry 3.27 wt% Nb at the fusion boundary and 3.90 wt% at the top, against
    4 wt% in the alloy.
  * The effective partition c_s*/c_l* rises from 0.57 to 0.70. The intercellular enrichment
    (95th percentile) falls from 8.1 to 5.6–6.0 wt%.
  * Less Laves phase is therefore expected in the upper part of each pool than at its
    fusion boundary.
* **Trapping accounts for part of this.** At 20 µm, switching it off (same W0):
  * lowers the cores from 3.72 to 3.60 wt% Nb;
  * raises the tip liquid from 5.50 to 5.79 wt% and the 95th percentile from 6.0 to 6.7 wt%;
  * raises the tip undercooling from 32 to 41 K.
  The rest of the trend comes from the faster growth itself. At the fusion boundary
  trapping changes nothing (k = 0.49 against 0.48).
* **No planar front within the pool.** With trapping, the absolute-stability velocity falls
  from 0.78 to 0.40 m/s at the top, where R = 0.29 m/s (R/V_abs = 0.72). The cells there are
  shallow, with weak boundaries, but still cells.
* **Limits.**
  * V_D = 1 m/s is assumed; values of 0.5–2 m/s are used for Ni–Nb.
  * The CGM enters as effective parameters. Kinetic undercooling and solute drag are not
    included.
  * The model is 2-D, with a frozen temperature field and a binary alloy.
  * The narrower windows at the top quantise the spacing in steps of about 10 %.

### 5.4 3-D grain cellular automaton

**Why.** §5.1–5.2 place grains by rules: the best-aligned neighbour, or a trace back along
the thermal gradient. A cellular automaton *grows* them instead:
* each grain's dendrite envelope advances along its own <100> axes;
* the advance rate is the tip velocity at the local undercooling;
* so competition between grains, nucleation ahead of the front, and regrowth across tracks
  and layers all follow from the kinetics.

It is also the tool for multi-layer builds, where grain size and texture develop over many
layers.

**Model** (`tpmsfem.process.grain_ca`; the formulation of Gandin & Rappaz 1997 and of ExaCA,
Rolchigo et al. 2022):
* **Cells.** Cubic cells, each solid cell carrying a grain id and each grain a cubic
  orientation.
* **Thermal history.** Per cell, melting events (t_m, t_l, cooling rate). After the liquidus
  the cell undercools linearly.
* **Growth and capture.** Every solid cell next to the liquid carries an octahedral envelope
  aligned with its grain. The envelope grows at the KGT tip velocity v(ΔT) and captures the
  liquid neighbours whose centres it reaches; the deepest envelope wins.
* **Decentring.** A captured cell gets a decentred envelope that keeps its parent's tip, so
  growth follows the crystal axes, not the grid.
* **Nucleation.** Optional: Gaussian nucleation undercoolings at random sites.
* **Growth mode.** Optional heat-flow bias α. Each envelope's frame is turned so that the
  grain's <100> nearest to the growth direction (∇t_l) rotates towards it by α times their
  angle. α = 0 gives dendrites growing along <100>. α = 1 is the cellular limit, where every
  grain grows along the heat flow.
* **Epitaxy.** Unmelted cells at the fusion boundary seed the pool.
* **Scan strategies.** The steady thermal history of one track (from `MeltPoolModel`) is shifted
  along every scan vector. This includes remelting and regrowth across tracks and layers, but
  not the preheating of one track by another.
* **Implementation.** A serial numba kernel over the interface cells. The case-B pool
  (2.7 M cells at 0.5 µm) runs in about 20 s.

**Verification** (`tests/test_grain_ca.py`):
* **Grid independence.** One grain in a uniformly undercooling melt grows as an octahedron
  along its crystal axes, whatever its tilt to the grid:
  * extent along x relative to that along <100>: 0.870 at 30° (exact 0.866) and 0.707 at 45°
    (exact 0.707);
  * size within 4 % of the analytic ∫v dt.
* **Competitive growth.** Directional growth from 120 random grains at G = 10⁶ K/m and
  V = 0.05 m/s: over 190 µm the mean angle between <100> and the growth direction falls from
  32° to 10°, and 12 grains survive. This is Walton–Chalmers selection.
* **Kinetics consistent with the phase field.** KGT for the Ni–Nb quasi-binary of §5.3 gives
  31 K tip undercooling at 0.1 m/s and 41 K at 0.3 m/s. The phase-field tips sit 28–40 K below
  the liquidus at these speeds.

**Against NIST's EBSD section (case B)** (`benchmarks/literature/nist_ambench2018_tracks_ca.py`).
* **Set-up.** The CA regrows the calibrated case-B pool in 3-D: 0.5 µm cells, 50 µm along
  the scan, the EBSD plane 40 µm from the upstream end.
* **Substrate.** The pixels, placement and fusion zone are those of §5.1. Two substrates:
  * *extruded*: the measured section extended unchanged along the scan, the best case for one
    section;
  * *hybrid*: the measured section within ±1.4 µm of the plane, and a statistically equivalent
    3-D Voronoi substrate elsewhere (orientations drawn from the measured substrate), as in
    §5.2.

| CA run | pixels within 10° of measured | grains in the section (measured 92) | area-weighted size (measured 10.1 µm) |
|---|---:|---:|---:|
| extruded substrate | 25.4 % | 47 | 15.3 µm |
| extruded, reversed scan direction | 21.7 % | 49 | 16.6 µm |
| extruded, 0.25 µm cells | 26.8 % | 47 | 16.3 µm |
| extruded, faster tip kinetics (5 K at 0.1 m/s) | 24.0 % | 38 | 14.5 µm |
| extruded, nucleation 10¹⁴ /m³ at 10 ± 2 K (2 % of the pool nucleated) | 24.5 % | 55 | 14.2 µm |
| extruded, nucleation 10¹⁵ /m³ at 30 ± 2 K (20 % nucleated) | 24.1 % | 60 | 14.1 µm |
| hybrid, 2.7 µm stand-in grains (5 runs) | 12–17 % | 144–168 | 8.5–10.9 µm |
| **hybrid, 7.9 µm stand-in grains (3 runs)** | 12–17 % | **90–99** | **11.2–14.0 µm** |
| extruded, heat-flow bias α = 0.5 / 1 (cellular growth) | 24.2 % / 23.8 % | 50 / 51 | 14.7 / 13.8 µm |
| hybrid 7.9 µm, heat-flow bias α = 1 (3 runs) | 12–14 % | 91–100 | 11.1–12.7 µm |
| *rules, for comparison:* gradient trace (§5.1) / with the hybrid substrate (§5.2) | 27.5 % / 26.7 % | 83 / 236 | 12.3 µm / — |

![3-D grain CA against the EBSD section](img/ca_ambench_caseB.png)

**Reading the results.**
* **Grain statistics are reproduced.** With a stand-in substrate of the measured
  area-weighted grain size, the CA predicts 90–99 grains against 92 measured. Their size is
  11–14 µm against 10.1 µm.
* **The extruded substrate gives too few grains: 47.** Two effects can explain this:
  * grains entering the section from substrate outside the plane (§5.2);
  * grain selection in the CA that is too strong (next point).
* **Point-wise agreement equals that of the rules.** The extruded substrate gives 25.4 %
  (21.7 % for the other scan direction), against 25.8–27.5 % for the 2-D rules.
* **It does not depend on the growth model.**
  * Cell size: halving it changes nothing (26.8 %, 47 grains).
  * Kinetics: faster tip kinetics give 24.0 % and 38 grains.
  * Growth mode: the heat-flow bias that turns each grain's growth axis towards the heat flow
    (cellular growth) gives 24.2 % at α = 0.5 and 23.8 % at α = 1, with 50–51 grains.
  * So over one 36 µm-deep track the strength of grain selection does not matter: the limit
    is the unknown 3-D substrate.
* **The hybrid substrate scores lower (12–17 %) for a reason of construction.** Its measured
  grains form a slab only 2.8 µm thick along the scan, which neighbouring stand-in grains
  overgrow in 3-D. The gradient trace (26.7 % with the same substrate) has no lateral growth,
  so it is not penalised.
* **Nucleation plays no role here.** At 10¹⁴ /m³ (10 K) it changes nothing. At 10¹⁵ /m³
  (30 K) a fifth of the pool, near the top, becomes new grains. The measured section shows no
  equiaxed band, so heterogeneous nucleation is weak in this track.
* **Use.** The CA is suited to grain size and texture statistics, and to builds of many tracks
  and layers (§5.5), where growth distances are long. There, competition and the growth mode
  (α) decide the texture. For point-wise maps of a single track, the gradient trace is as good.

### 5.5 Grains of multi-layer blocks: texture and grain width (3-D grain CA)

**Question.** How do grain width and texture develop over many layers, and how much do they
depend on the scan rotation?

**Method** (`benchmarks/literature/ca_multilayer.py`).
* **Build.** An IN625 block of 10 layers at the AM-Bench 2018 settings: the calibrated case-B
  track (179.2 W, 800 mm/s), 100 µm hatch, 20 µm layers, bidirectional.
* **Rotation.** 90° (x/y alternation) or 67° between layers.
* **Grid.** 480 × 480 µm, on 40 µm of substrate with random 8 µm grains. KGT kinetics, no
  nucleation.
* **Thermal history.** The steady track is shifted along every vector; tracks overrun the
  block, so pool ends do not enter. Preheating is not included.
* **Checks.** Cell size 1, 1.5 and 2 µm; a 240 µm block; three substrate realisations.

| run (layer 10) | <100> within 15° of BD | <110> within 15° of BD | grain width | grains |
|---|---:|---:|---:|---:|
| **90°, 1 µm cells** | **35 %** | 16 % | 56 µm | 237 |
| **67°, 1 µm cells** | **68 %** | 7 % | 72 µm | 164 |
| 90°, 1.5 µm cells | 17 % | 22 % | 56 µm | 265 |
| 90°, 2 µm cells, substrates 0 / 1 / 2 | 15 / 14 / 24 % | 27 / 32 / 32 % | 57 / 60 / 60 µm | 230 / 236 / 220 |
| 67°, 2 µm cells | 26 % | 24 % | 55 µm | 193 |
| 90° / 67°, 1 µm cells, 240 µm block | 19 / 21 % | 19 / 8 % | 55 / 48 µm | 79 / 56 |
| random orientations | 10 % | 21 % | | |

The substrate grains are 9 µm wide.

![IN625 blocks](img/ca_multilayer.png)

**Reading the results.**
* **A <100> fibre along BD builds up layer by layer, and 67° makes it stronger than 90°.**
  * At 1 µm cells it grows from 14 % in the first layer to 35 % (90°, levelling off) and 68 %
    (67°, still rising) in the tenth.
  * 67° beats 90° in every comparable pair of runs (68 vs 35 %, 26 vs 14–24 %, 21 vs 19 %).
* **The fibre needs fine cells.** At 2 µm cells it stalls at 14–26 %, and at 1.5 µm at 17 %.
  The three substrate realisations at 2 µm scatter by about ±5 points, far less than the jump
  to 35 % and 68 % at 1 µm.
  * Coarse cells blunt the orientation selection at every layer; already in the first layer
    the fibre is 14 % against 11 %.
  * Whether 1 µm is converged is not known: a 0.5 µm check of a 480 µm block (900 M cells)
    needs the GPU.
* **The fibre also needs a large block.** The 240 µm blocks at 1 µm give only 19–21 %. With
  56–79 grains left at layer 10, a large share of them lie against the side walls, where they
  cannot be overgrown from outside; this is the likely cause. Periodic side boundaries would
  remove it.
* **Grain width is robust where the fibre is weak.** The grains widen from 15 µm in the first
  layer to 55–60 µm in the tenth for every cell size, block and substrate at 90°, within 10 %.
  At 67° with 1 µm cells the strongly selected grains grow wider, to 72 µm. Width and texture
  develop together.
* **The track pattern is visible.**
  * At 90°, the grains in plan view form a checkerboard on the 100 µm hatch grid, with
    boundaries along the track overlaps.
  * At 67°, the plan view is dominated by large <100> grains (red), with tilted, blocky
    boundaries.
* **Against experiment.** LPBF Ni superalloys commonly show a <100> fibre along BD, which
  agrees qualitatively. How its strength depends on the scan rotation differs between published
  builds, and no measured texture of these builds was available to test the numbers.

### 5.6 Cells in 3-D with several elements, and what EDS would record

**Question.** What do the cells of the case-B melt pool look like in 3-D when all the main
alloying elements segregate, and how much of that segregation can an EDS map show?

**Model** (`process/phasefield3d.py`, `post/eds.py`; manual §15).
* **Phase field.** The 2-D model of §5.3 in 3-D, with one supersaturation per solute.
  * The interface temperature sees their sum, each weighted by the solute's share of the
    freezing range.
  * Every solute has its own anti-trapping current and solute trapping (continuous growth
    model, V_D = 1 m/s).
  * Anisotropy is cubic in any crystal orientation. Here <100> lies along the build direction.
  * With one solute and a box one cell thick, the model reproduces the 2-D model to 10⁻¹⁵
    (tested). Splitting a solute into two identical halves changes nothing. The numba and
    NumPy/CuPy kernels agree, and every solute is conserved to round-off.
* **Alloy.** IN625 of the AM-Bench 2018 feedstock (ICP analysis, Keller et al. 2017, Table 1).

| solute | content [wt%] | k (equilibrium) | k at 0.092 m/s | liquidus slope [K/wt%] | share of the freezing range |
|---|---:|---:|---:|---:|---:|
| Nb | 3.75 | 0.48 | 0.52 | −10.5 (Ni–Nb, as in §5.3) | 90 % |
| Mo | 8.83 | 0.85 | 0.86 | −2.9 (Ni–Mo liquidus secant, estimate) | 10 % |
| Cr | 20.7 | 1.02 | 1.02 | partitions only | 0 |
| Fe | 0.72 | 1.09 | 1.08 | partitions only | 0 |

  * **Sources.** Partition coefficients from Gan et al. (*Integr. Mater. Manuf. Innov.* 8
    (2019) 178, Table 4, after Hu et al. 2017).
  * **Freezing range.** 47 K at equilibrium, 40 K with trapping.
  * **Ni.** Ni is the balance of the four solutes. The ~1 wt% of Ti, Al, Co and Si in the
    powder is not modelled.
  * **Cr and Fe.** Their k is close to 1, so their effect on the interface temperature is
    below 2 % of the freezing range; they are carried without it.
  * **Diffusivity.** One liquid diffusivity, 3 × 10⁻⁹ m²/s.
* **EDS.** Maps of a section through the solidified block.
  * **SEM, X-ray source volume.** Its depth range is the Anderson–Hasler range of each X-ray
    line. It widens with depth as depth^1.5 and holds 90 % of the signal within that range.
  * **SEM, lines.** Nb Lα and Mo Lα at 15 kV are generated from about 0.7 µm, twice the cell
    spacing. At 5 kV the volume is below 0.1 µm, and L lines are used for Cr, Fe and Ni.
  * **STEM.** A 100 nm foil at 200 kV: the average through the foil plus about 5 nm of beam
    broadening.
  * **Counting noise.** Poisson noise with a continuum background. Concentration to counts
    is linear, without ZAF corrections.

**Run** (`benchmarks/literature/pf3d_in625_eds.py`).
* **Conditions.** The mid-pool point of §5.3, 22.5 µm below the surface: G = 6.6 × 10⁶ K/m,
  R = 0.092 m/s.
* **Box.** A periodic 1.6 × 1.6 µm box with a 1.2 µm window, grown 5 µm.
* **Resolution.** W0 = 10 nm (dx = 8 nm), 200 × 200 × 150 cells. At this point the 2-D
  spacing at W0 = 10 nm is within 8 % of the converged value (§5.3).
* **Cost.** 60,000 steps, 2.2 h on 4 CPU cores (numba).
* **2-D reference.** The same multicomponent model, one cell thick and 2.4 µm wide.

**Cells and segregation.** Compositions are taken over the material grown after the first third
of the run.

| | 2-D, same model | 3-D |
|---|---:|---:|
| cell spacing, material grown last (0.6–1 µm below the tips) | 0.34 µm | 0.32–0.33 µm |
| cell spacing, material grown 4 / 5 µm earlier | | 0.27 / 0.24 µm |
| Nb: 5 % / median / 95 % [wt%] | 3.15 / 3.37 / 5.95 | 2.95 / 3.44 / 5.84 |
| Mo: 5 % / median / 95 % [wt%] | 8.54 / 8.66 / 9.79 | 8.42 / 8.67 / 9.77 |
| Cr: 5 % / median / 95 % [wt%] | 20.44 / 20.75 / 20.78 | 20.45 / 20.75 / 20.82 |
| Fe: 5 % / median / 95 % [wt%] | 0.68 / 0.73 / 0.73 | 0.68 / 0.73 / 0.74 |
| Ni (balance): 5 % / median / 95 % [wt%] | 63.1 / 66.5 / 66.8 | 63.3 / 66.4 / 67.1 |

* **3-D spacing.** It is measured as sqrt(A / N) on transverse sections, with cells counted by
  watershed. For hexagonal cells the centre-to-centre distance is 7 % larger.
* **Measured range.** Case-B cells measured by SEM: 0.2–1.0 µm (Ghosh et al. 2018).

**What EDS records.** Transverse section 0.8 µm below the tips. The share of the true Nb
contrast (5–95 % spread) kept in a noise-free map:

| technique | Nb, Mo line | 90 % of the signal within | Nb contrast kept |
|---|---|---:|---:|
| SEM, 15 kV | Lα | 0.68 µm (Ni Kα: 0.45 µm) | 27 % |
| SEM, 5 kV | Lα (also Cr, Fe, Ni Lα) | 0.08 µm (Ni Lα: 0.11 µm) | 85 % |
| STEM, 100 nm foil, 200 kV | Kα | 0.01 µm | 96 % |

![3-D cells](img/pf3d_in625_eds_3d.png)

![EDS maps](img/pf3d_in625_eds.png)

**Reading the results.**
* **In 3-D the cells form a network of hexagon-like columns.**
  * The front is a field of rounded cell caps over narrow, deep liquid grooves.
  * Nb and Mo pile up along the cell walls and most of all along the cell edges, where three
    cells meet. These triple junctions form Nb-rich columns, up to 7–8 wt% Nb in the solid section.
  * Cr, Fe and Ni are slightly depleted there. Cr and Fe have k > 1, and Ni is the balance.
  * Isolated droplets of the last liquid reach 13–21 wt% Nb, the sites where Laves phase or
    carbides form in the real alloy (not modelled).
* **The 3-D spacing was still coarsening when the run ended.**
  * The front first breaks up into fine cells, then cells are eliminated: material grown 5 µm
    before the end has 0.24 µm, material grown last 0.33 µm.
  * The tips were still drifting ahead of the T0 isotherm, from 1.2 µm (mid-run) to 1.4 µm.
  * So 0.33 µm is a lower bound for the steady 3-D spacing at this point. It is close to the
    2-D value and inside the measured 0.2–1.0 µm.
  * A steady 3-D spacing needs a longer run, and a converged one W0 below 10 nm. Both are GPU
    work.
* **Segregation agrees between 2-D and 3-D.**
  * The 5–95 % ranges of Nb, Mo, Cr and Fe agree within 0.2 wt%. The 3-D cores are about
    0.2 wt% leaner in Nb.
  * Where the 2-D model is cheaper, it gives the right segregation statistics. 3-D is needed
    for the shape of the network: edges against walls, droplets, sections at any angle.
* **Only Nb and Mo segregation is measurable by EDS; Cr and Fe are not.**
  * In the section, Cr varies by 2 % of its content and Fe by 9 % of its 0.7 wt%. Both stay
    below the counting noise of a map in every technique: Cr because the variation is small,
    Fe because its signal is only about 11 counts per pixel here. Their segregation can only be
    quantified from line scans or point analyses with long counting times.
  * Ni depletion at the walls (about 5 %) is visible in STEM.
* **SEM-EDS at 15 kV cannot resolve these cells.**
  * At 15 kV, 90 % of the Nb and Mo signal comes from a 0.7 µm wide volume, two cells across.
    The map keeps about a quarter of the Nb contrast and shows the network only faintly.
  * At 5 kV (L lines for all elements) the volume shrinks below 0.1 µm, and 85 % of the
    contrast survives.
  * STEM on a thin foil keeps nearly all of it. The few percent lost come from walls inclined
    within the 100 nm foil.
  * Consequence: SEM-EDS at the usual 15–20 kV underestimates intercellular segregation in LPBF
    alloys several-fold. Quantitative comparison needs STEM-EDS, low-kV SEM, or this blurring
    applied to the simulation first (`tpmsfem.post.eds`).
* **Uncertain inputs.**
  * The SEM numbers depend on the shape assumed for the X-ray source volume. A single
    Gaussian with the same 90 % diameter would keep under 1 % of the contrast at 15 kV, against
    27 % here. Calibrate on a measured line scan when one is available.
  * The Mo liquidus slope is an estimate, and Mo's share of the freezing range is 10 %.
  * One liquid diffusivity is used for all solutes.

## 6. Roughness vs build angle (Kasperovich et al. 2021, LPBF IN718)

Kasperovich, Becker, Artzt et al., *Materials & Design* 207 (2021) 109858,
doi:10.1016/j.matdes.2021.109858 (gold open access).
* **Build.** Concept Laser M2, 30 µm layers, powder D10/D50/D90 = 22/33/45 µm.
* **Measurement.** Sa by laser scanning microscopy on 2.44 × 2.44 mm² of flat outer surfaces
  at 45° (upskin), 90° (vertical) and 135° (downskin).
* **Model inputs.** The edge radius is half the width of the contour track (180 W,
  1600 mm/s) from the calibrated melt-pool model: 85 µm wide, so 43 µm.
* **Fit.** The three particle parameters are fitted to the three measured Sa values. This is
  a calibrated description, not a prediction.

| contour | measured Sa 45° / 90° / 135° | fitted model | particles on a vertical wall | downskin / upskin factors |
|---|---|---|---:|---:|
| 180 W, 1600 mm/s (E_L 0.113 J/mm) | 9.01 ± 0.33 / 11.17 ± 1.44 / 52.80 ± 3.41 µm | 9.02 / 11.16 / 52.80 µm | 268 /mm² | k_down 4.3, k_up 0.33 |
| optimum, E_L 0.18 J/mm | 6.8 / 7.2 / 47.5 µm | 6.79 / 7.22 / 47.21 µm | 179 /mm² | 6.2, 0.18 |

![roughness](img/roughness_vs_angle.png)

**Reading the results.**
* **Parts of the roughness.** The layer staircase and melt-pool rounding alone give only
  0–5 µm. Powder stuck to the surface dominates, especially on down-facing surfaces, which
  rest on the powder bed.
* **TPMS application.** On a 3 mm, 30 %-dense gyroid sheet cell, 30 % of the area faces
  down by more than 120°. The area-weighted Sa is therefore 26 µm, against 11 µm for a
  vertical wall.
* **Why it matters.** For lattice implants this roughness is
  * a fatigue concern (notches),
  * a fluid-flow and osseointegration factor,
  * and a reason the as-built stiffness of thin sheets falls below the CAD value.

## 7. Solidification of a CoCrFeMnNi (Cantor alloy) gyroid lattice

**Question.** How do the solidification conditions, cells, Mn/Ni segregation and grain texture
of an LPBF gyroid-sheet lattice in the equiatomic CoCrFeMnNi high-entropy alloy change between
thin and thick walls, and on down-facing surfaces built over powder? No study of
CoCrFeMnNi TPMS solidification turned up in the literature. Script:
`benchmarks/literature/cantor_gyroid_solidification.py`.

**New model pieces.**
* **Part-aware substrate.** `MeltPoolModel.set_part_geometry`: the substrate is the part built
  so far, an implicit field, with loose powder elsewhere. Down-facing regions of a new layer
  therefore sit on powder.
* **Preheat.** `MeltPoolModel.set_temperature` sets a preheat field.
* **Scan paths.** `process/scanpath.py` builds contour and hatch vectors clipped to one layer
  of an implicit part, with laser-off jumps between them.
* **Wall thickness.** `geometry.tpms.sheet_with_thickness`: a TPMS sheet with a prescribed or
  graded wall thickness, from `f/|grad f|`. On a 3-D grid the thin-wall volume fraction is
  within 3 % of `t · S/V`.

**Alloy data** (`benchmarks/literature/cantor_material.py`).
* **Thermal.** k(T), cp(T), density 7980 kg/m³, latent heat 2.2e5 J/kg, solidus 1250 °C and
  liquidus 1340 °C, from Li et al., J. Mater. Sci. Technol. 77 (2021) 187. These are the
  inputs of their laser-deposition simulation; the paper does not give their measurement
  source.
* **Segregation.** The cell cores keep Co, Cr and Fe, and the walls collect Mn and Ni. The
  phase field uses the pseudo-binary CrFeCo–(MnNi): solute Mn + Ni at 40 wt%, k = 0.74.
  A laser-remelting study reports k_Mn = 0.81 and k_Ni = 0.92.
* **Freezing range.** The least certain input. It is run at 40, 60 (baseline) and 90 K.
* **Evaporation.** Mn evaporation at its 20 at% (Raoult's law).

**Validation: Piglione et al., Mater. Lett. 224 (2018) 22** (open access).
* **Their build.** A single LPBF layer of 11 tracks on a Renishaw AM250: 200 W, 80 µs
  exposure, 60 µm point distance, 125 µm hatch.
* **Their measurement.** Cell spacing 0.52–0.64 µm, mean 0.59 µm. The cells grow with a <001>
  axis normal to the fusion line.
* **Model setup.**
  * The same power and a 70 µm spot, scanned continuously at 0.75 m/s.
  * Absorptivity 0.35 and absorption depth 15 µm, not calibrated for this alloy.
  * A 30 µm powder layer on a Cantor plate.
* **Melt pool.** 97 µm wide; it melts through the powder and 24 µm into the plate. The
  fusion-boundary medians are G = 3.0e7 K/m and R = 0.027 m/s.

| freezing range | phase-field spacing (6 µm window) | Hunt | Kurz–Fisher | measured |
|---|---|---|---|---|
| 40 K | 0.50 µm | 0.54 µm | 0.95 µm | 0.52–0.64 µm (mean 0.59) |
| 60 K | 0.50 µm | 0.60 µm | 1.05 µm | |
| 90 K | 0.46 µm | 0.66 µm | 1.17 µm | |

The phase field is 15–22 % below the measured mean. Candidate causes:
* The AM250 exposes point by point with pauses, so its mean speed is below the 0.75 m/s
  assumed here, and a slower front grows coarser cells.
* The absorptivity is a guess.
* 10 µm of growth may stop short of full coarsening.

**Gyroid.**
* **Geometry.** A gyroid sheet with a 4 mm cell, its wall thickness graded linearly from
  0.2 mm at x = 0 to 0.6 mm at x = 4 mm.
* **Process.** The validation settings, 30 µm layers, 85 µm hatch, one contour, no rotation.
* **Windows.** Two 1.2 mm windows of the layer at 1.3 mm build height are simulated with the
  real solid and powder substrate and the clipped scan path: thin walls with a median of
  0.25 mm, and thick walls with a median of 0.52 mm.
* **Classes.** A fusion-boundary column is *down-facing* when the layer below it was powder.
  That covers 7.5 % of the melted columns in the thin window and 3.4 % in the thick one.

| walls | material below | G (median) | R (median) | cooling rate | cell spacing | Mn+Ni: cores / tip liquid / max between cells |
|---|---|---|---|---|---|---|
| thin (0.25 mm) | solid | 2.2e7 K/m | 0.025 m/s | 5.1e5 K/s | 0.50 µm | 0.91 / 1.16 / 1.37 × nominal |
| thin | powder | 2.5e7 K/m | 0.010 m/s | 2.3e5 K/s | **0.86 µm** | 0.91 / 1.19 / 1.38 |
| thick (0.52 mm) | solid | 1.85e7 K/m | 0.025 m/s | 4.4e5 K/s | 0.50 µm | 0.91 / 1.16 / 1.32 |
| thick | powder | 2.4e7 K/m | 0.010 m/s | 2.2e5 K/s | **0.86 µm** | 0.91 / 1.19 / 1.38 |

![Cantor gyroid](img/cantor_gyroid.png)

**Reading the results.**
* **Down-facing surfaces solidify differently. Wall thickness barely matters within a layer.**
  * Over powder, the solidification front at the fusion boundary moves 2.5× slower and cools
    half as fast.
  * The cells there are 1.7× coarser: 0.86 against 0.50 µm, in both thin and thick walls.
  * A homogenisation anneal needs a time that scales with the cell spacing squared, so the
    down-facing skin needs about 3× longer to homogenise than the rest of the lattice.
* **Thick walls are not hotter in a single layer.** Their gradient is 15 % *lower*, because
  more hatch tracks preheat each other, but that does not change the cell spacing.
* **Segregation is mild, as expected for k close to 1.** Cell cores hold about 0.91× the
  nominal Mn + Ni, and the last liquid about 1.4×, roughly 55 wt% Mn + Ni against 40.
  * This is much weaker than Nb in IN625 (k = 0.48).
  * It is consistent with the absence of second phases in as-built LPBF CoCrFeMnNi.
* **Texture.** After one layer grown from random substrate grains, 12.6–13.6 % of the
  fusion zone has a <100> within 15° of the build direction (random: 10.2 %). The strong
  <001> build texture reported for LPBF CoCrFeMnNi needs many layers of competitive growth,
  which this study does not simulate.

**Limits.**
* **One layer only.** Heat accumulation over many layers is not included, and in thin
  lattice walls it should raise the temperature and lower G as the build grows. A
  part-scale thermal model (`thermal/heat.py`) on the lattice could supply that preheat
  through `set_temperature`.
* **No keyhole physics.** With a 70 µm spot at 200 W the absorbed intensity is about
  36 kW/mm², and the conduction model caps the surface only at about 5400 °C. It has no
  keyhole, recoil pressure or Marangoni flow, so peak temperatures are not physical. The
  fusion-boundary G and R follow from the overall energy balance.
* **Uncalibrated inputs.** Absorptivity, layer thickness and the property data are not
  calibrated for CoCrFeMnNi.
* **Simplified solidification model.** The phase field is 2-D, and the alloy is treated as
  a pseudo-binary with an uncertain freezing range.
* **Mn loss is not tracked.** Mn evaporation cools the surface, but the loss of Mn from the
  alloy is not computed.

### 7.1 Heat accumulation over many layers

**Question.** §7 simulated one layer on an 80 °C substrate. Does heat building up over many
layers change the picture, and does it separate thin from thick walls? Script:
`benchmarks/literature/cantor_gyroid_multilayer.py`.

**Part-scale build** (`thermal/heat.py`).
* **Geometry.** One 4 × 4 mm gyroid cell column with the graded walls, on a 2 mm plate whose
  bottom is held at 80 °C. The surrounding powder is part of the model (2 % of the solid
  conductivity). The side faces are adiabatic, as for a cell inside a large lattice.
* **Layers.** 30 µm layers, 10 s per layer, lumped into 0.133 mm lumps on 1/15 mm voxels.
* **Energy input.** Each lump receives the energy of its layers where the laser puts it. The
  contour and hatch path of the slice deposits `f_net · A · P / v` per unit length.
  `f_net = 0.70` is the share of the absorbed energy that stays in the part in a single-track
  melt-pool run; the rest evaporates or radiates from the pool. That is about 3.1–3.9 J per
  layer for the cell.
* **Timing.** The lump's energy arrives in four pulses 1.1 layer times apart, so the
  temperature just before a pulse is what a new layer meets. Depositing a lump at once and
  waiting 4.4 layer times understates it.
* **Extent.** The build was simulated to 8.8 mm (66 of 75 lumps). The run was interrupted
  twice by infrastructure restarts; the two runs agree to 0.1 °C where they overlap.

**Result.**
* **Rise with height.** The temperature a new layer meets rises about 7 °C per millimetre of
  build height once the column is a few millimetres tall: 97 °C at 4.1 mm and 132 °C at
  8.8 mm. It wobbles by about ±5 °C with the slice.
* **No thin/thick difference.** The thin-wall and thick-wall regions never differ by more
  than 1 °C. Heat spreads sideways through the lattice much faster than a layer takes.

**Effect on solidification.** The two melt-pool windows of §7 were re-run with the part
starting hotter, followed by the phase field:
* at 140 °C, the preheat at 9.3 mm (extrapolated from 8.8 mm; this is the slice of §7 two
  cells higher);
* at 500 °C, an upper-bound case. On the 7 °C/mm slope that is a part about 60 mm tall at a
  10 s layer time, or a shorter part with a shorter layer time. The slope is extrapolated,
  not simulated, beyond 8.8 mm.

| start temperature | walls | below | G [K/m] | R [m/s] | cooling rate [K/s] | cell spacing |
|---|---|---|---|---|---|---|
| 80 °C (single layer) | thin / thick | solid | 2.2e7 / 1.85e7 | 0.025 | 5.1e5 / 4.4e5 | 0.50 / 0.50 µm |
| | thin / thick | powder | 2.5e7 / 2.4e7 | 0.010 | 2.3e5 / 2.2e5 | 0.86 / 0.86 µm |
| 140 °C (9.3 mm up) | thin / thick | solid | 2.0e7 / 1.7e7 | 0.025 / 0.024 | 4.6e5 / 3.9e5 | 0.50 / 0.55 µm |
| | thin / thick | powder | 2.5e7 / 2.2e7 | 0.010 / 0.011 | 2.2e5 / 2.1e5 | 0.86 / 0.86 µm |
| 500 °C (upper bound) | thin / thick | solid | 1.15e7 / 0.93e7 | 0.017 | 1.8e5 / 1.5e5 | 0.75 / 0.75 µm |
| | thin / thick | powder | 1.6e7 / 1.7e7 | 0.009 | 1.2e5 / 1.3e5 | 1.00 / 0.90 µm |

![heat accumulation](img/cantor_gyroid_multilayer.png)

**Reading the results.**
* **Ten millimetres of build height does not change the cells.** The 60 °C of accumulated
  heat lowers the cooling rate by about 10 %. The cell spacing stays within one counting step
  of the single-layer values, and the segregation is unchanged.
* **A much hotter part does.** At 500 °C the gradient halves and the cooling rate falls to a
  third. The cells over solid coarsen by 50 % (0.50 → 0.75 µm), and the down-facing cells by
  5–15 %. The down-facing skin stays the coarsest region, but the contrast shrinks from 1.7×
  to about 1.3×.
* **Wall thickness still does not matter**, at any of the three temperatures. The conclusion
  of §7 holds with heat accumulation included: in this lattice the microstructure contrast is
  between down-facing skin and the rest, not between thin and thick walls.
* **Segregation is insensitive.** Cell cores stay at 0.90–0.91× the nominal Mn + Ni, and the
  last liquid at 1.3–1.4×.

**Limits.**
* The build covers 8.8 mm; taller parts are a linear extrapolation.
* The adiabatic sides represent a cell deep inside a wide lattice. A slender part loses heat
  sideways and stays cooler; a part on a poorly conducting support stays hotter.
* The energy pulses are spread over a third of a layer time, which slightly overstates the
  temperature a new layer meets.
* A single 10 s layer time was run. The accumulated temperature rise scales roughly inversely
  with it.
* The limits of §7 (no keyhole physics, uncalibrated absorptivity, 2-D pseudo-binary phase
  field) apply unchanged.

### 7.2 First, middle and final layers: segregation and residual strain

**Segregation at three heights** (`benchmarks/literature/cantor_gyroid_layers.py`). The
phase field was run at the fusion-boundary conditions of three layers of the column:
* the first layer, on the plate at 80 °C (the single-track conditions on a solid plate);
* the middle layer at 4.4 mm, which starts from 102 °C;
* the final layer at 8.8 mm, which starts from 132 °C.

For the middle and final layers, G and R are interpolated in the start temperature between
the melt-pool runs at 80 °C and 140 °C (§7.1), for wall regions over solid and over powder.

| layer | starts from | over solid: spacing, cores, max between cells | over powder |
|---|---|---|---|
| first | 80 °C | 0.50 µm, 0.92×, 1.39× nominal Mn + Ni | does not apply (solid plate) |
| middle, 4.4 mm | 102 °C | 0.50 µm, 0.91×, 1.37× | 0.86 µm, 0.91×, 1.38× |
| final, 8.8 mm | 132 °C | 0.50 µm, 0.91×, 1.32× | 0.86 µm, 0.91×, 1.38× |

![segregation in three layers](img/cantor_gyroid_layers.png)

The three layers have the same cell spacing and the same segregation. Cell cores hold about
36 wt% Mn + Ni against the nominal 40, and the boundaries between cells peak at 53–54 wt%.
The contrast is within each layer, between regions over solid and over powder, not between
layers.

**Residual strain** (`benchmarks/literature/cantor_gyroid_residual.py`).
* **Model.** The graded column, 8 mm tall on a 1 mm plate, is built with the layer-lumped
  thermo-mechanical model (`residual/thermomech.py`): 0.1 mm voxels, 0.4 mm lumps born at the
  liquidus, 10 s per 30 µm layer, plate at 80 °C. The part then cools to 20 °C and is
  released from the plate.
* **Material.** E(T) and the expansion are from Laplanche et al., J. Alloys Compd. 623 (2015)
  348. The yield strength is that of as-built SLM CoCrFeMnNi in compression from Kim, Yang &
  Lee, Sci. Rep. 10 (2020) 8045: 757 MPa at 25 °C falling to 223 MPa at 900 °C.
* **Assumptions.** Linear hardening of 1.5 GPa at room temperature, scaled with the yield
  strength; the yield strength above 900 °C, continued to 40 MPa at the solidus.
* **Build order.** A gyroid slice can contain a piece that only joins the part a few layers
  higher. The powder holds it in the machine; in the model it would float. Each lump
  therefore activates only cells connected to what is built (90 of 37,196 cells wait, by at
  most two lumps).

| quantity (lattice, room temperature) | value |
|---|---|
| elastic strain ε_zz on the plate, 5th–95th percentile | −2590 … +2690 microstrain |
| elastic strain ε_zz, plate removed | −2420 … +2620 microstrain |
| elastic strain ε_xx, plate removed | −2040 … +1900 microstrain |
| von Mises stress, plate removed, median / 95th percentile | 584 / 748 MPa |
| equivalent plastic strain, median / 95th percentile | 0.9 / 2.5 % |

![residual strain](img/cantor_gyroid_residual.png)

**Reading the results.**
* **The walls are in bending.** Across each wall the build-direction strain changes sign:
  tension on one face, compression on the other, at about ±2500 microstrain. This follows
  the wall's inclination: each new lump shrinks on an inclined wall and bends it.
* **Removing the plate changes little.** The strains in the lattice are self-equilibrated
  wall by wall; only the first half millimetre above the plate relaxes.
* **The stresses are close to yield.** The median von Mises stress is 584 MPa against a
  room-temperature yield strength of 757 MPa, and the lattice has yielded throughout during
  the build (median plastic strain 0.9 %, 95th percentile 2.5 %).
* **Thick walls carry somewhat more strain.** The mean |ε_zz| is 1000–1750 microstrain in the
  thick-wall half and 750–1500 in the thin-wall half.
* **First, middle and final layers.** The first layers are held by the plate, with tension
  and compression side by side. The middle layers show the wall-bending pattern. The final
  layers are in mild compression in the build direction, since nothing was deposited on top
  of them.

**Limits.**
* The thinnest walls are two voxels wide, so their through-thickness strain gradient is
  poorly resolved. The numbers are indicative, not converged.
* The lumps are 13 layers thick. Lumping is known to change the stress level; in the
  AM-Bench bridge the same model landed within 3–9 % of the measured deflection, with
  measured flow curves.
* The hardening and the high-temperature yield strength are assumed, and the yield data are
  from another machine and parameter set. No residual-strain measurement on a CoCrFeMnNi
  lattice was found to compare with.
* Powder is not modelled, thermally or mechanically.

### 7.3 Checks before publication

**Question.** Which results of §7–§7.2 survive the checks a reviewer would ask for? These
are the checks that need no experiment:
* the validation offset, where the phase field gave 0.50 µm against a measured 0.59 µm;
* the numerics of the phase field;
* the uncertain inputs: absorptivity and the pseudo-binary parameters;
* the resolution of the residual-strain build.

Script: `benchmarks/literature/cantor_publication_checks.py` (about 9 h on 4 cores). Unless
stated, the phase field runs as in §7: a 6 µm window, W0 = 10 nm, 10 µm of growth, ΔT0 = 60 K.
A 6 µm window counts spacing in steps of about 8 % (6/12 = 0.50, 6/11 = 0.55, 6/10 = 0.60 µm).

![publication checks](img/cantor_publication_checks.png)

**1. The validation offset.** The validation track was re-run in three ways.

*Point-by-point exposure, as on the Renishaw AM250.* The beam dwells 80 µs on each point,
then moves 60 µm with the laser off. The delay is not reported, so 1, 20 and 50 µs were
tried. Result:
* the pool is the same, 101 × 24 µm;
* the fusion-boundary speed is 0.025–0.030 m/s, against 0.027 m/s for the continuous scan;
* the cells are 0.46–0.50 µm.

Each point melts and freezes almost on its own, so the pauses do not slow the solidification
front; longer pauses make it slightly faster. Point exposure does not explain the offset.

*A continuous scan at the slower mean speeds.* 0.60 m/s gives 0.46 µm; 0.46 m/s gives 0.55 µm.

*Absorptivity and the pseudo-binary inputs.*

| change from the baseline (A = 0.35, D = 3e-9 m²/s, k = 0.74, ε4 = 0.02) | pool | R [m/s] | spacing |
|---|---|---|---|
| none | 97 × 24 µm | 0.027 | 0.50 µm |
| A = 0.30 | 96 × 20 µm | 0.030 | 0.46 µm |
| A = 0.45 | 106 × 29 µm | 0.025 | 0.50 µm |
| A = 0.60 | 114 × 36 µm | 0.021 | **0.60 µm** |
| D doubled | | | **0.67 µm** |
| k = 0.81 (the reported k_Mn) | | | 0.50 µm |
| k = 0.88 (walls counted at a 0.05 contrast threshold) | | | 0.50 µm |
| ε4 = 0.01 / 0.04 | | | 0.59 / 0.55 µm |

* **Absorptivity and diffusivity both move the result.** Either a higher absorptivity or a
  higher liquid diffusivity brings the spacing onto the measurement. A = 0.6 is plausible at
  this intensity (about 36 kW/mm², near keyhole). D for this melt is not measured.
* **What would separate them.** Measured single-track widths and depths at 200 W would fix A
  through the pool size. D would then remain the free parameter.
* **The anisotropy has no clear effect.** Both directions coarsen the cells by one or two
  counting steps. That is the scatter of the spacing selection (point 2), not a trend.

**2. What the phase-field spacing is.**

*Numerics.* The spacing over solid does not change with:
* the interface width (W0 = 7 nm: 0.50 µm);
* the random seed (0.50 µm);
* a taller window (3 µm: 0.50 µm).

A 12 µm window grown for 20 µm gives 0.55 µm: 39 cells merge to 22 and then stay. Over
powder, the seed gives 0.83 against 0.86 µm, and the 12 µm window gives 0.92 µm.

*Stability band.* The front was seeded with a set spacing instead of random noise:

| seeded spacing [µm] | 0.40 | 0.60 | 0.75 | 1.00 | 1.20 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|---|
| over solid (G 3.0e7 K/m, R 0.027 m/s) | 0.40 | 0.60 | 0.75 | 0.50 | | 0.55 | |
| over powder (G 2.5e7 K/m, R 0.010 m/s) | | 0.60 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 |

* **Many spacings are stable.** Over solid, spacings from at least 0.40 to 0.75 µm persist,
  and coarser fronts split down to 0.50–0.55 µm. As in experiments and other phase-field
  studies, the spacing within this band depends on how the front developed.
* **The measurement is inside the band.** The measured 0.52–0.64 µm lies inside the band
  over solid. The value reached from a planar front, 0.50–0.55 µm, sits near its lower end.
* **How to report the validation.** The measurement lies inside the predicted band, and the
  value reached from a planar front is 7–15 % below the measured mean.
* **Over powder the band extends higher.** The coarsest stable spacing is 1.0–1.2 µm over
  powder, against 0.75–1.0 µm over solid. Coarse fronts split to 0.75–1.0 µm over powder,
  against 0.50–0.55 µm over solid.

**3. Does the down-facing contrast survive?** The thin-wall window was re-run at two more
absorptivities.

| absorptivity | over solid: R, spacing | over powder: R, spacing | ratio |
|---|---|---|---|
| 0.30 | 0.028 m/s, 0.46 µm | 0.010 m/s, 0.86 µm | 1.9 |
| 0.35 (§7) | 0.025 m/s, 0.50 µm | 0.010 m/s, 0.86 µm | 1.7 |
| 0.50 | 0.020 m/s, 0.67 µm | 0.0085 m/s, 1.00 µm | 1.5 |

The 12 µm window gives a ratio of 1.7 (0.92 / 0.55 µm). Coarse seeded fronts give 1.4–2.0
(0.75–1.0 µm against 0.50–0.55 µm). Across these checks the down-facing skin grows cells
**1.4–2.0× coarser, typically 1.7×**. The front over
powder is 2.3–2.8× slower in every case. The headline holds, but as a range rather than 1.7×.

**4. Segregation.**
* **Cell cores** hold 0.90–0.93× the nominal Mn + Ni in every run at k = 0.74.
* **The last liquid** holds 1.32–1.46×.
* **Dependence on k.** The amplitude follows k directly. At k = 0.81 the cores hold 0.94× and
  the walls 1.26×; at k = 0.88, 0.97× and 1.15×.
* **For a paper.** Segregation numbers need a multicomponent check (Scheil, or EDS across the
  cells).

**5. Residual strain: voxel and lump size.** A 2 mm column on a 0.5 mm plate was built three
ways. The statistics are over the lattice between 0.5 and 1.5 mm, after plate removal.

| build | ε_zz 5th / 95th pct. [µstrain] | mean abs(ε_zz) | von Mises median | plastic strain median | run time |
|---|---|---|---|---|---|
| 0.1 mm voxels, 0.4 mm lumps (§7.2) | −1760 / +2400 | 973 | 511 MPa | 1.01 % | 12 min |
| 0.05 mm voxels, 0.4 mm lumps | −1800 / +2370 | 970 | 516 MPa | 1.08 % | 138 min |
| 0.1 mm voxels, 0.2 mm lumps | −2200 / +2920 | 1173 | 540 MPa | 1.27 % | 8 min |

* **The voxel size is converged.**
  * Halving the voxel changes the strain statistics by less than 3 %, the stress by 1 % and
    the plastic strain by 7 %.
  * Cell by cell, the build-direction strain correlates at 0.95 with the 0.1 mm build (the
    fine cells averaged onto the coarse ones).
  * The two-voxel walls were not a problem.
* **The lump size is not converged.** Halving the lumps from 13 layers to 7 raises the strain
  by about 20 %, the stress by 6 % and the plastic strain by 25 %.
* **The pattern shifts too.** The thinner-lump build's strain correlates at 0.77 with the
  study's, and the sign agrees in 75 % of the cells. The wall bending is the same in kind,
  but where the tension and compression fall moves with the lump boundaries.
* **For a paper.** Quote the §7.2 strains as a lower bound, or extrapolate in lump size.

**What this means for a paper.**
* **Robust.**
  * The down-facing contrast: a 2.3–2.8× slower front and 1.4–2.0× coarser cells.
  * Wall thickness and heat accumulation barely matter.
  * Wall bending as the residual-strain mechanism, and stress near yield.
* **To quote as ranges.**
  * Cell spacing: report the stable band, plus the value reached from a planar front.
  * Residual strain magnitude: about +20 % for thinner lumps.
* **Needs an experiment.**
  * Absorptivity, from single-track widths and depths. It is the input that moves the
    validation most.
  * Multicomponent segregation.
* **Ruled out.** Point-by-point exposure as the cause of the validation offset.

### 7.4 Grains over 20 layers: 3-D grain cellular automaton

**Question.** §7 grew grains for one layer only. The strong <001> build texture reported for
LPBF CoCrFeMnNi needs many layers of competitive growth. How do grain size and texture
develop over many layers, in thin and in thick walls?

**Method** (`benchmarks/literature/cantor_gyroid_ca.py`). The 3-D grain CA of §5.4 builds 20
layers (0.7–1.3 mm build height) of the graded gyroid in two 1 × 1 mm windows: thin walls
(x = 0.2–1.2 mm) and thick walls (x = 2.6–3.6 mm).
* **Process.** The settings of §7: 200 W, 0.75 m/s, 30 µm layers, 85 µm hatch, one contour,
  no rotation, clipped to the part in every layer.
* **Thermal history.** The steady validation track (powder layer on the plate; melts
  52 µm below the powder top) is shifted along every scan vector.
  * The pool ends are rounded where the laser switches on or off.
  * Not included: the shorter pools of short vectors, preheating, and the deeper pool over
    powder at down-facing surfaces. Powder that the pool reaches joins the part, with the
    plate-pool shape.
* **Grains.** The part below the first layer has random 20 µm grains. KGT kinetics of the
  CrFeCo–(MnNi) pseudo-binary (40 wt%, k = 0.74, freezing range 60 K). No nucleation.
  2.5 µm cells (43 M cells per window, about 45–65 min each).

| | layer 1 | layer 5 | layer 10 | layer 20 |
|---|---:|---:|---:|---:|
| thin walls: grain width (area-weighted) | 28 µm | 47 µm | 64 µm | 98 µm |
| thin walls: <100> within 15° of BD | 10 % | 16 % | 18 % | 13 % |
| thick walls: grain width | 29 µm | 49 µm | 70 µm | 103 µm |
| thick walls: <100> within 15° of BD | 8 % | 11 % | 11 % | 8 % |

Random orientations give 10.2 %. The 20 µm starting grains give 22 µm.

![gyroid grains over 20 layers](img/cantor_gyroid_ca.png)

**Reading the results.**
* **Grains coarsen steadily, whatever the wall thickness.** Epitaxy across layers makes the
  grains columnar, crossing many layers. Their width grows from 28 µm to about 100 µm in
  20 layers (0.6 mm) and has not levelled off. In the IN625 blocks of §5.5 the width is the
  same at 1 and 2 µm cells where the fibre stays weak (90°). Where it is strong (67° at 1 µm),
  it is 30 % larger.
* **The grain pattern follows the scan.** Without rotation, the grains in plan view are
  elongated along the hatch direction, with boundaries on the 85 µm track spacing.
* **Down-facing walls carry dross.** On the overhanging side of the inclined walls, each
  layer's pool reaches into the powder below. This leaves a step of stray grains per layer:
  the down-skin roughness of such walls.
* **Texture is weak, and stronger in thin walls than in thick ones.**
  * In the thin walls, <100> along BD rises to 18 % by layer 10. In the thick walls it stays
    at the random level.
  * The reason has not been checked. One candidate: in thin walls a larger share of the
    melt comes from the contour track, whose pool follows the inclined wall.
* **Growth mode matters over many layers.** With the heat-flow bias α = 1 (cells growing
  along the heat flow, §5.4), selection is weaker:
  * thin walls at layer 20 hold 100 grains instead of 69;
  * the grains are 84 µm wide instead of 98 µm;
  * <100>‖BD rises more slowly: 10 % instead of 16 % at layer 5, with a peak of 16 %
    instead of 19 %.

  Unlike the single track of §5.4, the build therefore depends on the growth mode. Measured
  grain widths and textures of the lattice would calibrate α.
* **The texture numbers are lower bounds.** In the IN625 blocks (§5.5), going from 2 to 1 µm
  cells raises the <100> fraction after 10 layers from 15–24 % to 35 % (90°) and from 26 % to
  68 % (67°).
  * At 2.5 µm cells the lattice runs therefore underestimate the fibre, and probably the
    grain width where selection is strong.
  * Whether the CA reproduces the strong <001> fibre reported for LPBF CoCrFeMnNi needs 1 µm
    cells on the lattice: about 700 M cells per window, a GPU task.

### 7.5 Under compression: where could twins or HCP start? (Level 0)

**Question.** If the gyroid is compressed, where, and at what applied strain, can the local
stress extend stacking faults? Extended faults are the common first step of deformation twins
and of the stress-induced FCC → HCP transformation. This Level 0 says where either could
start, not which one forms.

**Method** (`benchmarks/literature/cantor_gyroid_faults.py`, `tpmsfem.post.fcc_faults`).
* **Lattice.** The graded gyroid column of §7: 4 × 4 × 8 mm, walls 0.2 → 0.6 mm along x,
  0.1 mm voxels, relative density 0.29. As designed: no build plate, no residual stress.
* **Load.** Compression along the build direction between bonded platens, up to 3 % nominal
  strain.
* **Material.** As-built LPBF CoCrFeMnNi at room temperature, with J2 plasticity:
  * E = 203 GPa (Laplanche et al. 2015);
  * compressive yield 757 MPa (Kim et al. 2020);
  * linear hardening 1500 MPa (assumed, as in the residual-stress study).
* **Criterion.** In every element and for every grain orientation, the largest shear stress on
  the 12 {111}⟨112⟩ systems of the leading Shockley partial (τ\*). A fault extends where τ\* ≥ τ_c.
  * Each grain takes the stress of its element (a Sachs-type assumption).
  * Only yielded elements count: nucleation needs active dislocation sources.
* **Threshold.** τ_c = 360 MPa: the onset in the best-oriented grains (Schmid factor 0.5) at
  the measured twinning stress of CoCrFeMnNi, 720 MPa at 77 K and 293 K (Laplanche et al.,
  *Acta Mater.* 118 (2016) 152), with the same Sachs assumption.
  * Lower thresholds stand for lower effective fault energies, in proportion: cryogenic
    temperature, Mn/Fe-rich variants, or a thick HCP embryo (γ_SF − 2σ, Olson & Cohen 1976).
  * A stricter bound lets both partials move freely (Byun 2003), with the measured γ_SF =
    30 mJ/m² (Okamoto et al. 2016). It needs about twice the measured twinning stress.
* **Orientations.** Volume-weighted samples (600 each) of the grain-CA textures of the thin-
  and thick-wall windows (§7.4), compared with a random texture.
* **Checks.**
  * The criterion's Schmid factors reproduce the known twinning asymmetry of FCC single
    crystals: [001] 0.47 in compression and 0.24 in tension; [111] 0.31 in tension and 0.16 in
    compression (tests/test_fcc_faults.py).
  * **Resolution.** A 0.05 mm run (298,000 voxels) was stopped by a container restart after the
    2 % step. Up to 2 % it gives:
    * the nominal stress within 2 %;
    * 8–14 % more (relative) fault-prone volume at τ_c = 360 MPa: 6.9 % against 6.2 % at 1 %
      strain, 11.9 % against 10.4 % at 2 %;
    * at τ_c = 240 MPa, within 4–9 % and slightly lower: 43.0 % against 43.6 % at 2 %;
    * a free-partials maximum of 28.7 mJ/m² against 22.5 at 2 %: thin-wall stress peaks are
      under-resolved at 0.1 mm.
    * Results in `results/cantor_gyroid_faults_vox0.05.json`.

**Results** (0.1 mm voxels). Share of the lattice volume where faults can extend, counting
only yielded material, with the CA textures:

| nominal compressive strain | 0.4 % | 0.6 % | 1 % | 2 % | 3 % |
|---|---:|---:|---:|---:|---:|
| nominal stress [MPa] | 59 | 70 | 78 | 85 | 88 |
| yielded volume | 8 % | 21 % | 39 % | 59 % | 69 % |
| **τ_c = 360 MPa (CoCrFeMnNi twinning onset)** | **1.0 %** | **3.1 %** | **6.2 %** | **10.4 %** | **13.1 %** |
| τ_c = 300 MPa (fault energy × 0.83) | 3.1 % | 8.7 % | 16.6 % | 26.5 % | 31.9 % |
| τ_c = 240 MPa (× 0.67) | 5.4 % | 14.8 % | 27.8 % | 43.6 % | 51.7 % |
| τ_c = 120 MPa (× 0.33) | 8.2 % | 21.4 % | 38.6 % | 58.7 % | 68.5 % |
| free-partials bound, 30 mJ/m² | 0 | 0 | 0 | 0 | 0 |

* **Free-partials bound.** The largest fault energy the stress overcomes under it is 18–24
  mJ/m² at 0.1 mm, and 28.7 mJ/m² at 0.05 mm at 2 % strain.
* **Texture.** The random texture gives the same fractions within 0.1 percentage points.
* **Thin and thick walls.** Their halves differ by under 0.2 points.

![Gyroid faults](img/cantor_gyroid_faults.png)

**Reading the results.**
* **Faults can extend as soon as the walls yield.**
  * The as-built yield stress (757 MPa) is already above the measured twinning stress of wrought
    CoCrFeMnNi (720 MPa). So grains with a leading-partial Schmid factor of at least 0.48 reach
    τ_c as soon as their wall yields.
  * Hardening then brings in less favourable grains.
  * The fault-prone material follows the plastic zones: the bending hinges halfway along the
    inclined wall segments, where plastic strain reaches about 7 % at 3 % nominal strain.
  * At room temperature this predicts twinning-type faults in about 6 % of the lattice at 1 %
    strain and 13 % at 3 %.
* **What this says about HCP.** Level 0 does not decide between twins and HCP: that depends on
  the FCC/HCP free-energy difference.
  * The equiatomic alloy forms HCP only at hydrostatic pressures of order 10–20 GPa (Tracy et
    al. 2017; Zhang et al. 2017), far above what a gyroid sees. So the faults here are most
    likely twins.
  * For an alloy or temperature with a third lower effective fault energy (a lower-SFE variant,
    cryogenic loading, or a thick HCP embryo), the fault-prone volume rises four-fold: 28 % of
    the lattice at 1 % strain, 52 % at 3 %.
  * Below about a third of the room-temperature fault energy, every yielded region qualifies.
    That is where a stress-induced FCC → HCP transformation in the gyroid becomes plausible.
* **The strict free-partials bound gives almost no extended faults.**
  * At 0.1 mm the local stresses never overcome more than 24 mJ/m², against the measured 30.
  * At 0.05 mm the peaks reach 28.7 mJ/m² at 2 % strain, so a few hot spots may reach the bound
    at finer resolution or larger strain.
  * Either way the calibrated criterion, not this bound, matches the measured twinning stress.
* **Texture and wall thickness hardly matter here.**
  * In compression along the build direction, the thin-wall texture has more favourably
    oriented grains: 28 % with a Schmid factor ≥ 0.45, against 25 % for random and 18 % for the
    thick walls.
  * But the walls are inclined and each is loaded along its own direction, so in the lattice the
    texture effect averages out.
  * The graded walls yield at the same nominal strains in both halves of the column.
* **Limits.**
  * Each grain takes the stress of its element: grain-to-grain stress differences need crystal
    plasticity (Level 2).
  * The hardening (1500 MPa) is assumed and controls how fast the fraction grows after onset.
  * The main results start from a stress-free lattice. The as-built start is below.
  * The threshold is calibrated on wrought, coarse-grained CoCrFeMnNi. The LPBF cell structure,
    dislocation density and Mn/Ni segregation may shift it.
  * Room temperature only; compression along the build direction only; a column two cells tall
    between bonded platens.
* **Next.**
  * Level 1: an Olson–Cohen kinetic law for the HCP fraction, coupled to the plasticity.
  * The as-built residual state as the starting point.
  * Measurement: in-situ synchrotron or neutron diffraction during compression, to look for HCP
    peaks, with EBSD/TEM of interrupted samples.


**From the as-built state** (`benchmarks/literature/cantor_gyroid_faults_asbuilt.py`).
* **Set-up.**
  * The thermo-mechanical build of §7.2 is re-run: layer lumps on a 1 mm plate, cooling to
    20 °C, release onto a 3-2-1 support. Its state is kept: residual stress and build plastic
    strain.
  * The column is then compressed with the plate bottom held and the lattice top moved down.
  * The stress-free reference is the same mesh, plate and supports.
  * Only material that yields during the compression counts: the build has already yielded
    much of the part.
* **Before any load.** The residual stress alone (lattice median von Mises 584 MPa, 95th
  percentile 748 MPa) already gives τ\* ≥ 360 MPa in 3.7 % of the lattice, and ≥ 240 MPa in 45 %.

| nominal compressive strain | 0.4 % | 0.6 % | 1 % | 2 % | 3 % |
|---|---:|---:|---:|---:|---:|
| nominal stress [MPa], as-built / stress-free | 44 / 58 | 57 / 69 | 71 / 78 | 84 / 85 | 88 / 88 |
| yielding during compression | 39 / 7.5 % | 47 / 21 % | 60 / 38 % | 75 / 59 % | 82 / 69 % |
| **faults can extend, τ_c = 360 MPa** | **5.5 / 0.8 %** | **6.9 / 3.0 %** | **9.4 / 6.1 %** | **13.6 / 10.4 %** | **16.2 / 13.0 %** |
| faults can extend, τ_c = 240 MPa | 27 / 4.9 % | 33 / 14 % | 42 / 28 % | 55 / 44 % | 61 / 52 % |

![As-built start](img/cantor_gyroid_faults_asbuilt.png)

* **The residual stresses bring the onset forward.** Where they add to the applied load, the
  walls yield at a lower applied strain.
  * At 0.4 % strain the as-built column has five times as much yielding material and 6.5 times
    as much fault-prone volume as the stress-free one.
  * The difference shrinks as the load takes over: 1.5 times at 1 % strain, 1.25 times at 3 %.
* **The macroscopic response softens early.** The nominal stress is 24 % lower at 0.4 % strain.
  The curves meet at about 2.5 %.
* **For an experiment.** In as-built lattices the first faults (twins) should appear at
  smaller strains than a stress-free model predicts. A stress-relief anneal would remove most of
  the difference.
* **Also with the build.**
  * The build model's own assumptions apply: lumped layers, no powder, 0.1 mm voxels (§7.2).
  * The stress-free reference sits on the plate, unlike the plate-free column above. It gives
    the same fault-prone volume within 0.1 point at 1 % strain.

### 7.6 How much HCP: a gyroid of a metastable FeMnCoCr alloy (Level 1a)

**Question.** If the gyroid were printed from an alloy that does transform, how much HCP would
compression produce, and where? The equiatomic CoCrFeMnNi of §7.5 twins rather than
transforming at room temperature, so its HCP fraction stays at zero. This section therefore uses
the metastable Fe50Mn30Co10Cr10 (at%), for which LPBF data exist.

**Method** (`benchmarks/literature/femncocr_gyroid_hcp.py`, `tpmsfem.post.hcp_kinetics`).
* **Lattice and loading.** As in §7.5: the graded gyroid column, 0.1 mm voxels, compressed along
  the build direction between bonded platens. Here up to 5 % nominal strain (small-strain J2;
  geometric nonlinearity not included).
* **Material.** LPBF Fe50Mn30Co10Cr10 at room temperature (Micromachines 15 (2024) 123,
  doi:10.3390/mi15010123, Table 1):
  * yield 487.6 MPa, tensile strength 744.9 MPa, elongation 22.3 %;
  * linear hardening 1300 MPa, estimated from the tensile strength;
  * E = 200 GPa and ν = 0.28, assumed.
  * The measured flow curve already contains the TRIP hardening, so the coupling is one-way:
    the transformation strain is not fed back.
* **Kinetics.** An Olson–Cohen law in each element's equivalent plastic strain:
  f = f0 + (1 − f0)[1 − exp(−β (1 − exp(−α ε_p))^n)].
  * It is fitted to the same study's EBSD fractions in tension: 0.94 % as built, 28.1 % at 20 %
    strain.
  * One measured point fixes one rate constant. Two shapes bound the rest: exponential (n = 1)
    and sigmoidal (n = 2, α = 5).

**Results.** The two kinetic shapes are given as exponential / sigmoidal. As built, the alloy
holds 0.94 % HCP.

| nominal compressive strain | 1 % | 2 % | 3 % | 5 % |
|---|---:|---:|---:|---:|
| nominal stress [MPa] | 53 | 56 | 58 | 62 |
| yielded volume | 51 % | 68 % | 76 % | 85 % |
| plastic strain, lattice mean / 95th percentile | 0.2 / 1.1 % | 0.6 / 2.6 % | 0.9 / 4.1 % | 1.7 / 7.0 % |
| **HCP, lattice mean** | **1.3 / 1.0 %** | **1.8 / 1.2 %** | **2.4 / 1.4 %** | **3.5 / 2.2 %** |
| HCP, 95th percentile | 2.7 / 1.2 % | 5.1 / 2.2 % | 7.4 / 3.7 % | 11.6 / 7.7 % |
| lattice volume with ≥ 10 % HCP | 0 / 0 | 0.4 / 0.1 % | 2.0 / 0.5 % | 7.1 / 3.2 % |

![FeMnCoCr gyroid HCP](img/femncocr_gyroid_hcp.png)

**Reading the results.**
* **Little HCP forms at typical lattice-test strains, and only in the hinges.**
  * The plastic strain of a compressed gyroid is very uneven: at 5 % nominal strain the lattice
    mean is 1.7 %, but the bending hinges of the inclined walls reach 7 % (95th percentile).
  * So the lattice-average HCP rises only from 0.94 % to 2–3.5 %, while the hinges reach 8–12 %.
  * A diffraction measurement averaged over the whole lattice would see little. One focused on
    the hinges, or EBSD of an interrupted sample, would see the transformation.
* **The shape of the kinetics at small strain is the main uncertainty.**
  * The one measured point (28 % at 20 % strain) leaves the small-strain part of the curve open.
  * The exponential and sigmoidal shapes differ by a factor of 1.6 on the mean at 5 % strain,
    and by up to 2.3 on the hinges at 1–3 %.
  * A measured HCP fraction at 2–10 % strain would remove most of this. That could come from
    interrupted tests or in-situ diffraction of the bulk alloy.
* **Against the equiatomic alloy.**
  * Fe50Mn30Co10Cr10 yields lower (488 against 757 MPa), so the lattice yields earlier and more
    widely: 51 % of it at 1 % strain, against 39 % for CoCrFeMnNi.
  * The equiatomic alloy forms no HCP at room temperature; its faults would become twins
    (§7.5).
* **Limits.**
  * The kinetics are calibrated in tension and applied to the local plastic strain whatever the
    stress state: the stress-state dependence (Stringfellow et al. 1992) is not included, nor the
    tension-compression asymmetry of the partials (§7.5).
  * The coupling is one-way: no transformation strain or volume change acts on the stresses.
  * Small-strain J2 with estimated hardening and assumed elastic constants. The few voxels with
    plastic strains above about 20 % are outside its range, so maxima are not reported.
* **Next.**
  * Level 1b: transformation strain and stress-state-dependent kinetics in the constitutive
    update.
  * Calibration on HCP fractions measured at several strains.
