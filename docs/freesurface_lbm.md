# Free-surface lattice Boltzmann melt pool (powder-scale LPBF)

`tpmsfem.process.freesurface` is our own melt-flow solver for laser powder-bed fusion. The
melt is a liquid with a moving surface; the gas above it is not simulated. It is written
against tpmsfem's array backend, so the same code runs on the CPU (NumPy) and on a GPU
(CuPy).

It is the model type of Lei et al., Cell Rep. Phys. Sci. 6 (2025) 102674 (the "HF-LBM" /
PS-MP model), whose code is not public. Status: stages 1 and 2 (bare plate, full surface
physics) are done and verified. Powder beds and multi-track GPU runs come next.

## 1. Method

**Flow** (`FreeSurfaceLBM`, lattice units).
* **Lattice and collision.** A D3Q19 lattice with two-relaxation-time collision (TRT, magic
  parameter 3/16) and Guo forcing. Liquid IN625 needs a relaxation time of about 0.504; TRT
  keeps that stable.
* **Free surface** (Koerner et al. 2005). Each cell is gas, interface, liquid or wall.
  * Interface cells carry a fill level and exchange mass with their neighbours through the
    streamed populations.
  * Populations arriving from the gas are rebuilt from the gas pressure. That pressure
    carries the Laplace pressure σκ and the recoil pressure.
  * Cells fill and empty, and the excess mass is redistributed, so mass is conserved to
    round-off.
* **Curvature.** From the smoothed fill-level field. Walls act as mirrors, which gives a
  neutral 90° contact. A height-function curvature (as in HF-LBM) is available as an option.
* **Melting and solidification.** Carman–Kozeny (Darcy) damping −C(1−f_l)²/(f_l³+b) ρu,
  applied semi-implicitly (C = 1e9 s⁻¹). Solid metal is part of the lattice and stays still
  for any C, so plate, powder and melt share one set of cells.

**Heat** (`FreeSurfaceMeltPool`, SI units).
* **Solver.** Enthalpy on the same cells: finite volume, explicit, with conduction and upwind
  advection by the melt velocity. Each cell's heat capacity is its metal mass.
* **Laser.** Absorbed below the first surface seen from above, with an exponential absorption
  depth. This is the same discretisation as the conduction model `MeltPoolModel`, so the two
  can be compared directly.
* **Surface terms.** These act on interface cells. The surface area per cell is its gas faces
  divided by |n_x| + |n_y| + |n_z|, so a flat surface gets exactly one cell of area. The
  terms are:
  * Marangoni stress (dσ/dT)∇_sT, as a continuum surface force;
  * recoil pressure 0.54 p_sat(T), from Clausius–Clapeyron;
  * evaporation cooling (Hertz–Knudsen–Langmuir, the same `Evaporation` model as the conduction
    model);
  * radiation.

**Time step.** It is set by the melt speed: Mach < 0.1. That gives 0.1–0.3 µs at 5–10 µm
cells, as in HF-LBM (0.3 µs at 6 µm).

## 2. Verification

`tests/test_freesurface.py` and `tools/lbm/compare_freesurface_track.py`.

| check | result |
|---|---|
| mass conservation (droplets, collapsing column, melt pools) | 1e-13 relative |
| Laplace pressure of a resting droplet, R = 6–10 cells | within 1 % of 2σ/R |
| spurious currents at a resting droplet | 2e-4 to 8e-4 (lattice units) |
| surface shear stress on a periodic layer → Couette profile τz/μ | within 1 % in every cell |
| Marangoni force on a flat surface | exactly dσ/dT · ∂T/∂x / dx in the surface cell, zero below |
| plate at rest, surface tension on | stays exactly at rest (u < 1e-12) |
| heat only, AM-Bench case B at 5 µm, against `MeltPoolModel` | 0.3 % RMS temperature difference; pool 290 × 110 × 30 µm vs 295 × 110 × 30 µm; energy to 1e-15 |
| energy balance with flow | round-off without surface losses or with radiation; 0.2 % with evaporation; 0.3–0.5 % with Marangoni flow (heat carried with the moving surface) |

## 3. AM-Bench 2018-02 case B with flow

`tools/lbm/freesurface_caseB.py`.
* **Case.** An IN625 plate, 179.2 W at 0.8 m/s.
* **Inputs.** The absorptivity (0.3234) and absorption depth (15.3 µm) of the calibrated
  conduction model; viscosity 5 mPa s; σ = 1.8 N/m; dσ/dT = −1.1e-4 N/(m K).
* **Grid.** 10 µm cells on 4 CPU cores, 98,000 cells, 4,700 steps of 0.2 µs, about 14 min
  per run.

| | width | depth | peak T | melt speed | surface |
|---|---|---|---|---|---|
| measured (NIST AMMT) | 123.5 ± 6.5 µm | 36.0 ± 1.9 µm | | | |
| flow without Marangoni | 115 µm | 35.5 µm | 2450 °C | 0.02 m/s | flat (±2 µm) |
| flow with Marangoni | 120 µm | 34.5 µm | 2320 °C | 0.3–0.35 m/s | ±2–6 µm |

Width and depth are interpolated at the solidus in the cross-section at 75 % of the track.

![case B](img/freesurface_caseB.png)

**Reading the results.**
* **Marangoni flow does what it should.** With dσ/dT < 0 the hot surface flows outwards. The
  pool widens (115 → 120 µm), gets a little shallower, and its peak temperature drops by
  130 °C.
* **Both runs fall inside the measured scatter.** This is not an independent validation,
  because the absorptivity was calibrated on the conduction model. The next step is to
  calibrate it in the flow model itself, on cases A–C together.
* **Resolution.** 10 µm cells resolve the 36 µm depth with only four cells. The surface
  shows single-cell speckle of a few µm, which is interface noise at this resolution, not
  topography. Production runs need 2.5–6 µm cells, which means a GPU.

## 4. Performance and the GPU

* **CPU.** On 4 cores with NumPy, the solver does 0.5 million cell updates per second
  (MLUPS). It is written as whole-array operations, so on a GPU the same code runs through
  CuPy unchanged.
* **GPU (estimate).** That array form should give a few hundred MLUPS on a modern GPU; this
  was not measured here. Fusing the step into a few CUDA kernels, as HF-LBM does, is worth
  several times more.
* **Grid refinement.** HF-LBM uses three grid levels: 36 µm for the layer, 12 µm per region
  and 6 µm only around the current track. That, not raw speed, is what makes thousands of
  tracks per week possible on one 12 GB TITAN V. A moving fine window is therefore the next
  structural step. With it, a single 12–24 GB GPU is enough.

## 5. Compared with HF-LBM (Lei et al. 2025)

| | HF-LBM / PS-MP | tpmsfem free surface (now) |
|---|---|---|
| flow | single-phase free-surface LBM | the same (D3Q19 TRT, Koerner) |
| heat | thermal LBM in the melt, finite differences outside | finite volume everywhere |
| curvature | height function | smoothed gradient (height function optional) |
| Marangoni, recoil, evaporation | yes | yes |
| melting / solidification | yes | Darcy damping, enthalpy |
| powder bed | built-in generator | **next** |
| grid | 3-level refinement 36 / 12 / 6 µm | uniform; **moving fine window next** |
| hardware | CUDA, one TITAN V | NumPy / CuPy; fused kernels **next** |
| microstructure | LBM-CA grains, CET | our CA grains and phase field, from the same thermal history |
| code | not released | Apache-2.0, in this repository |

## 6. Powder beds and calibration (in progress)

**Powder generator** (`process/powder.py`, tested). It produces:
* log-normal particle sizes (median 30 µm, 15–45 µm);
* rain-and-roll packing, where each particle rolls downhill until it is held;
* a recoater cut.

Measured: packing 0.43 for a 40 µm layer and 0.50–0.52 for deep beds; no overlaps; voxelised
volume within 0.1 %.

**Solver changes for powder.**
* **Laser.** Absorbed by Beer–Lambert along the metal in each column, so light passes
  through gaps and a thin sliver of a particle takes only its share.
* **Cold powder.** Touching solid surface cells exchange no mass.
* **Stability.** A lattice speed limiter (0.25), a recoil cap (both counted), and gravity.
* **Mass.** Conserved to 1e-13 on plate and powder.

**Powder tracks are not reliable yet.** When the powder melts and the surface churns, the
heat balance goes wrong:
* The finite-volume enthalpy advection disagrees with the free-surface mass exchange, giving
  energy errors of 5–16 % and particles running away to 6,000–10,000 °C.
* Carrying the enthalpy with the lattice mass flux (`energy_transport="lbm"`, the idea behind
  HF-LBM's thermal distribution) cuts the powder energy error to 3 %. On the plate, however, it
  runs hotter than the finite-volume version, which should not happen.

So the next work is:
* find that error in the `"lbm"` transport;
* a conservative treatment of nearly empty cells;
* implicit evaporation cooling;
* smaller time steps for recoil-driven flow (a GPU job).

**Calibration on AM-Bench case B** (10 µm cells, absorption depth 5 µm instead of the
conduction model's fitted 15.3 µm):

| dσ/dT [N/(m K)] | absorptivity | width | depth | peak T |
|---|---|---|---|---|
| −1.1e-4 (outward flow) | 0.32 | 136 µm | 30.5 µm | 2870 °C |
| +1.0e-4 (inward flow) | 0.32 | 151 µm | 35.3 µm | 6000 °C (table limit) |
| measured | | 123.5 ± 6.5 µm | 36.0 ± 1.9 µm | |

* **Outward flow gives a pool too wide and too shallow.** A higher absorptivity would widen it
  further.
* **Inward flow fixes the depth** but focuses the heat, until recoil spreads the surface.
* **What is missing.** A single constant dσ/dT cannot give the measured shape. IN625 contains
  sulfur and oxygen, which make dσ/dT positive at lower temperatures and negative at higher
  ones (Sahoo et al. 1988). That temperature-dependent surface tension is the next physics to
  add. The conduction model hides the gap with its fitted 15 µm absorption depth.
* **Not finished.** The runs at absorptivity 0.40 were lost to a container restart and were
  not repeated.

## 7. Next steps

1. **Heat that moves with the melt, conservatively.** Fix the `"lbm"` energy transport, add
   implicit evaporation and handle nearly empty cells. Then single tracks on powder: balling,
   width and roughness.
2. **Temperature-dependent surface tension (Sahoo, sulfur-bearing alloys).** Then calibrate
   the absorptivity on AM-Bench cases A–C at 5 µm.
3. **Moving fine window and coarse thermal background.** Fused CUDA kernels.
4. **Height-function curvature with consistency checks.** A better curvature for powder and
   keyholes.
5. **Multi-track layers on a GPU.** For example the 20-track IN718 square of Lei et al.:
   225 W, 1.0 m/s, about 100 µm hatch.

## Reproducing

```bash
pytest -q tests/test_freesurface.py
python tools/lbm/compare_freesurface_track.py frozen      # heat only, 5 um, ~2 min
python tools/lbm/freesurface_caseB.py marangoni 10        # ~15 min on 4 cores
python tools/lbm/freesurface_caseB.py flat 10
python tools/lbm/freesurface_figure.py
```
