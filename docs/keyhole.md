# Keyholes: depth, stability and pores

**Question.** When does the laser drill a keyhole instead of melting a shallow pool? How deep is
it, how much does it fluctuate, and when does it start trapping pores? The mechanism: local
evaporation on the front wall, a bulge, a vapour jet against the back wall, and a bubble
pinched off at the keyhole tip that the solidification front freezes in as a pore.

The work runs in two stages:
* **Stage 1 (this page, done).** Process maps from the scaling laws of Gan et al. These are
  checked against operando X-ray keyhole depths, Ti-6Al-4V first, then applied to IN625 and the
  CoCrFeMnNi gyroid.
* **Stage 2 (next).** Keyhole dynamics in the free-surface flow solver: ray tracing with
  multiple reflections, a bubble model, and pore pinch-off. Small cases are tested on a CPU;
  production runs go on a GPU (plan at the end).

## Stage 1: scaling laws

**Laws.** From Gan, Kafka, Parab, Zhao, Fang, Heinonen, Sun & Liu, Nat. Commun. 12 (2021) 2379
(CC BY 4.0). They are implemented in `tpmsfem.process.defects`:

| quantity | law (Gan et al.) | function |
|---|---|---|
| keyhole number | Ke = η P / ((T_l − T₀) π ρ c_p √(α v r₀³)) | `keyhole_number` |
| process absorptivity | η = 0.7 [1 − exp(−0.6 Ke_m L_d*)], Ke_m L_d* = η_m P / ((T_l − T₀) π ρ c_p v r₀²); floored at η_m | `keyhole_absorptivity`, `keyhole_porosity_number` |
| keyhole depth | e* = e / r₀ = 0.4 (Ke − 1.4), zero below 1.4 | `keyhole_aspect_ratio`, `keyhole_depth` |
| fluctuation | Δe* = 0.36 Ke^0.86 (range of e* over time) | `keyhole_fluctuation` |
| regime | conduction ≤ 1.4 < transition ≤ 6 < keyhole | `keyhole_regime` |
| stability | stable ≤ 16 (almost no pores) < intermediate ≤ 30 < chaotic (pores, spatter) | `keyhole_stability` |
| porosity | φ = 0.10 erfc(NED − 1.51) + 0.01 erf(Ke_m L_d* − 2.32) + 0.055 | `normalized_energy_density`, `gan_porosity` |
| all at once | arrays over power and speed | `keyhole_state` |

**Conventions.** The thresholds hold only with the conventions under which they were fitted:
* liquid properties at the liquidus;
* the process absorptivity of Eq. 6. It rises from the flat-melt value η_m towards 0.7 as the
  depression traps reflections.

A fixed flat-surface absorptivity puts Ke on a different scale. The paper does not state the
unit of φ in the porosity law, so only its two midpoints are used as criteria:
* NED = 1.51, for lack of fusion;
* Ke_m L_d* = 2.32, for keyhole pores.

**Script:** `benchmarks/literature/keyhole_maps.py` (seconds). **Tests:**
`tests/test_defects.py`. They check the limits, the regime edges, that the diffusivity cancels
in Ke_m L_d*, the reproduction of the data set's absorptivity column, and three X-ray depths.

## Validation on X-ray keyhole depths

**Data.** Gan et al. Supplementary Data 1, transcribed unchanged to
`benchmarks/literature/data/gan2021_keyhole_xray.json`. It holds 87 keyhole depths from
synchrotron X-ray imaging of bare plates under argon (APS 32-ID-B):
* 71 Ti-6Al-4V, the tracks of Cunningham et al., Science 363 (2019) 849: r₀ = 47.5 and 70 µm,
  100–520 W, 0.4–1.2 m/s, depths 15–445 µm;
* 9 Al6061 and 7 SS316.

**Two routes for the absorptivity.**
* *Published η.* Use the absorptivity column of the data set. This checks Ke and the depth law
  as implemented.
* *Predictive.* Compute η from Eq. 6 with the flat-melt η_m of their Supplementary Table 1:
  Ti-6Al-4V 0.26, SS316 0.33, Al6061 0.01. This is the chain applied to a new alloy:
  P, v, r₀ → η → Ke → e.

| material | n | published η: depth RMSE, bias, R² (e*) | predictive: depth RMSE, bias, R² (e*) |
|---|---|---|---|
| **Ti-6Al-4V** | 71 | **18 µm, −7 µm, 0.977** | **17 µm, −2 µm, 0.981** |
| SS316 | 7 | 25 µm, −24 µm, 0.958 | 12 µm, −1 µm, 0.991 |
| Al6061 | 9 | 23 µm, +15 µm, 0.970 | 87 µm, +80 µm, 0.59 |
| all | 87 | 20 µm, −6 µm, 0.977 | 32 µm, +7 µm, 0.925 |

![keyhole maps](img/keyhole_maps.png)

**Reading the validation.**
* **Ti-6Al-4V keyhole depths are predicted to 17–18 µm RMS over 15–445 µm** (mean relative
  error 16–17 %), from the process parameters alone (panel b).
* **The law is not just a Ti-6Al-4V fit.** Fitted to the 16 Al6061 and SS316 tracks only,
  e* = 0.388 (Ke − 1.02) predicts the 71 Ti-6Al-4V depths with RMSE 18 µm (R² 0.978).
  Gan's universal 0.4 (Ke − 1.4) gives the same. Ti-6Al-4V alone gives 0.404 (Ke − 1.08)
  (panel a).
* **The regime edges hold on the data.**
  * Of the 35 tracks with Ke ≤ 6, 33 have e* ≤ 2, Gan's transition range.
  * Of the 52 tracks above 6, 51 have e* > 2.
* **The absorptivity column is Eq. 6 with a slightly different η_m.** It is reproduced exactly
  with η_m = 0.242 (Ti-6Al-4V), 0.256 (SS316) and 0.0075 (Al6061), floored at the table values.
  * Using the table values instead costs nothing for Ti-6Al-4V and SS316.
  * For Al6061 it overestimates the depth by 80 µm. With η_m near 0.01, a 33 % change moves η
    and Ke steeply.
  * Our alloys have η_m near the steels, so they are in the well-behaved range.
* **Front-wall angle, for stage 2.** tan θ = 0.77 e* + 0.49 (r = 0.988, n = 87). A flow
  solver with ray tracing has to reproduce this wall angle, not only the depth.
* **Fluctuation (Eq. 5) is not checked here.** The data set gives the time-averaged depth only.

**IN625 check on AM-Bench 2018** (bare plate, r₀ 85 µm). The measured pools have D/W =
0.28–0.29.
* With η_m = 0.33, Ke is 2.0 (A), 1.9 (B) and 1.5 (C): just past the conduction edge. With the
  85 µm spot, η stays at η_m.
* The predicted depressions are shallow: 21, 15 and 4 µm.
* This agrees with the independent calibration of the conduction model on case B: absorptivity
  0.323 and an absorption depth of 15.3 µm, the effective treatment of a shallow vapour
  depression.
* The agreement depends on η_m: 6 µm with η_m = 0.28, 25 µm with 0.38.

## IN625: Locatelli et al. (2026), with the consistent Ke

**The experiment.** Print Genius 150, 30 µm layers, 50 µm hatch, plate at 80 °C. Their text
reports:
* keyhole pores at 250 and 300 W, 0.5 m/s;
* dense parts at 150–300 W and 1.0–1.5 m/s.

The spot radius is not reported, so it is scanned from 20 to 60 µm. The pore criterion is
Ke > 16.

| η_m | outcomes reproduced (of 10) | spot radius that reproduces all |
|---|---|---|
| 0.28 | 10 | 39–45 µm |
| 0.33 | 10 | 40–46 µm |
| 0.38 | 10 | 41–47 µm |

* **All ten outcomes are reproduced for r₀ = 40–46 µm, whatever η_m** (panel c). At 43 µm:
  * the porous cases have Ke = 18.6 and 23.1;
  * the dense ones have Ke = 3.3–13.4;
  * the 1/e² diameter is 86 µm, typical of LPBF machines.
* **This supersedes the 30 µm radius of [porosity_maps.md](porosity_maps.md).** That estimate
  used a fixed absorptivity of 0.32 and properties at 700 °C, not Gan's convention. On the
  consistent scale, 30 µm reproduces only 5 of the 10 outcomes: the dense 1.0 m/s cases come
  out at Ke 17–28.
* **The porosity-law midpoint agrees.** At 43 µm, Ke_m L_d* = 2.32 reproduces 9 of 10. The miss
  is 300 W at 1.0 m/s (2.56), which is reported dense (dashed line in panel d).

## The CoCrFeMnNi gyroid

**Inputs.** The study's settings: 200 W, 0.75 m/s, r₀ 35 µm, 30 µm layers. The part
temperatures come from the heat-accumulation study ([process_models.md](process_models.md)
§7.1). η_m = 0.33 (0.28–0.38).

| part temperature | η | Ke | keyhole depth (range over time) | regime | r₀ 45 µm: Ke, depth |
|---|---|---|---|---|---|
| 80 °C (plate) | 0.62 | 14.8 (14.0–15.3) | 187 µm (124–251) | keyhole, stable | 8.3, 125 µm |
| 140 °C (~9 mm up) | 0.62 | 15.7 | 201 µm (133–268) | keyhole, stable | 9.0, 136 µm |
| 300 °C | 0.65 | 18.8 | 244 µm (165–322) | keyhole, intermediate | 11.0, 173 µm |
| 500 °C (upper bound) | 0.67 | 24.1 | 318 µm (221–416) | keyhole, intermediate | 14.8, 241 µm |

**Reading the gyroid results.**
* **The gyroid study runs in keyhole mode.** The keyhole is about 190 µm deep, six layers.
  This is at the edge of the stable range (Ke 14.8 against 16), panels e and f.
* **Heat accumulation pushes it over.** Ke crosses 16 at a part temperature of about 155 °C.
  The heat-accumulation study (about 7 °C per mm) puts this near 11 mm of build height, above
  the study's 8 mm columns. With η_m = 0.38 the crossing comes at 122 °C, about 6 mm up, so
  the top of a column would cross it. Taller builds, or a hotter part, enter the range where
  keyhole pores appear.
* **The spot size decides more than the alloy data.** With a 45 µm radius, as inferred for the
  IN625 machine, Ke stays at 8–15 up to 500 °C: stable throughout.
  * The full η_m range moves Ke by ±0.6 only.
  * Measuring the spot of the machine used is therefore the most useful single input.
* **Consequence for the earlier gyroid results.** The conduction model of §7 gave a 54 µm deep
  pool at this point. A 190 µm keyhole means a deeper, narrower pool and different
  solidification conditions at its root. The cell sizes and segregation of §7 belong to the
  upper pool, where the conduction model was validated (Piglione et al.). Stage 2 will quantify
  the root.

## Limits

* **Range of the data.** The laws were fitted to bare plates in argon with r₀ = 44–70 µm.
  * The gyroid's 35 µm spot is an extrapolation, as is the low end of the inferred IN625
    radius.
  * A powder layer shifts the absorptivity at low power. Gan et al. find the effect on the
    keyhole small once the powder melts.
* **η_m is not measured for IN625 or CoCrFeMnNi.** The SS316 value is used, with ±0.05 bounds.
* **IN625 liquid properties.** The repository table ends at 870 °C and is held constant to the
  liquidus. Ke scales as (ρ c_p k)^−1/2, so a 30 % error in that product moves Ke by 15 %.
* **Pores are inferred from thresholds,** Ke 16 and 30 and the midpoint of the porosity law,
  not simulated. The fluctuation law is taken from the paper without an independent check.
* **The IN625 check is again one data set from text.** The spot is inferred, not measured.

## Stage 2 (next): keyhole dynamics in the flow solver

The free-surface lattice Boltzmann solver ([freesurface_lbm.md](freesurface_lbm.md)) already
has melting, Marangoni flow, recoil pressure, evaporation and radiation. A keyhole needs four
more things:
1. **Ray tracing with Fresnel absorption.** Multiple reflections replace the vertical
   Beer–Lambert column. The target is Eq. 6 (η rising to ~0.6 in deep keyholes) and the
   front-wall angle above.
2. **A bubble model.** Each enclosed gas region gets its own ideal-gas pressure, so a bubble
   pinched off at the tip survives and can be trapped as a pore.
3. **Fine cells.** 2–3 µm cells, without the recoil cap and speed limiter used at 10 µm.
4. **Diagnostics.** Keyhole depth and its fluctuation over time, front-wall angle, absorbed
   power, and pore count and size.

**Validation targets:** the 71 Ti-6Al-4V depths above, and the pore onset between Ke 16 and
30. **Applications:** IN625 and the gyroid, at 80 and 300 °C. Small cases are tested on a
CPU; production cases run on a GPU.
