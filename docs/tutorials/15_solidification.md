# Tutorial 15: Solidification conditions, spacing and grains

**Goal.** From a melt-pool run, extract the thermal gradient G and the solidification speed R
along the fusion boundary, predict cell spacing and the columnar-to-equiaxed transition, grow
grains with a reduced model, and export the thermal history for ExaCA. About 1 minute.

**You will learn**
* `gr_fields` and `summary`;
* spacing laws (Hunt, Kurz–Fisher), Hunt's CET criterion;
* `grow_grains` and IPF colours; ExaCA export.

## Step 1: a melt pool

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector
from tpmsfem.process import solidification as S

th = ThermalProperties.from_si([20.0, 1290.0, 1350.0, 3000.0], [9.8, 26.0, 30.0, 30.0], [410.0, 650.0, 720.0, 720.0],
                               8440.0, 2.9e5, 1290.0, 1350.0)
IN625 = Material("IN625", E=208e3, nu=0.28, thermal=th)
mp = MeltPoolModel(IN625, h=0.01, length=0.6, width=0.15, depth=0.08, T0=25.0, emissivity=0.4, evaporation=Evaporation())
res = mp.run(Laser(180.0, 0.35, 0.05, depth=0.015), [ScanVector(0.1, 0.0, 0.5, 0.0, speed=800.0)], t_extra=2e-4)
```

## Step 2: G and R

```python
ok, G, R, cr = S.gr_fields(res)                            # SI: K/m, m/s, K/s
print(f"{ok.sum()} cells solidified")
for k, v in S.summary(res).items():
    if isinstance(v, dict) and "p50" in v:
        print(f"{k:16s} median {v['p50']:.3g}")
```

Output (from the test run):

```text
627 cells solidified
G_K_per_m        median 8.33e+06
R_m_per_s        median 0.0809
cooling_rate_K_per_s median 5.64e+05
G_over_R         median 1.04e+08
spacing_kurz_fisher_um median 1.76
```

G is highest at the bottom of the pool and R grows towards the top (R = v cos θ).

## Step 3: cell spacing and CET

```python
sc = S.SolidificationConstants(freezing_range=60.0, D_liquid=3e-9, gibbs_thomson=3.65e-7, partition=0.48)
for model in ("hunt", "kurz-fisher"):
    lam = S.primary_spacing(G[ok], R[ok], sc, model=model)
    print(f"{model:11s}: primary spacing median {np.median(lam):.2f} um")
print(f"equiaxed fraction (Hunt CET): {(S.hunt_cet(G[ok], R[ok], sc) == 2).mean():.3f}")
```

Output (from the test run):

```text
hunt       : primary spacing median 0.80 um
kurz-fisher: primary spacing median 1.76 um
equiaxed fraction (Hunt CET): 0.144
```

## Step 4: grains

```python
gr = S.grow_grains(res, substrate_grain_size=0.03, seed=0)       # epitaxy + competitive growth
print(S.grain_stats(gr, res, x_section=0.4))
n = S.export_exaca(res, "exaca_track.csv")                      # for the ExaCA cellular automaton
print(f"{n} cells exported for ExaCA")
```

Output (from the test run):

```text
{'grains_in_fusion_zone': 50, 'nucleated': 0, 'section_grains': 2, 'section_mean_width_um': 38.95128238975147, 'frac_100_build_15deg': 0.13078149920255183, 'frac_100_scan_15deg': 0.14354066985645933}
1254 cells exported for ExaCA
```

For realistic 3-D grains and texture use the cellular automaton (Tutorial 17); for the cells
themselves, the phase field (Tutorial 16).

## Next
[Tutorial 16: cells and segregation](16_phase_field.md).
