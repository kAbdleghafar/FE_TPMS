# Tutorial 16: Cells and segregation with the phase field

**Goal.** Grow solidification cells of IN625 at melt-pool conditions with the quantitative
phase field in 2-D, measure spacing and segregation, then a small 3-D multi-element run and
the EDS maps a microscope would record. About 3 minutes.

**You will learn**
* `DiluteAlloy`, solute trapping, `DirectionalPF`;
* spacing, tips and segregation;
* `DirectionalPF3D` with several solutes, and synthetic SEM/STEM EDS.

Units: SI (m, s, K).

## Step 1: the alloy and solute trapping

```python
import numpy as np
from tpmsfem.process import phasefield as PF

alloy = PF.IN625_NB                                   # IN625 as Ni-4 wt% Nb (Ghosh et al.)
for V in (0.01, 0.05, 0.2):
    a = alloy.at_velocity(V, V_D=1.0)                 # continuous growth model
    print(f"V = {V:4.2f} m/s: k {a.k:.3f}, freezing range {a.freezing_range:.1f} K")
print(f"constitutional velocity at G = 1e7 K/m: {PF.constitutional_velocity(alloy, 1e7):.2e} m/s")
```

Output (from the test run):

```text
V = 0.01 m/s: k 0.485, freezing range 44.6 K
V = 0.05 m/s: k 0.505, freezing range 41.3 K
V = 0.20 m/s: k 0.567, freezing range 32.6 K
constitutional velocity at G = 1e7 K/m: 6.59e-04 m/s
```

At LPBF speeds the partition coefficient rises towards 1 (trapping) and the freezing range
shrinks.

## Step 2: grow cells (2-D)

```python
fast = alloy.at_velocity(0.05, V_D=1.0)
pf = PF.DirectionalPF(fast, G=1e7, V=0.05, width_m=1.2e-6, height_m=0.6e-6, W0_m=10e-9, seed=0)
r = pf.run(1.0e-6 / 0.05, adapt_dt=True)              # 1 um of growth
lam, n = r.spacing()
print(f"{n:.0f} cells, spacing {lam * 1e6:.2f} um")
print(r.segregation())
```

Output (from the test run):

```text
4 cells, spacing 0.34 um
{'min': 0.7823811288726458, 'p50': 0.937125448791764, 'p95': 1.7150992026097107, 'p99': 1.946232198580448, 'max': 3.0040215168840967}
```

Choose the interface width W0 so that D/V ≥ 5–10 W0, and check by halving it; the cost scales
as W0⁻⁵. `adapt_dt=True` is 1.7× faster.

## Step 3: 3-D, several elements, EDS

```python
from tpmsfem.post import eds
from tpmsfem.process import phasefield3d as P3

a3 = P3.IN625_AMBENCH.at_velocity(0.092, V_D=1.0)              # Nb, Mo, Cr, Fe
print(a3.names, f"freezing range {a3.freezing_range:.1f} K")
pf3 = P3.DirectionalPF3D(a3, G=6.6e6, V=0.092, size_m=(0.4e-6, 0.4e-6), height_m=0.4e-6, W0_m=10e-9, seed=0)
r3 = pf3.run(0.6e-6 / 0.092, adapt_dt=True)
tip = r3.tip_row()
fields = {e: r3.grown(e) for e in r3.names + [r3.balance]}     # wt%, (z, y, x)
sem = eds.sem_eds(fields, r3.dx_m, index=max(tip - 20, 1), beam_kV=5.0)
stem = eds.stem_eds(fields, r3.dx_m, index=max(tip - 20, 1), foil_m=100e-9)
for e in ("Nb", "Mo"):
    print(f"{e}: SEM 5 kV resolution {sem.resolution_m[e] * 1e9:.0f} nm")
```

Output (from the test run):

```text
['Nb', 'Mo', 'Cr', 'Fe'] freezing range 40.0 K
Nb: SEM 5 kV resolution 81 nm
Mo: SEM 5 kV resolution 77 nm
```

SEM-EDS at 15 kV keeps only ~27 % of the Nb contrast of 0.3 µm cells; 5 kV keeps 85 %, STEM
96 % (study: [process_models.md](../process_models.md) §5.6).

## Next
[Tutorial 17: grains and texture](17_grain_ca.md).
