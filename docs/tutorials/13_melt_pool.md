# Tutorial 13: The laser melt pool (conduction model)

**Goal.** Simulate a single laser track on an IN625 plate, measure the melt pool, run a power
sweep, add a powder layer, and see how calibration works. About 2 minutes.

**You will learn**
* `MeltPoolModel`: grid, laser, scan vectors, evaporation;
* fusion-zone width and depth, pool length, peak temperature;
* where the conduction model stops being valid (keyholes).

Units: mm, s, W, °C.

## Step 1: material and one track

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector, raster
from tpmsfem.process.meltpool import fusion_zone, pool_length

th = ThermalProperties.from_si(T=[20.0, 1290.0, 1350.0, 3000.0], k_W_mK=[9.8, 26.0, 30.0, 30.0],
                               cp_J_kgK=[410.0, 650.0, 720.0, 720.0], rho_kg_m3=8440.0, latent_J_kg=2.9e5,
                               T_solidus=1290.0, T_liquidus=1350.0)
IN625 = Material("IN625", E=208e3, nu=0.28, thermal=th)

def track(P, v=800.0, powder=0.0, h=0.01):
    mp = MeltPoolModel(IN625, h=h, length=0.6, width=0.15, depth=0.08, T0=25.0, emissivity=0.4,
                       evaporation=Evaporation(), powder=powder)       # half-symmetric about y = 0
    res = mp.run(Laser(power=P, absorptivity=0.35, radius=0.05, depth=0.015),
                 [ScanVector(0.1, 0.0, 0.5, 0.0, speed=v)], t_extra=2e-4)
    return mp, res

mp, res = track(180.0)
w, d = fusion_zone(res, x_section=0.4)
print(f"180 W, 0.8 m/s: width {1e3 * w:.0f} um, depth {1e3 * d:.0f} um, peak {res.T_peak.max():.0f} C")
```

Output (from the test run):

```text
180 W, 0.8 m/s: width 113 um, depth 50 um, peak 3639 C
```

* `Laser(power, absorptivity, radius, depth)`: Gaussian, 1/e² radius in mm, an absorption
  depth that mimics a shallow vapour depression.
* `Evaporation()` caps the surface near the boiling point (nickel by default).

## Step 2: a power sweep

```python
for P in (120.0, 180.0, 240.0):
    _, r = track(P)
    w, d = fusion_zone(r, x_section=0.4)
    print(f"{P:.0f} W: width {1e3 * w:4.0f} um, depth {1e3 * d:3.0f} um, D/W {d / w:.2f}")
```

Output (from the test run):

```text
120 W: width   95 um, depth  38 um, D/W 0.40
180 W: width  113 um, depth  50 um, D/W 0.44
240 W: width  123 um, depth  62 um, D/W 0.51
```

When the depth approaches half the width and more, the real pool is keyhole-shaped: use the
keyhole laws (Tutorial 18) or the flow model (Tutorials 20–21) there.

## Step 3: a powder layer and hatches

```python
mp2, r2 = track(180.0, powder=0.03)
w2, d2 = fusion_zone(r2, x_section=0.4)
print(f"on a 30 um powder layer: width {1e3 * w2:.0f} um, depth into the plate {1e3 * d2:.0f} um")
vecs = raster(0.1, 0.5, 0.0, 0.2, hatch=0.08, speed=800.0)
print(f"raster: {len(vecs)} vectors; first {vecs[0]}")
```

Output (from the test run):

```text
on a 30 um powder layer: width 114 um, depth into the plate 17 um
raster: 3 vectors; first ScanVector(x0=0.1, y0=np.float64(0.0), x1=0.5, y1=np.float64(0.0), speed=800.0, power=None)
```

For several tracks use `symmetric_y=False` and a wider domain; `set_part_geometry(field)` makes
the substrate a TPMS lattice built so far (Tutorial 14), `set_temperature(T)` adds a preheat.

## Step 4: calibration

The absorptivity and absorption depth are the uncertain inputs. Fit them to one measured
cross-section, then predict the others: NIST AM-Bench 2018 case B in
`benchmarks/literature/nist_ambench2018_tracks.py` (widths of A and C within 5 %).

## Next
[Tutorial 14: scan paths for TPMS layers](14_scan_paths.md).
