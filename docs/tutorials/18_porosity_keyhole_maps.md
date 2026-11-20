# Tutorial 18: Porosity maps: lack of fusion and keyholes

**Goal.** Decide, for a set of process parameters, whether a part will be dense, show
lack-of-fusion pores or keyhole pores, using melt-pool geometry and the keyhole scaling laws of
Gan et al. (2021). Build a power–speed map. About 10 seconds.

**You will learn**
* the lack-of-fusion criterion and unfused fraction;
* the keyhole number Ke, the process absorptivity, keyhole depth, fluctuation, regime and
  stability;
* the conventions the thresholds need;
* what the validation and the studies found.

## Step 1: lack of fusion

Neighbouring tracks and layers must overlap: full fusion when (h/W)² + (L/D)² ≤ 1 (Tang et al.
2017), with hatch h, layer L, melt-pool width W and depth D from the top of the powder.

```python
import numpy as np
from tpmsfem.process.defects import lof_criterion, lof_porosity, volumetric_energy_density

W, D, layer = 97.0, 54.0, 30.0                       # um, from a melt-pool run with powder (Tutorial 13)
for hatch in (70.0, 85.0, 100.0):
    print(f"hatch {hatch:.0f} um: criterion {lof_criterion(W, D, hatch, layer):.2f}, "
          f"unfused {100 * lof_porosity(W, D, hatch, layer, cv=0.1, samples=4):.2f} %")
print(f"VED = {volumetric_energy_density(200.0, 0.75, 85e-6, 30e-6) / 1e9:.0f} J/mm^3  (not a good predictor alone)")
```

Output (from the test run):

```text
hatch 70 um: criterion 0.83, unfused 0.00 %
hatch 85 um: criterion 1.08, unfused 0.86 %
hatch 100 um: criterion 1.37, unfused 8.28 %
VED = 105 J/mm^3  (not a good predictor alone)
```

## Step 2: the keyhole laws

Ke = η P / ((T_l − T₀) π ρ c_p √(α v r₀³)). With liquid properties at the liquidus and the
process absorptivity η (which rises from the flat-melt η_m to ~0.7 as the keyhole traps light):

* keyhole depth e = r₀ · 0.4 (Ke − 1.4), its fluctuation Δe* = 0.36 Ke^0.86;
* conduction ≤ 1.4 < transition ≤ 6 < keyhole;
* stable ≤ 16 < intermediate ≤ 30 < chaotic (pores and spatter).

```python
from tpmsfem.process.defects import keyhole_state

ti64 = dict(eta_min=0.26, rho=3920.0, cp=830.0, alpha=33.4 / (3920.0 * 830.0), T_l=1660.0, T0=30.0)  # Gan et al.
st = keyhole_state(260.2, 0.6, 47.5e-6, **ti64)                # an X-ray case: measured depth 169 um
print(f"Ti-6Al-4V 260 W, 0.6 m/s: eta {float(st['eta']):.2f}, Ke {float(st['Ke']):.1f}, depth {1e6 * float(st['depth']):.0f} um "
      f"({1e6 * float(st['depth_min']):.0f}-{1e6 * float(st['depth_max']):.0f}), {st['regime']}, {st['stability']}")
```

Output (from the test run):

```text
Ti-6Al-4V 260 W, 0.6 m/s: eta 0.58, Ke 11.2, depth 187 um (118-255), keyhole, stable
```

## Step 3: a power–speed map

```python
cantor = dict(eta_min=0.33, rho=7980.0, cp=642.0, alpha=26.9 / (7980.0 * 642.0), T_l=1340.0, T0=80.0)
P, v = np.meshgrid([150.0, 200.0, 250.0, 300.0], [0.5, 0.75, 1.0, 1.5], indexing="ij")
st = keyhole_state(P, v, 35e-6, **cantor)
print("Ke (rows: power 150-300 W; columns: speed 0.5-1.5 m/s)")
print(np.round(st["Ke"], 1))
print(st["stability"])
```

Output (from the test run):

```text
Ke (rows: power 150-300 W; columns: speed 0.5-1.5 m/s)
[[14.  10.   7.6  4.9]
 [19.7 14.8 11.6  7.8]
 [25.2 19.5 15.7 10.9]
 [30.6 24.2 19.8 14.2]]
[['stable' 'stable' 'stable' 'stable']
 ['intermediate' 'stable' 'stable' 'stable']
 ['intermediate' 'intermediate' 'stable' 'stable']
 ['chaotic' 'intermediate' 'intermediate' 'stable']]
```

## Step 4: keep Ke constant while slowing down

At turn points the beam decelerates and Ke rises. The power that keeps Ke constant:

```python
from tpmsfem.process.defects import keyhole_power

Ke0 = float(keyhole_state(200.0, 1.0, 25e-6, **ti64)["Ke"])
for vv in (1.0, 0.5, 0.2, 0.1):
    print(f"v = {vv:.1f} m/s: {keyhole_power(Ke0, vv, 25e-6, **ti64):5.1f} W keeps Ke = {Ke0:.1f}")
```

Output (from the test run):

```text
v = 1.0 m/s: 200.0 W keeps Ke = 19.9
v = 0.5 m/s: 136.6 W keeps Ke = 19.9
v = 0.2 m/s:  85.1 W keeps Ke = 19.9
v = 0.1 m/s:  60.1 W keeps Ke = 19.9
```

## Results so far

* **Validation (Gan et al. X-ray data, 71 Ti-6Al-4V tracks).** From power, speed and spot
  alone the depths (15–445 µm) come out with RMSE 17 µm, R² 0.98; the law fitted to Al6061 +
  SS316 only predicts Ti-6Al-4V as well.
* **IN625 (Locatelli et al.).** The Ke 16 pore threshold reproduces all 10 reported outcomes
  for a spot radius of 40–46 µm.
* **CoCrFeMnNi gyroid (200 W, 0.75 m/s, r₀ 35 µm).** Keyhole mode, about 190 µm deep (Ke 14.8);
  heat accumulation passes Ke 16 near 155 °C; a 45 µm spot keeps it stable.

Full maps: `benchmarks/literature/porosity_maps.py`, `keyhole_maps.py`;
[porosity_maps.md](../porosity_maps.md), [keyhole.md](../keyhole.md).

## Next
[Tutorial 19: surface roughness](19_roughness.md).
