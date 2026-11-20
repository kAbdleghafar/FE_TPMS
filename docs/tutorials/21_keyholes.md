# Tutorial 21: Keyholes: ray tracing, bubbles, closure and break-through

**Goal.** Work with the keyhole physics of the flow solver: trace the laser in a keyhole with
Fresnel absorption, follow bubbles and pores, run a tiny keyhole melt pool with diagnostics,
and use the tools of the closed-keyhole studies (ramps, turns, penetration through thin walls).
About 2 minutes.

**You will learn**
* `RayTracer`, `fresnel_absorptivity`;
* `BubbleModel`: pressures, frozen pores, open sides;
* `keyhole_metrics`, `open_gas_reaches`;
* the numerics a keyhole needs (time step, sub-grid viscosity);
* what the studies have found so far.

Units: SI. Cell coordinates where stated.

## Step 1: Fresnel absorption

```python
import numpy as np
from tpmsfem.process.keyhole import fresnel_absorptivity, index_from_absorptivity

n, k = index_from_absorptivity(0.26)               # flat-melt absorptivity of Ti-6Al-4V, k = n
for deg in (0, 45, 70, 83, 89):
    print(f"incidence {deg:2d} deg: absorbed {fresnel_absorptivity(np.cos(np.radians(deg)), n, k):.3f}")
```

Output (from the test run):

```text
incidence  0 deg: absorbed 0.260
incidence 45 deg: absorbed 0.269
incidence 70 deg: absorbed 0.337
incidence 83 deg: absorbed 0.431
incidence 89 deg: absorbed 0.187
```

Grazing light on a steep keyhole wall is absorbed more (up to 0.43 near 83°), and what is
reflected hits the wall again.

## Step 2: trace a beam in a keyhole

```python
from tpmsfem.process.keyhole import RayTracer, smooth_fill

N, top, R = 50, 60.0, 10.0                         # cells
X, Y, Z = np.meshgrid(*(np.arange(m) + 0.5 for m in (N, N, 80)), indexing="ij")
rt = RayTracer(A0=0.26)
for depth in (0.0, 15.0, 30.0, 45.0):
    fill = (Z < top - depth * np.clip(1 - ((X - N / 2) ** 2 + (Y - N / 2) ** 2) / R ** 2, 0, None)).astype(float)
    fill[[0, -1]] = fill[:, [0, -1]] = fill[:, :, [0, -1]] = 0.0
    dep, st = rt.trace(smooth_fill(fill), fill > 0, N / 2, N / 2, power=1.0, radius=4.0)
    print(f"keyhole depth / rim radius {depth / R:3.1f}: absorbed {st['absorbed']:.2f}, {st['mean_bounces']:.1f} bounces")
```

Output (from the test run):

```text
keyhole depth / rim radius 0.0: absorbed 0.26, 1.0 bounces
keyhole depth / rim radius 1.5: absorbed 0.45, 1.9 bounces
keyhole depth / rim radius 3.0: absorbed 0.52, 2.2 bounces
keyhole depth / rim radius 4.5: absorbed 0.63, 2.9 bounces
```

This is the rise of absorptivity with keyhole depth that Gan et al.'s Eq. 6 expresses
(0.26 → ~0.6).

## Step 3: bubbles and pores

Every enclosed gas region gets an ideal-gas pressure; a bubble frozen in after its walls were
molten is a `pore`.

```python
from tpmsfem.process.freesurface import GAS, INTERFACE, LIQUID, WALL
from tpmsfem.process.keyhole import BubbleModel, open_gas_reaches

def walls_and_interface(flag):
    gas = flag == GAS
    nb = np.zeros_like(gas)
    for ax in range(3):
        for d in (-1, 1):
            nb |= np.roll(gas, d, ax)
    flag = flag.copy()
    flag[(flag == LIQUID) & nb] = INTERFACE
    flag[[0, -1]] = WALL
    flag[:, [0, -1]] = WALL
    flag[:, :, [0, -1]] = WALL
    return flag

n = 24
flag = np.full((n, n, n), GAS, np.int8)
flag[:, :, :16] = LIQUID                           # melt with gas above
g = np.arange(n) + 0.5
Xg, Yg, Zg = np.meshgrid(g, g, g, indexing="ij")
flag[(Xg - 12) ** 2 + (Yg - 12) ** 2 + (Zg - 9) ** 2 < 16] = GAS     # a bubble in the melt
flag = walls_and_interface(flag)
fill = np.where(flag == LIQUID, 1.0, 0.0)
bm = BubbleModel()
bm.update(flag, fill, T=np.full(flag.shape, 2000.0), T_solidus=1600.0)     # molten around it
bm.update(flag, fill, T=np.full(flag.shape, 1000.0), T_solidus=1600.0)     # then frozen
b = bm.bubbles[0]
print(f"bubble of {b['volume']:.0f} cells at {np.round(b['centroid'], 1)}, frozen {b['frozen']}, pore {b['pore']}")
print("open gas reaches below z = 10:", open_gas_reaches(flag, 10))
```

Output (from the test run):

```text
bubble of 480 cells at [12. 12.  9.], frozen True, pore True
open gas reaches below z = 10: (False, 0)
```

## Step 4: a tiny keyhole melt pool

The full machinery on a small plate: rays, bubbles, sub-grid viscosity, diagnostics every
100 steps.

```python
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, ScanVector
from tpmsfem.process.freesurface import FreeSurfaceMeltPool
from tpmsfem.process.keyhole import keyhole_metrics

th = ThermalProperties.from_si([0.0, 20000.0], [30.0, 30.0], [800.0, 800.0], 4000.0, 2.9e5, 1600.0, 1660.0)
h = 5e-6
metal = np.zeros((30, 20, 24), bool)
metal[1:-1, 1:-1, 1:15] = True                     # plate top at z = 15 cells
md = FreeSurfaceMeltPool(Material("Ti64-like", 110e3, 0.3, thermal=th), h, metal, dt=h / 250.0, viscosity=3e-3,
                         sigma=1.6, T0=25.0, T_max=8000.0, dsigma_dT=-2.6e-4,
                         evaporation=Evaporation(T_boil=3290.0, L_v=9.3e6, M=0.048), energy_transport="lbm")
md.rays, md.ray_every = RayTracer(A0=0.26), 5
md.bubbles = BubbleModel()
md.lbm.smagorinsky = 0.15                          # needed for keyholes: cell Reynolds number >> 10

def report(m, t, x, y):
    k = keyhole_metrics(m, x, y, 25e-6, 15 * h)
    print(f"t {t * 1e6:5.1f} us: depth {1e6 * k['depth']:5.1f} um, absorbed {m.ray_stats['absorbed']:.2f}, bubbles {k['n_bubbles']}")

res = md.run(Laser(300.0, 0.0, 25e-6), [ScanVector(h + 50e-6, 10 * h, h + 90e-6, 10 * h, 2.0)],
             callback=report, callback_every=100)
print(f"mass error {res.mass_error:.0e}, energy residual {res.energy_residual:.2%}")
```

Output (from the test run):

```text
t   2.0 us: depth   5.1 um, absorbed 0.26, bubbles 0
t   4.0 us: depth   9.0 um, absorbed 0.28, bubbles 0
t   6.0 us: depth  12.5 um, absorbed 0.29, bubbles 0
t   8.0 us: depth  15.6 um, absorbed 0.36, bubbles 0
t  10.0 us: depth  20.7 um, absorbed 0.40, bubbles 0
t  12.0 us: depth  29.8 um, absorbed 0.47, bubbles 0
t  14.0 us: depth  34.2 um, absorbed 0.46, bubbles 0
t  16.0 us: depth  40.2 um, absorbed 0.54, bubbles 0
t  18.0 us: depth  44.4 um, absorbed 0.55, bubbles 0
t  20.0 us: depth  48.6 um, absorbed 0.55, bubbles 0
mass error -1e-13, energy residual 1.25%
```

For production runs, use 2.5 µm cells on a GPU, a longer track, `save_state`/`load_state`
checkpoints, and the scripts below.

## Step 5: the closed-keyhole tools

```python
from tpmsfem.process.scanpath import ramp_down, turn_point

off = ramp_down(0.0, 0.0, 300e-6, 0.0, speed=0.6, power=260.0, ramp=100e-6, n=4)
print("end-of-track ramp:", [(round(1e6 * v.x0), round(v.power)) for v in off])
turn, t0 = turn_point(1e-3, 0.0, 1.0, 50e-6, 1.11e3, approach=100e-6, dt_seg=10e-6)
print(f"Martin-type turn: {len(turn)} segments, {-2 * t0 * 1e3:.2f} ms")
```

Output (from the test run):

```text
end-of-track ramp: [(0, 260), (200, 228), (225, 162), (250, 98), (275, 32)]
Martin-type turn: 200 segments, 2.00 ms
```

| script | what it runs |
|---|---|
| `tools/lbm/keyhole_ti64.py T3` | a steady Ti-6Al-4V X-ray case; `--scenario ramp|turn|turn-ke` for closures |
| `tools/lbm/keyhole_ti64.py M200 --scenario turn` | a turn point of Martin et al. (2019) |
| `tools/lbm/keyhole_overhang.py --slab 90e-6` | a keyhole in a thin slab over powder |
| `tools/lbm/run_keyhole_gpu.sh`, `run_closed_keyhole_gpu.sh` | the GPU batches |
| `benchmarks/literature/keyhole_penetration.py` | where the keyhole breaks through the gyroid's walls |

## Results so far

* **Solver checks.** A flat plate absorbs exactly A0; a paraboloid keyhole 0.30 → 0.52 → 0.62
  with depth; mass conserved to 1e-12. Liquid metal keyholes need dt = dx/(250 m/s), one-way
  interface exchange and a sub-grid viscosity (Cs 0.15), otherwise the bulk melt runs away.
* **CPU check, Ti-6Al-4V 260 W, 0.6 m/s, 5 µm.** A stable keyhole deepens to 100 µm in 120 µs;
  the absorbed fraction rises 0.26 → 0.61 (Gan et al.: 0.57).
* **Closure after turn-off.** That 100 µm keyhole refills in ~30 µs (capillary time ~15 µs),
  leaving a 7 µm crater and no pore: a keyhole stopped at half its steady depth closes without
  trapping gas.
* **Break-through, CoCrFeMnNi 90 µm slab over powder.** The keyhole pierces the slab at 92 µs
  and stays open, flickering; 240,000 µm³ of powder melts 60 µm deep under a 100 µm track.
* **Gyroid map.** At the study settings the keyhole reaches through the solid into the powder
  under 23 % of every layer (32 % at its deepest), 37 % in 0.2–0.3 mm walls, against 5 % for a
  conduction pool. A gyroid layer has 28–44 track ends per mm², 5–8× a solid block.
* **Pending on the GPU.** The six X-ray cases at 2.5 µm, Martin's turn points with and without
  the constant-Ke power schedule, and the slabs of 90–250 µm.

Details: [keyhole.md](../keyhole.md).

## Next
[Tutorial 22: EBSD analysis](22_ebsd.md).
