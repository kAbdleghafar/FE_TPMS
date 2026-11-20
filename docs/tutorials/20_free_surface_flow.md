# Tutorial 20: Melt flow with a free surface, on powder

**Goal.** Rain a powder layer onto a plate, melt a track with the free-surface lattice
Boltzmann solver (Marangoni flow, recoil, evaporation), and check its conservation. A coarse
demonstration, about 2 minutes; production runs need 2.5–5 µm cells on a GPU.

**You will learn**
* the powder generator;
* `FreeSurfaceMeltPool`: inputs, time step, surface physics;
* the result: mass and energy checks, speed, surface.

Units: SI (m, s, W).

## Step 1: the powder bed

```python
import numpy as np
from tpmsfem.fem.materials import Material, ThermalProperties
from tpmsfem.process import Evaporation, Laser, ScanVector
from tpmsfem.process.freesurface import FreeSurfaceMeltPool, SahooSurfaceTension
from tpmsfem.process.powder import rain_powder, voxelise

dx = 10e-6
nx, ny, nz = 40, 20, 14
metal = np.zeros((nx, ny, nz))
metal[1:-1, 1:-1, 1:7] = 1.0                                  # 60 um plate
bed = rain_powder(extent=(dx, (nx - 1) * dx, dx, (ny - 1) * dx), floor=7 * dx, layer=40e-6, seed=0)
print(f"{len(bed)} particles, packing {bed.packing_density:.2f}, median diameter {2e6 * np.median(bed.radii):.0f} um")
metal = np.maximum(metal, voxelise(bed, dx, metal.shape))     # fill levels in [0, 1]
```

Output (from the test run):

```text
146 particles, packing 0.42, median diameter 22 um
```

## Step 2: melt a track

```python
th = ThermalProperties.from_si([20.0, 1290.0, 1350.0, 3000.0], [9.8, 26.0, 30.0, 30.0], [410.0, 650.0, 720.0, 720.0],
                               8440.0, 2.9e5, 1290.0, 1350.0)
IN625 = Material("IN625", E=208e3, nu=0.28, thermal=th)
fs = FreeSurfaceMeltPool(IN625, dx, metal, dt=2e-7, viscosity=5e-3, sigma=1.8,
                         dsigma_dT=SahooSurfaceTension(a_s=0.01, A=3.8e-4), evaporation=Evaporation(),
                         energy_transport="lbm")
print(f"relaxation time {fs.tau:.4f} (liquid metal: close to 0.5)")
out = fs.run(Laser(180.0, 0.35, 50e-6, depth=5e-6), [ScanVector(80e-6, ny * dx / 2, 300e-6, ny * dx / 2, 0.8)],
             t_extra=2e-5)
print(f"u_max {out.u_max:.2f} m/s, peak {out.T_peak.max():.0f} C, mass error {out.mass_error:.1e}, "
      f"energy residual {out.energy_residual:.1%}")
```

Output (from the test run):

```text
relaxation time 0.5036 (liquid metal: close to 0.5)
u_max 4.19 m/s, peak 3519 C, mass error -1.4e-13, energy residual 2.0%
```

* The time step sets the lattice: keep the lattice Mach number u·dt/dx below ~0.1.
* `energy_transport="lbm"` moves the heat exactly with the mass; what the bounded transport
  cannot place is booked as `energy_residual`: a few percent on coarse powder runs like this
  one (many nearly empty surface cells), well below 1 % on a bare plate.
* `SahooSurfaceTension` gives a temperature-dependent dσ/dT with a surface-active element:
  inward flow when cool, outward when hot.

## Step 3: look at the surface

```python
s = out.surface                                    # surface height per column [m]
print(f"surface after the track: {1e6 * s[5:-5, 5:-5].min():.0f} .. {1e6 * s[5:-5, 5:-5].max():.0f} um above the domain floor")
```

Output (from the test run):

```text
surface after the track: 80 .. 104 um above the domain floor
```

The scripts `tools/lbm/freesurface_caseB.py` (AM-Bench), `freesurface_powder_track.py` and
`freesurface_animation.py` run production cases; status and limits in
[freesurface_lbm.md](../freesurface_lbm.md).

## Next
[Tutorial 21: keyholes](21_keyholes.md).
