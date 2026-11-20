# Tutorial 14: Scan paths for a TPMS layer

**Goal.** Slice a graded gyroid at one build height, generate the contour and hatch vectors,
see how short lattice vectors are, and plot the layer. About 10 seconds.

**You will learn**
* `slice_scan_path` and its options;
* why TPMS layers have many more track ends than solid parts;
* timed paths: ramps and turn points.

## Step 1: slice and scan one layer

```python
import numpy as np
from tpmsfem.geometry.tpms import sheet_with_thickness
from tpmsfem.process.scanpath import slice_mask, slice_scan_path

gyroid = sheet_with_thickness("gyroid", 4.0, lambda x, y, z: 0.2 + 0.1 * x)   # walls 0.2 -> 0.6 mm along x
path, stats = slice_scan_path(gyroid, z=1.3, bounds=(0.0, 4.0, 0.0, 4.0), hatch=0.085, speed=750.0,
                              contour_offset=0.035, angle_deg=67.0)
print(stats)
```

Output (from the test run):

```text
{'contours': 5, 'contour_length_mm': 19.530271077495428, 'hatch_vectors': 68, 'hatch_length_mm': 34.91, 'scan_time_s': 0.08295018327612243}
```

The laser jumps between vectors with the power off (`power=0` vectors); contours are resampled
into short segments.

## Step 2: how short are lattice vectors?

```python
from tpmsfem.geometry import box

x = np.arange(0.0025, 4.0, 0.005)
area = slice_mask(gyroid, 1.3, x, x).sum() * 0.005 ** 2
print(f"gyroid layer: {stats['hatch_vectors']} hatch vectors, mean length "
      f"{stats['hatch_length_mm'] / stats['hatch_vectors']:.2f} mm, {2 * stats['hatch_vectors'] / area:.0f} track ends per mm^2")
_, sb = slice_scan_path(box((0.01, 0.01, 0), (3.99, 3.99, 5)), z=1.3, bounds=(0.0, 4.0, 0.0, 4.0), hatch=0.085,
                        speed=750.0, contour_offset=0.035)
print(f"solid block:  {sb['hatch_vectors']} hatch vectors, mean length {sb['hatch_length_mm'] / sb['hatch_vectors']:.2f} mm, "
      f"{2 * sb['hatch_vectors'] / 15.9:.0f} track ends per mm^2")
```

Output (from the test run):

```text
gyroid layer: 68 hatch vectors, mean length 0.51 mm, 28 track ends per mm^2
solid block:  45 hatch vectors, mean length 3.82 mm, 6 track ends per mm^2
```

Every track end is a place where the keyhole can collapse and leave a pore (Tutorial 21).

## Step 3: plot the layer

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
for v in path:
    ax.plot([v.x0, v.x1], [v.y0, v.y1], lw=0.5, color="0.8" if v.power == 0.0 else "C0")
ax.set_aspect("equal")
fig.savefig("gyroid_layer_scan.png", dpi=100)
```

## Step 4: timed paths, ramps and turns

For the flow model, paths can carry time-varying speed and power (SI units):

```python
from tpmsfem.process.scanpath import ramp_down, turn_point

ramp = ramp_down(0.0, 0.0, 400e-6, 0.0, speed=0.6, power=260.0, ramp=100e-6, n=5)
print("ramp-down powers [W]:", [round(v.power) for v in ramp])
turn, t0 = turn_point(1e-3, 0.0, speed=1.0, hatch=50e-6, accel=1.11e3, approach=100e-6, dt_seg=5e-6)
print(f"galvo turn: {len(turn)} segments over {-2 * t0 * 1e6:.0f} us, slowest {min(v.speed for v in turn):.2f} m/s")
```

Output (from the test run):

```text
ramp-down powers [W]: [260, 234, 182, 130, 78, 26]
galvo turn: 400 segments over 2002 us, slowest 0.13 m/s
```

## Next
[Tutorial 15: solidification conditions](15_solidification.md).
