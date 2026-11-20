# Tutorial 22: EBSD analysis

**Goal.** Analyse an orientation map: segment grains, measure their shapes, and compute the
kernel average misorientation (KAM). A synthetic map stands in for a measured one, so the
tutorial runs anywhere. About 5 seconds.

**You will learn**
* orientations from Euler angles and cubic misorientation;
* grain segmentation and shapes;
* KAM, and how to read a real .ctf file.

## Step 1: a synthetic map

Four grains with a gradient of orientation (a small lattice curvature) and noise in one of them:

```python
import numpy as np
from tpmsfem.process import ebsd as E

ny, nx, step = 60, 80, 1.0                         # points, um
y, x = np.mgrid[0:ny, 0:nx]
eul = np.zeros((ny, nx, 3))
eul[..., 0] = np.where(x < 40, 0.0, np.radians(30.0))
eul[..., 1] = np.where(y < 30, 0.0, np.radians(25.0))
eul[..., 2] = np.where((x >= 40) & (y >= 30), np.radians(40.0) * x / nx, 0.0)  # bent grain: 0.5 deg per point
eul += np.random.default_rng(0).normal(0.0, np.radians(0.1), eul.shape)
g = E.euler_to_matrix(eul)
valid = np.ones((ny, nx), bool)
print(f"misorientation across the vertical boundary: {E.misorientation(g[10, 39], g[10, 40]):.1f} deg")
```

Output (from the test run):

```text
misorientation across the vertical boundary: 30.1 deg
```

## Step 2: grains and their shapes

```python
labels = E.segment_grains(g, valid, threshold=10.0, min_size=10)
shapes = E.grain_shapes(labels, (step, step))
print(f"{labels.max() + 1} grains")
for i in range(labels.max() + 1):
    print(f"grain {i}: area {shapes['area'][i]:6.0f} um^2, d_eq {shapes['d_eq'][i]:5.1f} um, "
          f"length/width {shapes['length'][i] / shapes['width'][i]:.2f}")
```

Output (from the test run):

```text
4 grains
grain 0: area   1200 um^2, d_eq  39.1 um, length/width 1.33
grain 1: area   1200 um^2, d_eq  39.1 um, length/width 1.33
grain 2: area   1200 um^2, d_eq  39.1 um, length/width 1.33
grain 3: area   1200 um^2, d_eq  39.1 um, length/width 1.33
```

## Step 3: kernel average misorientation

KAM highlights local lattice curvature (plastic strain, residual stress).

```python
k = E.kam(g, valid, max_angle=5.0)
for name, sel in (("flat grains", x < 40), ("bent grain", (x >= 40) & (y >= 30))):
    print(f"{name}: mean KAM {np.nanmean(k[sel]):.2f} deg")
```

Output (from the test run):

```text
flat grains: mean KAM 0.21 deg
bent grain: mean KAM 0.38 deg
```

## Step 4: a measured map

<!-- not run: needs a measured .ctf map, e.g. NIST AMB2018-02 (fetched by the benchmark script) -->
```python
m = E.read_ctf("High_resolution_Map_Data_3.ctf")
g = m.orientations()                               # (ny, nx, 3, 3)
labels = E.segment_grains(g, m.phase > 0, threshold=10.0, min_size=10)
print(labels.max() + 1, "grains")
```

`benchmarks/literature/nist_ambench2018_tracks_ebsd.py` compares predicted grains (Tutorial 17)
with the NIST map.

## Next
[Tutorial 23: the command line](23_command_line.md).
