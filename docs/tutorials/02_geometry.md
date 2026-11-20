# Tutorial 2: Designing TPMS lattices and implants

**Goal.** Build TPMS geometries: uniform and graded lattices, walls of a prescribed thickness,
combinations with solid parts, and parametric implants. About 10 seconds.

**You will learn**
* that geometry is an implicit field, negative inside the solid;
* the six surfaces, sheet vs network, density vs iso-level;
* grading and booleans.

## Step 1: the available surfaces

```python
import numpy as np
from tpmsfem.geometry import SURFACES, TPMS, box, cylinder, level_for_density, relative_density

print(sorted(SURFACES))
```

Output (from the test run):

```text
['diamond', 'fks', 'gyroid', 'iwp', 'neovius', 'primitive', 'schwarz_p']
```

Each surface comes as a **sheet** (the thickened surface: two interpenetrating channels) or a
**network** (one solid phase: strut-like).

## Step 2: density and iso-level

The relative density (solid volume fraction) is set through an iso-level of the surface
equation. The library converts between them:

```python
for kind in ("sheet", "network"):
    for rho in (0.2, 0.3, 0.4):
        c = level_for_density("gyroid", kind, rho)
        print(f"gyroid {kind:7s} density {rho:.1f}: level {c:+.3f} -> back to {relative_density('gyroid', kind, c):.3f}")
```

Output (from the test run):

```text
gyroid sheet   density 0.2: level +0.309 -> back to 0.200
gyroid sheet   density 0.3: level +0.458 -> back to 0.300
gyroid sheet   density 0.4: level +0.616 -> back to 0.400
gyroid network density 0.2: level -0.910 -> back to 0.200
gyroid network density 0.3: level -0.616 -> back to 0.300
gyroid network density 0.4: level -0.310 -> back to 0.400
```

## Step 3: a field is a function

A `Field` is evaluated at points; negative means solid. That is all the mesher needs.

```python
gyr = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.30)
pts = np.random.default_rng(0).uniform(0, 4, size=(200_000, 3))
inside = gyr(pts[:, 0], pts[:, 1], pts[:, 2]) < 0
print(f"Monte Carlo density {inside.mean():.3f} (target 0.30), nominal {gyr.nominal_relative_density():.3f}")
```

Output (from the test run):

```text
Monte Carlo density 0.293 (target 0.30), nominal 0.300
```

## Step 4: grading and wall thickness

A callable density grades the lattice. For a prescribed wall *thickness* use
`sheet_with_thickness`, which is accurate to a few percent (scaling the level misses thin
walls by 10–20 %).

```python
from tpmsfem.geometry.tpms import sheet_with_thickness

graded = TPMS("schwarz_p", cell_size=4.0, kind="sheet",
              density=lambda x, y, z: 0.2 + 0.2 * np.clip(z / 20.0, 0.0, 1.0))   # 20 % -> 40 % up z
walls = sheet_with_thickness("gyroid", 4.0, thickness=lambda x, y, z: 0.2 + 0.1 * x)  # 0.2 mm + 0.1 mm per mm
for zc in (2.0, 10.0, 18.0):
    p = np.random.default_rng(1).uniform(0, 4, size=(100_000, 3)) + [0, 0, zc - 2]
    print(f"z ~ {zc:4.1f} mm: local density {(graded(p[:, 0], p[:, 1], p[:, 2]) < 0).mean():.3f}")
```

Output (from the test run):

```text
z ~  2.0 mm: local density 0.223
z ~ 10.0 mm: local density 0.303
z ~ 18.0 mm: local density 0.382
```

## Step 5: combine with solids

Booleans: `|` union, `&` intersection, `-` difference.

```python
part = (cylinder((0, 0, 0), (0, 0, 20), 8.0) & gyr) | box((-8, -8, 0), (8, 8, 1.0))   # lattice on a base
pts = np.random.default_rng(2).uniform([-8, -8, 0], [8, 8, 20], size=(200_000, 3))
print(f"solid fraction of the bounding box: {(part(pts[:, 0], pts[:, 1], pts[:, 2]) < 0).mean():.3f}")
```

Output (from the test run):

```text
solid fraction of the bounding box: 0.270
```

## Step 6: implants

```python
from tpmsfem.geometry import AcetabularCup, FemoralStem

stem = FemoralStem(design="lattice_core", lattice=TPMS("gyroid", cell_size=6.0, kind="sheet", density=0.5))
lo, hi = stem.bounds()
print("stem bounds [mm]:", np.round(lo, 1), np.round(hi, 1), "head centre", np.round(stem.head_center, 1))
cup = AcetabularCup(porous_thickness=2.0, lattice=TPMS("diamond", cell_size=2.0, kind="network", density=0.35))
print("cup bounds [mm]:", [np.round(b, 1) for b in cup.bounds()])
```

Output (from the test run):

```text
stem bounds [mm]: [-39.7 -15.   -1. ] [ 11.   15.  154.7] head centre [-24.7   0.  139.7]
cup bounds [mm]: [array([-27.5, -27.5,   0. ]), array([27.5, 27.5, 27.5])]
```

## Check yourself
* The Monte Carlo density matches the target within about 0.01.
* The graded Schwarz-P gets denser with height.

## Exercises
1. Compare the density accuracy of `TPMS(..., density=...)` and `sheet_with_thickness` for a
   0.15 mm wall.
2. Rotate a lattice with `TPMS(..., rotation=rotation_matrix((1, 0, 0), 45))`.

## Next
[Tutorial 3: meshing](03_meshing.md).
