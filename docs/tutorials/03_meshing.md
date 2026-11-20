# Tutorial 3: Meshing a lattice

**Goal.** Turn a TPMS field into meshes for analysis: voxel hex8 meshes (the default),
surface triangulations (STL) and, optionally, tetrahedra; choose the resolution and check it.
About 30 seconds.

**You will learn**
* `voxelize` and its options;
* how many voxels a wall needs;
* surfaces, areas and volumes from marching cubes.

## Step 1: a voxel mesh

```python
import numpy as np
from tpmsfem.geometry import TPMS, box
from tpmsfem.mesh import implicit_to_surface, surface_area_volume, voxelize, write_stl

lattice = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.3)
block = box((0, 0, 0), (8, 8, 8)) & lattice
mesh = voxelize(block, (0, 0, 0), (8, 8, 8), h=0.2)        # h: voxel edge [mm]
print(mesh.summary())
mesh.save("gyroid_block.vtu")                              # ParaView; .inp, .msh, ... via meshio
```

Output (from the test run):

```text
Mesh[hex8] 20,352 cells, 36,379 nodes, 109,137 dofs, bbox [0. 0. 0.] .. [8. 8. 8.], voxel h=0.2
```

## Step 2: how fine?

The wall of a 30 % gyroid sheet with a 4 mm cell is about 0.4 mm thick. Rule of thumb: at
least 4 voxels across the thinnest wall. Watch how the meshed volume fraction converges:

```python
for h in (0.4, 0.2, 0.1):
    m = voxelize(block, (0, 0, 0), (8, 8, 8), h=h)
    print(f"h = {h:.2f} mm: {m.n_cells:7d} elements, volume fraction {m.n_cells * h**3 / 8**3:.3f}")
```

Output (from the test run):

```text
h = 0.40 mm:    1800 elements, volume fraction 0.225
h = 0.20 mm:   20352 elements, volume fraction 0.318
h = 0.10 mm:  153848 elements, volume fraction 0.300
```

`subsample=2` classifies each voxel from 8 sample points instead of 1, which smooths the
volume error at a given h.

## Step 3: options

```python
two = voxelize(block, (0, 0, 0), (8, 8, 8), h=0.2, keep="largest")      # drop floating islands
plate = box((0, 0, 0), (8, 8, 1.0))
multi = voxelize(block | plate, (0, 0, 0), (8, 8, 8), h=0.2,
                 material_fields=[(1, plate), (2, block)])              # two material ids
print("material ids:", np.unique(multi.cell_data["material"]))
```

Output (from the test run):

```text
material ids: [1 2]
```

For homogenisation (Tutorial 5) mesh one cell with `periodic=(True, True, True)`.

## Step 4: a surface (STL)

```python
verts, faces = implicit_to_surface(block, (0, 0, 0), (8, 8, 8), h=0.1)
write_stl("gyroid_block.stl", verts, faces)
area, volume = surface_area_volume(verts, faces)
print(f"{len(faces)} triangles, surface {area:.0f} mm^2, volume {volume:.1f} mm^3, "
      f"relative density {volume / 8**3:.3f}, area per volume {area / 8**3:.2f} 1/mm")
```

Output (from the test run):

```text
248624 triangles, surface 843 mm^2, volume 151.3 mm^3, relative density 0.295, area per volume 1.65 1/mm
```

The surface area per volume matters for heat transfer, cell attachment in implants and
powder adhesion (Tutorial 19).

## Step 5: tetrahedra (optional)

Voxels make stair-steps: stresses at single elements on the surface are inflated. For
accurate hot spots, mesh the STL with tetrahedra (needs gmsh, TetGen or fTetWild; a few
minutes):

<!-- not run: needs a few minutes and a tetrahedral mesher -->
```python
from tpmsfem.mesh.adapters import gmsh_tetmesh, tet_quality
tet = gmsh_tetmesh(verts, faces, size=0.3, order=2)        # tet10
print(tet.summary(), tet_quality(tet))
```

## Check yourself
* The volume fraction approaches 0.30 as h shrinks.
* Read meshes from other tools with `tpmsfem.mesh.read_mesh(path)`; voxelise a CAD STL with
  `voxelize_stl(path, h)`.

## Next
[Tutorial 4: a compression test](04_linear_static.md).
