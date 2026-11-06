"""Implicit field -> conforming voxel (hex8) mesh.

Voxel meshes are the workhorse of micro-FE for lattices: meshing never fails,
every element has the same stiffness matrix (so the GPU kernel is a dense GEMM)
and resolution is a single number. Accuracy needs roughly >= 4 voxels across
the thinnest wall; check with a mesh-convergence study (see ``bench.metrics.gci``).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..backend import asnumpy, get_xp
from .mesh import Mesh

# Abaqus C3D8 corner order in (i, j, k) offsets
HEX_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])


def classify(field, lo, h: float, shape, subsample: int = 1, slab: int = 32) -> np.ndarray:
    """Boolean solid mask on a voxel grid, evaluated on the active backend.

    With ``subsample=s`` each voxel is sampled at s^3 points and is solid when
    at least half of them are inside (better volume fraction for thin walls).
    """
    xp = get_xp()
    nx, ny, nz = shape
    lo = np.asarray(lo, float)
    mask = np.zeros(shape, dtype=bool)
    s = int(subsample)
    off = (np.arange(s) + 0.5) / s
    xs = xp.asarray(lo[0] + (np.arange(nx)[:, None] + off[None, :]).ravel() * h)
    ys = xp.asarray(lo[1] + (np.arange(ny)[:, None] + off[None, :]).ravel() * h)
    for k0 in range(0, nz, slab):
        k1 = min(nz, k0 + slab)
        zs = xp.asarray(lo[2] + (np.arange(k0, k1)[:, None] + off[None, :]).ravel() * h)
        X, Y, Z = xp.meshgrid(xs, ys, zs, indexing="ij")
        inside = field(X, Y, Z) < 0
        if s > 1:
            inside = inside.reshape(nx, s, ny, s, k1 - k0, s).mean(axis=(1, 3, 5)) >= 0.5
        mask[:, :, k0:k1] = asnumpy(inside)
    return mask


def surface_to_mask(verts, faces, lo, h: float, shape, batch: int = 200_000) -> np.ndarray:
    """Solid mask of a closed triangle surface (e.g. an STL from CAD) on a voxel grid.

    A ray is cast along +z through every voxel-column centre and a voxel is
    inside when an odd number of surface crossings lie below its centre
    (parity rule). Ray positions are shifted by a tiny irrational fraction of
    ``h`` so that rays never pass exactly through triangle edges or vertices
    of CAD geometry whose coordinates are round numbers.

    Fully vectorised: triangles are processed in batches; every
    (triangle, column) candidate pair from the triangle's bounding box is
    tested at once, and the parity is filled per column with an XOR scan.
    """
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, np.int64)
    lo = np.asarray(lo, float)
    nx, ny, nz = shape
    ox, oy = lo[0] + 1.31e-5 * h, lo[1] + 2.77e-5 * h      # jittered ray origins
    zc = lo[2] + (np.arange(nz) + 0.5) * h
    flips = np.zeros((nx * ny, nz + 1), np.uint8)          # parity flips per column, per crossing level
    tri = verts[faces]                                     # (nt, 3, 3)
    for s0 in range(0, len(tri), batch):
        T = tri[s0:s0 + batch]
        x0, y0, z0 = T[:, 0, 0], T[:, 0, 1], T[:, 0, 2]
        x1, y1, z1 = T[:, 1, 0], T[:, 1, 1], T[:, 1, 2]
        x2, y2, z2 = T[:, 2, 0], T[:, 2, 1], T[:, 2, 2]
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        # candidate column index ranges (column centres at o + (i + 0.5) h)
        i0 = np.maximum(0, np.ceil((np.minimum(np.minimum(x0, x1), x2) - ox) / h - 0.5)).astype(np.int64)
        i1 = np.minimum(nx - 1, np.floor((np.maximum(np.maximum(x0, x1), x2) - ox) / h - 0.5)).astype(np.int64)
        j0 = np.maximum(0, np.ceil((np.minimum(np.minimum(y0, y1), y2) - oy) / h - 0.5)).astype(np.int64)
        j1 = np.minimum(ny - 1, np.floor((np.maximum(np.maximum(y0, y1), y2) - oy) / h - 0.5)).astype(np.int64)
        ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
        cnt = np.where(np.abs(det) > 1e-300, ni * nj, 0)   # vertical triangles never cross a z-ray
        if cnt.sum() == 0:
            continue
        t = np.repeat(np.arange(len(T)), cnt)              # (triangle, column) candidate pairs
        local = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ii = i0[t] + local // nj[t]
        jj = j0[t] + local % nj[t]
        X = ox + (ii + 0.5) * h
        Y = oy + (jj + 0.5) * h
        l0 = ((y1[t] - y2[t]) * (X - x2[t]) + (x2[t] - x1[t]) * (Y - y2[t])) / det[t]
        l1 = ((y2[t] - y0[t]) * (X - x2[t]) + (x0[t] - x2[t]) * (Y - y2[t])) / det[t]
        l2 = 1.0 - l0 - l1
        hit = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not hit.any():
            continue
        zhit = l0[hit] * z0[t[hit]] + l1[hit] * z1[t[hit]] + l2[hit] * z2[t[hit]]
        col = ii[hit] * ny + jj[hit]
        k = np.searchsorted(zc, zhit, side="right")        # first voxel centre above the crossing
        np.bitwise_xor.at(flips, (col, k), 1)
    inside = np.bitwise_xor.accumulate(flips[:, :nz], axis=1).astype(bool)
    return inside.reshape(nx, ny, nz)


def voxelize_stl(path_or_surface, h: float, lo=None, hi=None, *, keep: str = "largest",
                 extra_fields=None) -> Mesh:
    """Voxel hex8 mesh of a closed STL surface (path, or ``(verts, faces)``).

    ``extra_fields``: optional list of ``(material_id, field)`` implicit solids
    added to the part (e.g. a build plate ``box``); the STL part gets id 1,
    the extra solids their own ids.
    """
    if isinstance(path_or_surface, str):
        import meshio

        m = meshio.read(path_or_surface)
        verts, faces = m.points, m.cells_dict["triangle"]
    else:
        verts, faces = path_or_surface
    verts = np.asarray(verts, float)
    lo = verts.min(axis=0) if lo is None else np.asarray(lo, float)
    hi = verts.max(axis=0) if hi is None else np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / h - 1e-9)) for d in range(3))
    part = surface_to_mask(verts, faces, lo, h, shape)
    mats = np.where(part, 1, 0).astype(np.int64)
    mask = part.copy()
    for mid, f in (extra_fields or []):
        sel = classify(f, lo, h, shape) & ~part
        mats[sel] = mid
        mask |= sel
    mask = keep_connected(mask, keep=keep)
    return mask_to_mesh(mask, lo, h, mats)


def keep_connected(mask: np.ndarray, keep: str = "largest", min_voxels: int = 1,
                   periodic=(False, False, False)) -> np.ndarray:
    """Remove voxel islands. Connectivity is by faces only: voxels that touch
    along an edge or a corner form a mechanism (singular stiffness). Along
    ``periodic`` axes, voxels on opposite faces are neighbours."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n <= 1:
        return mask
    if any(periodic):
        parent = np.arange(n + 1)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for d, per in enumerate(periodic):
            if not per:
                continue
            a = np.take(lab, 0, axis=d).ravel()
            b = np.take(lab, -1, axis=d).ravel()
            for x, y in set(zip(a[(a > 0) & (b > 0)].tolist(), b[(a > 0) & (b > 0)].tolist())):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
        roots = np.array([find(i) for i in range(n + 1)])
        lab = roots[lab]
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if keep == "largest":
        return lab == np.argmax(sizes)
    if keep == "all":
        return np.isin(lab, np.flatnonzero(sizes >= min_voxels))
    raise ValueError(keep)


def mask_to_mesh(mask: np.ndarray, lo, h: float, materials: Optional[np.ndarray] = None,
                 periodic=(False, False, False)) -> Mesh:
    """Hex8 mesh of the solid voxels. Cells are ordered with z slowest, which
    keeps layer-by-layer activation cache-friendly.

    Along ``periodic`` axes the nodes on the last face are identified with
    those on the first face (the box must span exactly one period). Only the
    voxel kernel (shared element geometry) is valid on such meshes: node
    coordinates of wrapped nodes are those of the first face.
    """
    nx, ny, nz = mask.shape
    kji = np.argwhere(mask.transpose(2, 1, 0))          # sorted by k, then j, then i
    ijk = kji[:, ::-1].astype(np.int64)
    corner = ijk[:, None, :] + HEX_CORNERS[None, :, :]   # (ne, 8, 3)
    for d, per in enumerate(periodic):
        if per:
            corner[..., d] %= mask.shape[d]
    gid = (corner[..., 0] * (ny + 1) + corner[..., 1]) * (nz + 1) + corner[..., 2]
    used, inv = np.unique(gid.ravel(), return_inverse=True)
    cells = inv.reshape(-1, 8)
    k = used % (nz + 1)
    j = (used // (nz + 1)) % (ny + 1)
    i = used // ((nz + 1) * (ny + 1))
    pts = np.asarray(lo, float)[None, :] + h * np.stack([i, j, k], axis=1).astype(float)
    mesh = Mesh(pts, cells, "hex8",
                meta=dict(voxel_size=float(h), origin=np.asarray(lo, float), grid_shape=mask.shape,
                          periodic=tuple(bool(p) for p in periodic)))
    mesh.cell_data["ijk"] = ijk
    if materials is not None:
        mesh.cell_data["material"] = materials[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int64)
    return mesh


def voxelize(field, lo: Sequence[float], hi: Sequence[float], h: float, *,
             subsample: int = 1, keep: str = "largest",
             material_fields: Optional[Sequence] = None,
             periodic=(False, False, False)) -> Mesh:
    """Voxel hex8 mesh of ``{field < 0}`` inside the box ``[lo, hi]``.

    ``periodic``: axes along which the box is one period of the geometry
    (for homogenisation, see :mod:`tpmsfem.fem.homogenize`).

    ``material_fields``: optional list of ``(material_id, field)``; a solid voxel
    gets the id of the *last* field that contains its centre (default 0).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / h - 1e-9)) for d in range(3))
    mask = classify(field, lo, h, shape, subsample=subsample)
    mask = keep_connected(mask, keep=keep, periodic=periodic)
    if not mask.any():
        raise ValueError("field produced an empty voxel mask")
    mats = None
    if material_fields:
        mats = np.zeros(shape, np.int64)
        for mid, f in material_fields:
            mats[classify(f, lo, h, shape)] = mid
    return mask_to_mesh(mask, lo, h, mats, periodic=periodic)
