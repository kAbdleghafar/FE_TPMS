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


def keep_connected(mask: np.ndarray, keep: str = "largest", min_voxels: int = 1) -> np.ndarray:
    """Remove voxel islands. Connectivity is by faces only: voxels that touch
    along an edge or a corner form a mechanism (singular stiffness)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    if keep == "largest":
        return lab == np.argmax(sizes)
    if keep == "all":
        return np.isin(lab, np.flatnonzero(sizes >= min_voxels))
    raise ValueError(keep)


def mask_to_mesh(mask: np.ndarray, lo, h: float, materials: Optional[np.ndarray] = None) -> Mesh:
    """Hex8 mesh of the solid voxels. Cells are ordered with z slowest, which
    keeps layer-by-layer activation cache-friendly."""
    nx, ny, nz = mask.shape
    kji = np.argwhere(mask.transpose(2, 1, 0))          # sorted by k, then j, then i
    ijk = kji[:, ::-1].astype(np.int64)
    corner = ijk[:, None, :] + HEX_CORNERS[None, :, :]   # (ne, 8, 3)
    gid = (corner[..., 0] * (ny + 1) + corner[..., 1]) * (nz + 1) + corner[..., 2]
    used, inv = np.unique(gid.ravel(), return_inverse=True)
    cells = inv.reshape(-1, 8)
    k = used % (nz + 1)
    j = (used // (nz + 1)) % (ny + 1)
    i = used // ((nz + 1) * (ny + 1))
    pts = np.asarray(lo, float)[None, :] + h * np.stack([i, j, k], axis=1).astype(float)
    mesh = Mesh(pts, cells, "hex8",
                meta=dict(voxel_size=float(h), origin=np.asarray(lo, float), grid_shape=mask.shape))
    mesh.cell_data["ijk"] = ijk
    if materials is not None:
        mesh.cell_data["material"] = materials[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.int64)
    return mesh


def voxelize(field, lo: Sequence[float], hi: Sequence[float], h: float, *,
             subsample: int = 1, keep: str = "largest",
             material_fields: Optional[Sequence] = None) -> Mesh:
    """Voxel hex8 mesh of ``{field < 0}`` inside the box ``[lo, hi]``.

    ``material_fields``: optional list of ``(material_id, field)``; a solid voxel
    gets the id of the *last* field that contains its centre (default 0).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / h - 1e-9)) for d in range(3))
    mask = classify(field, lo, h, shape, subsample=subsample)
    mask = keep_connected(mask, keep=keep)
    if not mask.any():
        raise ValueError("field produced an empty voxel mask")
    mats = None
    if material_fields:
        mats = np.zeros(shape, np.int64)
        for mid, f in material_fields:
            mats[classify(f, lo, h, shape)] = mid
    return mask_to_mesh(mask, lo, h, mats)
