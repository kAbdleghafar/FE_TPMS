"""Implicit field -> watertight triangle surface (marching cubes).

The surface is the input to the open-source tetrahedral meshers (Gmsh, TetGen,
fTetWild) in :mod:`tpmsfem.mesh.adapters`. The sampling grid is padded with an
"outside" layer so the surface is always closed.
"""
from __future__ import annotations

import numpy as np

from ..backend import asnumpy, get_xp


def sample_field(field, lo, hi, h: float, pad: int = 1):
    xp = get_xp()
    lo = np.asarray(lo, float) - pad * h
    hi = np.asarray(hi, float) + pad * h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [xp.asarray(lo[d] + h * np.arange(n[d])) for d in range(3)]
    X, Y, Z = xp.meshgrid(*axes, indexing="ij")
    vals = asnumpy(field(X, Y, Z)).astype(np.float64)
    return vals, lo


def implicit_to_surface(field, lo, hi, h: float, keep_largest: bool = True):
    """Triangulate ``{field = 0}`` clipped to the box ``[lo, hi]``.

    Returns ``(vertices (n,3), faces (m,3))`` with outward-facing triangles.
    """
    from skimage import measure

    from .. import geometry as G

    clip = G.fields.box(lo, hi)
    vals, origin = sample_field(field & clip, lo, hi, h)
    vals[0, :, :] = vals[-1, :, :] = vals[:, 0, :] = vals[:, -1, :] = vals[:, :, 0] = vals[:, :, -1] = abs(h)
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0, spacing=(h, h, h),
                                                allow_degenerate=False)
    verts = verts + origin
    if keep_largest:
        verts, faces = largest_component(verts, faces)
    # orient triangles counter-clockwise seen from outside (positive volume)
    if surface_area_volume(verts, faces)[1] < 0:
        faces = faces[:, ::-1]
    return verts, np.ascontiguousarray(faces)


def largest_component(verts, faces):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    if ncomp == 1:
        return verts, faces
    flab = lab[faces[:, 0]]
    big = np.argmax(np.bincount(flab))
    faces = faces[flab == big]
    used, inv = np.unique(faces, return_inverse=True)
    return verts[used], inv.reshape(faces.shape)


def surface_area_volume(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1).sum()
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return float(area), float(vol)


def write_stl(path: str, verts, faces):
    import meshio

    meshio.write_points_cells(path, np.asarray(verts, float), [("triangle", np.asarray(faces))],
                              binary=True)
