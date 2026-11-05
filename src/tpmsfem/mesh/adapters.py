"""Adapters to open-source tetrahedral meshers.

All adapters take a closed triangle surface (``verts``, ``faces``) -- e.g. from
:func:`tpmsfem.mesh.surface.implicit_to_surface` or an STL exported from CAD --
and return a :class:`~tpmsfem.mesh.mesh.Mesh` of ``tet4`` or ``tet10``.

=========  =================================  ===================================
mesher     install                            notes
=========  =================================  ===================================
Gmsh       ``pip install gmsh``               robust, size control, 2nd order
TetGen     ``pip install tetgen``             fast, quality (radius-edge) control
fTetWild   build from github.com/wildmeshing  most robust on messy/self-
           (``FloatTetwild_bin`` on PATH)     intersecting TPMS surfaces
any other  write .msh/.inp/.vtu/.med          read back with ``read_mesh``
=========  =================================  ===================================

Tet10 node order everywhere in tpmsfem is the Abaqus/VTK order; files are read
through meshio, which converts Gmsh's ordering.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from .mesh import Mesh, read_mesh
from .surface import write_stl


def _quadratic_from_linear(mesh: Mesh) -> Mesh:
    """Promote tet4 -> tet10 with straight-sided mid-edge nodes."""
    edges_local = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
    e = mesh.cells[:, edges_local].reshape(-1, 2)
    key = np.sort(e, axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    mids = 0.5 * (mesh.points[uniq[:, 0]] + mesh.points[uniq[:, 1]])
    pts = np.vstack([mesh.points, mids])
    cells = np.hstack([mesh.cells, mesh.n_points + inv.reshape(-1, 6)])
    return Mesh(pts, cells, "tet10", dict(mesh.cell_data), meta=dict(mesh.meta))


# --------------------------------------------------------------------------- Gmsh
def gmsh_tetmesh(verts, faces, size: Optional[float] = None, order: int = 1,
                 optimize: bool = True, remesh_surface: bool = False, angle_deg: float = 40.0,
                 verbose: bool = False) -> Mesh:
    """Tetrahedralise a closed surface with the Gmsh API.

    By default the input triangulation is kept as the boundary (robust for
    marching-cubes TPMS surfaces) and only the volume is meshed, graded by
    ``size``. ``remesh_surface=True`` splits the surface into patches
    (``classifySurfaces``), reparametrises and remeshes it to ``size`` too.
    """
    import gmsh

    tmp = tempfile.mkdtemp(prefix="tpmsfem_gmsh_")
    stl = os.path.join(tmp, "surf.stl")
    msh = os.path.join(tmp, "vol.msh")
    write_stl(stl, verts, faces)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.merge(stl)
        if remesh_surface:
            gmsh.model.mesh.classifySurfaces(np.deg2rad(angle_deg), True, True, np.pi)
            gmsh.model.mesh.createGeometry()
        surfs = [s[1] for s in gmsh.model.getEntities(2)]
        loop = gmsh.model.geo.addSurfaceLoop(surfs)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        if size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(size))
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)          # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], 1)
        gmsh.write(msh)
    finally:
        gmsh.finalize()
    m = read_mesh(msh, "tet10" if order == 2 else "tet4")
    shutil.rmtree(tmp, ignore_errors=True)
    m.meta["mesher"] = "gmsh"
    return m


def gmsh_from_file(path: str, size: Optional[float] = None, order: int = 1) -> Mesh:
    """Mesh a CAD/geometry file Gmsh can read (.step/.stp/.brep/.geo/.stl)."""
    import gmsh

    tmp = tempfile.mkdtemp(prefix="tpmsfem_gmsh_")
    msh = os.path.join(tmp, "vol.msh")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        if size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(size))
        gmsh.model.mesh.generate(3)
        if order == 2:
            gmsh.model.mesh.setOrder(2)
        gmsh.write(msh)
    finally:
        gmsh.finalize()
    m = read_mesh(msh, "tet10" if order == 2 else "tet4")
    shutil.rmtree(tmp, ignore_errors=True)
    return m


# ------------------------------------------------------------------------- TetGen
def tetgen_tetmesh(verts, faces, order: int = 1, minratio: float = 1.5,
                   mindihedral: float = 10.0, maxvolume: Optional[float] = None) -> Mesh:
    """Constrained Delaunay tetrahedralisation with TetGen (via ``pip install tetgen``)."""
    import tetgen

    tg = tetgen.TetGen(np.asarray(verts, float), np.asarray(faces, np.int32))
    kw = dict(order=1, minratio=minratio, mindihedral=mindihedral, quality=True)
    if maxvolume is not None:
        kw["maxvolume"] = maxvolume
    out = tg.tetrahedralize(**kw)
    nodes, elems = out[0], out[1]
    m = Mesh(np.asarray(nodes, float), np.asarray(elems, np.int64), "tet4")
    m = _fix_orientation(m)
    m.meta["mesher"] = "tetgen"
    return _quadratic_from_linear(m) if order == 2 else m


# ----------------------------------------------------------------------- fTetWild
def ftetwild_tetmesh(verts, faces, edge_length_rel: float = 0.05, eps_rel: float = 1e-3,
                     order: int = 1, exe: Optional[str] = None, extra_args=()) -> Mesh:
    """Robust tetrahedral meshing with fTetWild (Hu et al., 2020).

    ``edge_length_rel`` (-l) and ``eps_rel`` (-e) are relative to the bounding
    box diagonal. Needs the ``FloatTetwild_bin`` executable (set ``exe`` or put
    it on PATH).
    """
    exe = exe or shutil.which("FloatTetwild_bin") or shutil.which("fTetWild")
    if exe is None:
        raise FileNotFoundError("FloatTetwild_bin not found; build it from "
                                "https://github.com/wildmeshing/fTetWild")
    tmp = tempfile.mkdtemp(prefix="tpmsfem_ftw_")
    stl = os.path.join(tmp, "surf.stl")
    out = os.path.join(tmp, "vol.msh")
    write_stl(stl, verts, faces)
    cmd = [exe, "-i", stl, "-o", out, "-l", str(edge_length_rel), "-e", str(eps_rel),
           "--no-color", *extra_args]
    subprocess.run(cmd, check=True, capture_output=True)
    m = read_mesh(out, "tet4")
    m = _fix_orientation(m)
    shutil.rmtree(tmp, ignore_errors=True)
    m.meta["mesher"] = "ftetwild"
    return _quadratic_from_linear(m) if order == 2 else m


def _fix_orientation(m: Mesh) -> Mesh:
    """Make every tet positively oriented (some tools emit mixed orientation)."""
    p = m.points
    c = m.cells
    v = np.einsum("ij,ij->i", np.cross(p[c[:, 1]] - p[c[:, 0]], p[c[:, 2]] - p[c[:, 0]]),
                  p[c[:, 3]] - p[c[:, 0]])
    neg = v < 0
    if neg.any():
        c = c.copy()
        c[neg, 1], c[neg, 2] = c[neg, 2].copy(), c[neg, 1].copy()
        m = Mesh(p, c, m.cell_type, m.cell_data, meta=m.meta)
    keep = np.abs(v) > 1e-14 * np.abs(v).max()
    return m if keep.all() else m.subset(np.flatnonzero(keep))


def tet_quality(mesh: Mesh) -> dict:
    """Simple quality report: normalised shape ratio (1 = regular tet)."""
    p = mesh.points[mesh.cells[:, :4]]
    v = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6
    e = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    lrms = np.sqrt(np.mean([np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in e], axis=0))
    q = 6 * np.sqrt(2) * v / lrms**3
    return dict(min=float(q.min()), p01=float(np.percentile(q, 1)), mean=float(q.mean()),
                n_bad=int((q < 0.1).sum()))
