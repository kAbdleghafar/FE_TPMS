"""Geometric multigrid: same answers as the direct solver, mesh-independent convergence."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.homogenize import homogenize
from tpmsfem.geometry import TPMS, box, lattice_cube
from tpmsfem.mesh import voxelize
from tpmsfem.residual.am import AMProcess

TI = tf.Material("ti", 110_000.0, 0.3)


def _compress(mesh, solver, mat=TI, strain=1e-3, n_inc=1):
    md = tf.SolidModel(mesh, mat)
    z = mesh.points[:, 2]
    md.fix(np.flatnonzero(z < 1e-9))
    md.fix(np.flatnonzero(z > z.max() - 1e-9), (2,), -strain * z.max())
    info = md.solve(solver=solver, n_inc=n_inc)
    return md, info


def test_gmg_matches_direct_on_lattice():
    mesh = voxelize(lattice_cube(TPMS("gyroid", 2.0, "sheet", 0.35), 4.0, plate=0.4), (0, 0, 0), (4, 4, 4), 0.125)
    a, _ = _compress(mesh, "direct")
    b, info = _compress(mesh, "gmg")
    assert len(b._cache[("gmg", b._version)].levels) >= 2
    assert info.krylov_iterations < 60
    assert np.abs(a.displacement() - b.displacement()).max() < 1e-6 * np.abs(a.displacement()).max()


def test_gmg_iterations_do_not_grow_with_refinement():
    its = []
    for h in (0.25, 0.125):
        mesh = voxelize(box((0, 0, 0), (8, 2, 2)), (0, 0, 0), (8, 2, 2), h)
        _, info = _compress(mesh, "gmg")
        its.append(info.krylov_iterations)
    assert its[1] <= its[0] + 5


def test_gmg_periodic_homogenization():
    mesh = voxelize(TPMS("gyroid", 1.0, "sheet", 0.3), (0, 0, 0), (1, 1, 1), 1 / 32, periodic=(True,) * 3)
    ra = homogenize(tf.SolidModel(mesh, TI), components=(0, 1, 2, 3), solver="direct")
    rb = homogenize(tf.SolidModel(mesh, TI), components=(0, 1, 2, 3), solver="gmg")
    assert np.allclose(rb.C, ra.C, rtol=1e-6)


def test_gmg_layer_by_layer_with_elastic_plate():
    part = box((0, 0, 0), (6, 2, 3))
    plate = box((-1, -1, -2), (7, 3, 0))
    mesh = voxelize(part | plate, (-1, -1, -2), (7, 3, 3), 0.25, material_fields=[(1, part), (2, plate)])
    res = []
    for solver in ("direct", "gmg"):
        md = tf.SolidModel(mesh, {1: TI, 2: TI})
        am = AMProcess(md, layer_height=0.5, substrate=np.flatnonzero(mesh.cell_data["material"] == 2), solver=solver)
        am.build(verbose=False)
        am.remove_from_plate(verbose=False)
        res.append(md.displacement())
    assert np.abs(res[0] - res[1]).max() < 1e-6 * np.abs(res[0]).max()


def test_gmg_preconditions_plastic_newton():
    mat = tf.Material("j2", 110_000.0, 0.3, yield_stress=300.0, hardening=2_000.0)
    mesh = voxelize(lattice_cube(TPMS("gyroid", 2.0, "sheet", 0.4), 4.0, plate=0.4), (0, 0, 0), (4, 4, 4), 0.2)
    a, _ = _compress(mesh, "direct", mat, strain=6e-3, n_inc=3)
    b, _ = _compress(mesh, "gmg", mat, strain=6e-3, n_inc=3)
    assert a.plastic_strain()[1].max() > 0
    assert np.abs(a.displacement() - b.displacement()).max() < 1e-5 * np.abs(a.displacement()).max()
