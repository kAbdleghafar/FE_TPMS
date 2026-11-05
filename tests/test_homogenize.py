"""Periodic homogenisation: exact results for dense cells, laminates and plates."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.homogenize import homogenize
from tpmsfem.geometry import TPMS, box
from tpmsfem.geometry.fields import Field
from tpmsfem.mesh import voxelize

STEEL = tf.Material("steel", 200_000.0, 0.3)


def _full(n=4, periodic=(True, True, True), L=1.0):
    return voxelize(box((-1, -1, -1), (2, 2, 2)), (0, 0, 0), (L, L, L), L / n, periodic=periodic)


def test_periodic_mesh_topology():
    m = _full(4)
    assert m.n_points == 4**3 and m.n_cells == 4**3          # every node shared by 8 cells
    m = _full(4, periodic=(True, True, False))
    assert m.n_points == 4 * 4 * 5


def test_dense_cell_recovers_isotropic_stiffness():
    m = _full(3)
    r = homogenize(tf.SolidModel(m, STEEL))
    lam, mu = STEEL.lam, STEEL.mu
    C = np.diag([lam + 2 * mu] * 3 + [mu] * 3)
    C[:3, :3] += lam * (1 - np.eye(3))
    assert np.allclose(r.C, C, rtol=1e-8, atol=1e-6)
    assert r.young("xx") == pytest.approx(STEEL.E, rel=1e-8)
    assert r.zener() == pytest.approx(1.0)


def test_dense_plate_free_in_z():
    m = _full(3, periodic=(True, True, False))
    r = homogenize(tf.SolidModel(m, STEEL), components=(0, 1, 5))
    assert r.young("xx") == pytest.approx(STEEL.E, rel=1e-8)
    assert r.poisson("xx", "yy") == pytest.approx(STEEL.nu, rel=1e-8)
    assert r.shear("xy") == pytest.approx(STEEL.mu, rel=1e-8)


def test_laminate_voigt_reuss():
    # two layers along z, nu = 0 -> E_x = Voigt (arithmetic), E_z = Reuss (harmonic)
    a, b = tf.Material("a", 100.0, 0.0), tf.Material("b", 300.0, 0.0)
    m = voxelize(box((-1, -1, -1), (2, 2, 2)), (0, 0, 0), (1, 1, 1), 0.25, periodic=(True,) * 3,
                 material_fields=[(1, box((-1, -1, 0.5), (2, 2, 2)))])
    r = homogenize(tf.SolidModel(m, {0: a, 1: b}))
    assert r.young("xx") == pytest.approx(200.0, rel=1e-8)
    assert r.young("zz") == pytest.approx(150.0, rel=1e-8)


def test_gyroid_cubic_symmetry_and_volume_fraction():
    lat = TPMS("gyroid", cell_size=1.0, kind="sheet", density=0.3)
    m = voxelize(lat, (0, 0, 0), (1, 1, 1), 1 / 24, periodic=(True,) * 3)
    r = homogenize(tf.SolidModel(m, STEEL))
    C = r.C
    assert r.volume_fraction == pytest.approx(0.3, abs=0.02)
    # cubic symmetry: C11 = C22 = C33, C44 = C55 = C66, C12 = C13 = C23
    assert np.ptp(np.diag(C)[:3]) / C[0, 0] < 0.02
    assert np.ptp(np.diag(C)[3:]) / C[3, 3] < 0.02
    assert abs(C[0, 3]) < 1e-3 * C[0, 0]
