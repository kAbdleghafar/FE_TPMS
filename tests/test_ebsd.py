"""EBSD toolkit: orientation matrices, cubic disorientation, segmentation, KAM."""
import numpy as np
import pytest

from tpmsfem.process import ebsd as E
from tpmsfem.process import solidification as S


def test_euler_matrices_are_rotations_and_symmetry_group():
    rng = np.random.default_rng(0)
    e = rng.uniform(0, [2 * np.pi, np.pi, 2 * np.pi], size=(50, 3))
    g = E.euler_to_matrix(e)
    assert np.allclose(np.einsum("nij,nkj->nik", g, g), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(g), 1.0)
    assert len(E.SYM) == 24


def test_disorientation_properties():
    rng = np.random.default_rng(1)
    g = S.random_rotations(200, rng)
    assert np.allclose(E.misorientation(g, g), 0.0, atol=1e-5)
    # a symmetry-equivalent orientation has zero disorientation
    assert np.allclose(E.misorientation(g, np.einsum("ij,njk->nik", E.SYM[7], g)), 0.0, atol=1e-5)
    # a 30 deg rotation about [001] is 30 deg; the cubic maximum is 62.8 deg
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert E.misorientation(np.eye(3), rz) == pytest.approx(30.0, abs=1e-6)
    mis = E.misorientation(g[:100], g[100:])
    assert mis.max() <= 62.81


def test_segmentation_and_kam_on_a_two_grain_map():
    ny, nx = 20, 30
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    g = np.broadcast_to(np.eye(3), (ny, nx, 3, 3)).copy()
    g[:, 15:] = rz
    valid = np.ones((ny, nx), bool)
    lab = E.segment_grains(g, valid, threshold=10.0, min_size=5)
    assert lab.max() + 1 == 2 and len(np.unique(lab[:, :15])) == 1
    k = E.kam(g, valid)
    assert np.nanmax(k) < 1e-5                                              # 40 deg boundary excluded (< 5 deg kernel)
    sh = E.grain_shapes(lab, (0.5, 0.5))
    assert np.allclose(sh["area"], 15 * 20 * 0.25)
