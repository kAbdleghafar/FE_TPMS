"""Verification against closed-form solutions (code vs. math)."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.bench import analytical as A
from tpmsfem.geometry import box
from tpmsfem.mesh import Mesh, voxelize

STEEL = tf.Material("steel", 200_000.0, 0.3)


def _box_hex_mesh(n=(3, 3, 3), L=(1.0, 1.0, 1.0), distort=0.0, seed=0):
    """Structured hex8 mesh (not flagged as voxel), optionally distorted."""
    m = voxelize(box((0, 0, 0), L), (0, 0, 0), L, L[0] / n[0])
    pts = m.points.copy()
    if distort:
        rng = np.random.default_rng(seed)
        lo, hi = pts.min(0), pts.max(0)
        inner = np.all((pts > lo + 1e-9) & (pts < hi - 1e-9), axis=1)
        pts[inner] += distort * (L[0] / n[0]) * rng.uniform(-1, 1, (inner.sum(), 3))
    return Mesh(pts, m.cells, "hex8")


def _linear_field(p):
    G = np.array([[1.0, 0.3, -0.2], [0.1, -0.5, 0.4], [0.25, 0.2, 0.7]]) * 1e-3
    return p @ G.T, G


@pytest.mark.parametrize("bbar", [False, True])
def test_patch_test_distorted_hex(bbar):
    m = _box_hex_mesh(distort=0.25)
    md = tf.SolidModel(m, STEEL, bbar=bbar)
    bn = m.boundary_nodes()
    ub, G = _linear_field(m.points)
    for c in range(3):
        md.fix(bn, (c,), ub[bn, c])
    md.solve(solver="direct")
    assert np.allclose(md.displacement(), ub, atol=1e-12)
    eps = 0.5 * (G + G.T)
    sig = STEEL.lam * np.trace(eps) * np.eye(3) + 2 * STEEL.mu * eps
    s6 = md.stress()
    assert np.allclose(s6[..., 0], sig[0, 0], rtol=1e-9)
    assert np.allclose(s6[..., 3], sig[1, 2], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_patch_test_tets(order):
    pytest.importorskip("tetgen")
    from tpmsfem.mesh.adapters import tetgen_tetmesh
    from tpmsfem.mesh.surface import implicit_to_surface

    v, f = implicit_to_surface(box((0, 0, 0), (1, 1, 1)), (-0.1,) * 3, (1.1,) * 3, 0.1)
    m = tetgen_tetmesh(v, f, order=order)
    md = tf.SolidModel(m, STEEL)
    bn = m.boundary_nodes()
    ub, G = _linear_field(m.points)
    for c in range(3):
        md.fix(bn, (c,), ub[bn, c])
    md.solve(solver="direct")
    assert np.allclose(md.displacement(), ub, atol=1e-11)


def test_cantilever_converges_to_beam_theory():
    P, L, b, h = 10.0, 10.0, 1.0, 1.0
    ref = A.cantilever_tip_deflection(P, L, STEEL.E, STEEL.nu, b, h)
    errs = []
    for hv in (0.25, 0.125):
        m = voxelize(box((0, 0, 0), (L, b, h)), (0, 0, 0), (L, b, h), hv)
        md = tf.SolidModel(m, STEEL)
        md.fix(m.nodes_where(lambda x, y, z: x < 1e-9))
        faces, ft, _ = m.boundary_faces()
        tip = faces[m.points[faces].mean(1)[:, 0] > L - 1e-9]
        md.add_surface_load(tip, ft, total_force=(0, 0, -P))
        md.solve(solver="direct")
        errs.append(abs(-md.displacement()[:, 2].min() / ref - 1))
    assert errs[1] < errs[0] and errs[1] < 0.04


def test_eigenstrain_free_and_clamped():
    m = voxelize(box((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), 0.25)
    eps = 1e-3
    # free body (3-2-1 support): stress-free, u = eps * x
    md = tf.SolidModel(m, STEEL)
    a = m.closest_node((0, 0, 0)); b = m.closest_node((1, 0, 0)); c = m.closest_node((0, 1, 0))
    md.fix([a], (0, 1, 2)); md.fix([b], (1, 2)); md.fix([c], (2,))
    md.set_eigenstrain(np.arange(m.n_cells), (eps, eps, eps, 0, 0, 0))
    md.solve(solver="direct")
    assert np.abs(md.stress()).max() < 1e-8 * STEEL.E * eps
    assert np.allclose(md.displacement(), eps * (m.points - m.points[a]), atol=1e-12)
    # clamped: sigma = -E eps / (1 - 2 nu) I
    md = tf.SolidModel(m, STEEL)
    md.fix(m.boundary_nodes())
    md.set_eigenstrain(np.arange(m.n_cells), (eps, eps, eps, 0, 0, 0))
    md.solve(solver="direct")
    ref = A.clamped_isotropic_eigenstress(eps, STEEL.E, STEEL.nu)
    assert np.allclose(md.stress()[..., :3], ref, rtol=1e-9)


def test_bilayer_curvature():
    L, h, hv, de = 16.0, 1.0, 0.125, 1e-3
    mat = tf.Material("nu0", 100_000.0, 0.0)
    m = voxelize(box((0, 0, 0), (L, 1.0, h)), (0, 0, 0), (L, 1.0, h), hv)
    md = tf.SolidModel(m, mat)
    top = m.cells_where(lambda x, y, z: z > h / 2)
    md.set_eigenstrain(top, (de, 0, 0, 0, 0, 0))
    a = m.closest_node((0, 0, 0)); b = m.closest_node((L, 0, 0)); c = m.closest_node((0, 1, 0))
    md.fix([a], (0, 1, 2)); md.fix([b], (1, 2)); md.fix([c], (2,))
    md.solve(solver="direct")
    p, u = m.points, md.displacement()
    sel = (np.abs(p[:, 2] - h / 2) < 1e-9) & (np.abs(p[:, 1] - 0.5) < 1e-9) & (np.abs(p[:, 0] - L / 2) < L / 4)
    coef = np.polyfit(p[sel, 0], u[sel, 2], 2)
    kappa = -2 * coef[0]   # top expands -> curls down (concave down)
    assert kappa == pytest.approx(A.bilayer_curvature(de, h), rel=0.02)


def test_uniaxial_plasticity_linear_hardening():
    mat = tf.Material("j2", 200_000.0, 0.3, yield_stress=300.0, hardening=5_000.0)
    m = voxelize(box((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), 0.5)
    md = tf.SolidModel(m, mat)  # B-bar on by default for plastic hex8
    assert md.bbar
    x, y, z = m.points.T
    md.fix(np.flatnonzero(x < 1e-9), (0,)); md.fix(np.flatnonzero(y < 1e-9), (1,)); md.fix(np.flatnonzero(z < 1e-9), (2,))
    top = np.flatnonzero(z > 1 - 1e-9)
    for strain in (0.001, 0.004, 0.01):
        md.fix(top, (2,), strain)
        md.solve(n_inc=2, solver="direct")
        assert md.stress()[..., 2].mean() == pytest.approx(
            float(A.uniaxial_linear_hardening(strain, mat.E, 300.0, 5_000.0)), rel=1e-6)
    # elastic unloading to zero force leaves eps_p = eps - sigma/E
    s_peak = md.stress()[..., 2].mean()
    md.release(top, (2,))
    md.solve(solver="direct")
    assert np.abs(md.stress()).max() < 1e-6 * s_peak
    assert md.displacement()[top, 2].mean() == pytest.approx(0.01 - s_peak / mat.E, rel=1e-6)


def test_lame_sphere_tet10():
    pytest.importorskip("gmsh")
    a, b, p = 1.0, 2.0, 100.0
    m = A.gmsh_sphere_shell_octant(a, b, size=0.25, order=2)
    md = tf.SolidModel(m, STEEL)
    x, y, z = m.points.T
    md.fix(np.flatnonzero(np.abs(x) < 1e-9), (0,))
    md.fix(np.flatnonzero(np.abs(y) < 1e-9), (1,))
    md.fix(np.flatnonzero(np.abs(z) < 1e-9), (2,))
    faces, ft, _ = m.boundary_faces()
    fr = np.linalg.norm(m.points[faces[:, :3]].mean(1), axis=1)
    md.add_surface_load(faces[fr < 0.5 * (a + b)], ft, pressure=p)
    md.solve(solver="direct")
    r = np.linalg.norm(m.points, axis=1)
    ur = (md.displacement() * m.points).sum(1) / r
    _, _, ur_ref = A.lame_sphere(r, a, b, p, STEEL.E, STEEL.nu)
    used = np.unique(m.cells)
    assert np.abs(ur[used] / ur_ref[used] - 1).max() < 0.01


def test_hill_elastic_plastic_sphere():
    pytest.importorskip("gmsh")
    a, b, sy = 1.0, 2.0, 250.0
    mat = tf.Material("epp", 200_000.0, 0.3, yield_stress=sy, hardening=0.0)
    c = 1.5
    p = A.hill_sphere_pressure(c, a, b, sy)
    m = A.gmsh_sphere_shell_octant(a, b, size=0.18, order=2)
    md = tf.SolidModel(m, mat)
    x, y, z = m.points.T
    md.fix(np.flatnonzero(np.abs(x) < 1e-9), (0,))
    md.fix(np.flatnonzero(np.abs(y) < 1e-9), (1,))
    md.fix(np.flatnonzero(np.abs(z) < 1e-9), (2,))
    faces, ft, _ = m.boundary_faces()
    fr = np.linalg.norm(m.points[faces[:, :3]].mean(1), axis=1)
    md.add_surface_load(faces[fr < 0.5 * (a + b)], ft, pressure=p)
    md.solve(n_inc=5, solver="direct")
    r = np.linalg.norm(m.points, axis=1)
    outer = np.flatnonzero(np.abs(r - b) < 1e-6)
    ub = ((md.displacement()[outer] * m.points[outer]).sum(1) / r[outer]).mean()
    assert ub == pytest.approx(A.hill_sphere_outer_displacement(c, b, sy, mat.E, mat.nu), rel=0.02)
