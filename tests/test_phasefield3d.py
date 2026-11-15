"""3-D multicomponent phase field (tpmsfem.process.phasefield3d)."""
import numpy as np
import pytest

from tpmsfem.process import phasefield as PF
from tpmsfem.process import phasefield3d as P3

MODEL = PF.DiluteAlloy(c_inf=1.0, k=0.3, freezing_range=10.0, D=1e-9, gibbs_thomson=1e-7, eps4=0.04)
W0 = 2.5 * MODEL.d0
V = MODEL.D / (10 * W0)                 # D/V = 10 W0
G = MODEL.freezing_range / (300 * W0)   # l_T = 300 W0
KERNELS = ["array"] + (["numba"] if PF.numba_available() else [])


@pytest.mark.parametrize("kernel", KERNELS)
def test_one_solute_one_cell_thick_is_the_2d_model(kernel):
    m2 = PF.DirectionalPF(MODEL, G, V, 40 * 0.8 * W0, 40 * W0, noise=0.3, seed=3, kernel=kernel)
    m3 = P3.DirectionalPF3D(P3.MultiAlloy.from_dilute(MODEL), G, V, (40 * 0.8 * W0, 0.8 * W0), 40 * W0, noise=0.3,
                            seed=3, kernel=kernel, ny=1)
    assert (m3.nz, m3.ny, m3.nx) == (m2.nz, 1, m2.nx) and m3.dt == pytest.approx(m2.dt, rel=1e-14)
    for _ in range(300):
        m2.step()
        m3.step()
    assert np.abs(m3.phi[:, 0] - m2.phi).max() < 1e-12
    assert np.abs(m3.U[0, :, 0] - m2.U).max() < 1e-12


@pytest.mark.parametrize("kernel", KERNELS)
def test_two_identical_half_solutes_equal_one(kernel):
    a1 = P3.MultiAlloy.from_dilute(MODEL)
    s = a1.solutes[0]
    a2 = P3.MultiAlloy((P3.Solute("A", s.c_inf / 2, s.k, s.m), P3.Solute("B", s.c_inf / 2, s.k, s.m)), D=MODEL.D,
                       gibbs_thomson=MODEL.gibbs_thomson, eps4=MODEL.eps4)
    assert a2.freezing_range == pytest.approx(a1.freezing_range) and np.allclose(a2.weights, 0.5)
    R = P3.rotation_matrix((20.0, 30.0, 10.0))
    x = P3.DirectionalPF3D(a1, G, V, (16 * 0.8 * W0, 12 * 0.8 * W0), 30 * W0, noise=0.3, seed=1, kernel=kernel, orientation=R)
    y = P3.DirectionalPF3D(a2, G, V, (16 * 0.8 * W0, 12 * 0.8 * W0), 30 * W0, noise=0.3, seed=1, kernel=kernel, orientation=R)
    for _ in range(60):
        x.step()
        y.step()
    assert np.abs(x.phi - y.phi).max() < 1e-13
    assert np.abs(y.U[0] - y.U[1]).max() < 1e-13 and np.abs(x.U[0] - y.U[0]).max() < 1e-13


def _in625_model(kernel, seed=5):
    return P3.DirectionalPF3D(P3.IN625_AMBENCH, 6.6e6, 0.092, (20 * 8e-9, 16 * 8e-9), 40 * 8e-9, W0_m=10e-9, noise=0.3,
                              seed=seed, kernel=kernel, orientation=P3.rotation_matrix((15.0, 25.0, 40.0)))


def test_numba_kernel_matches_array_kernel():
    if not PF.numba_available():
        pytest.skip("numba not installed")
    a, n = _in625_model("array"), _in625_model("numba")
    for _ in range(100):
        a.step()
        n.step()
    assert np.abs(a.phi - n.phi).max() < 1e-12 and np.abs(a.U - n.U).max() < 1e-12


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_solute_conserved_in_closed_box(kernel):
    m = _in625_model(kernel)
    s0 = m.solute()
    m.run(150 * m.dt * m.tau0, shift_at=None, record_every=10 ** 9)
    assert np.abs(m.solute() / s0 - 1.0).max() < 1e-12


def test_in625_alloy_and_trapping():
    a = P3.IN625_AMBENCH
    assert a.names == ["Nb", "Mo", "Cr", "Fe"]
    assert a.freezing_range == pytest.approx(10.5 * 0.52 * 3.75 / 0.48 + 2.9 * 0.15 * 8.83 / 0.85, rel=1e-12)
    assert a.weights[0] > 0.85 and a.weights[2] == 0.0 and a.weights[3] == 0.0
    t = a.at_velocity(0.5, 1.0)
    for s0, s1 in zip(a.solutes, t.solutes):
        assert abs(1.0 - s1.k) < abs(1.0 - s0.k)        # trapping moves every k towards 1
    assert t.freezing_range < a.freezing_range
    with pytest.raises(ValueError):
        P3.MultiAlloy((P3.Solute("X", 1.0, 1.0, -1.0),))


def test_rotation_matrix_is_orthonormal_and_identity_at_zero():
    R = P3.rotation_matrix((33.0, 47.0, 12.0))
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(P3.rotation_matrix((0.0, 0.0, 0.0)), np.eye(3))


def test_checkpoint_round_trip(tmp_path):
    m = _in625_model("numba" if PF.numba_available() else "array")
    m.run(30 * m.dt * m.tau0, record_every=10)
    m._shift(4)                                      # store some solidified rows as well
    p = str(tmp_path / "ck.npz")
    m.checkpoint(p)
    m2 = _in625_model(m.kernel, seed=99)
    m2.restore(p)
    for _ in range(10):
        m.step()
        m2.step()
    assert np.array_equal(m.phi, m2.phi) and np.array_equal(m.U, m2.U)
    r, r2 = m.result(), m2.result()
    assert np.array_equal(r.solid["Nb"], r2.solid["Nb"]) and r.front_history == r2.front_history


def test_cells_counted_with_periodic_cores():
    """A synthetic hexagonal pattern of Nb-lean cores, shifted so cores straddle the edges."""
    n, lam = 96, 16
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    nb = np.full((n, n), 6.0)
    centres = [((i + 0.5 * (j % 2)) * lam + 5, j * lam + 3)
               for i in range(n // lam) for j in range(n // lam)]
    for cx, cy in centres:
        dxp = (xx - cx + n / 2) % n - n / 2
        dyp = (yy - cy + n / 2) % n - n / 2
        nb -= 3.0 * np.exp(-(dxp ** 2 + dyp ** 2) / (2 * (lam / 4.0) ** 2))
    phi = np.ones((4, n, n))
    comp = {"Nb": np.repeat(nb[None], 4, axis=0)}
    r = P3.PF3DResult(phi, comp, {}, 1e-8, 0.0, 0, 0.0, params=dict(solutes=[dict(name="Nb", k=0.5)]))
    lab, ncell, spacing = r.cells(1)
    assert ncell == len(centres)
    assert spacing == pytest.approx(lam * 1e-8, rel=1e-9)


def test_cells_counted_when_walls_are_weak_next_to_rich_junctions():
    """Voronoi cells whose walls are barely enriched while the triple junctions are strongly
    enriched (as in the case-B run): a global threshold merges all cores into one; the
    watershed finds every cell."""
    n = 120
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, n, size=(30, 2))
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    d = []
    for py, px in pts:
        dy = (yy - py + n / 2) % n - n / 2
        dx_ = (xx - px + n / 2) % n - n / 2
        d.append(np.hypot(dy, dx_))
    d = np.sort(np.array(d), axis=0)
    nb = 3.0 + 0.6 * np.exp(-((d[1] - d[0]) / 1.5) ** 2) + 4.0 * np.exp(-((d[2] - d[0]) / 2.0) ** 2)
    phi = np.ones((3, n, n))
    r = P3.PF3DResult(phi, {"Nb": np.repeat(nb[None], 3, axis=0)}, {}, 1e-8, 0.0, 0, 0.0,
                      params=dict(solutes=[dict(name="Nb", k=0.5)]))
    thr = 0.5 * (np.percentile(nb, 5) + np.percentile(nb, 99))
    from scipy import ndimage
    assert ndimage.label(nb < thr)[1] <= 3                    # the cores percolate under a global threshold
    _, ncell, _ = r.cells(1)
    assert abs(ncell - 30) <= 1
