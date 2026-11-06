"""Heat solver checks against closed-form solutions and energy balance."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.geometry import box
from tpmsfem.mesh import voxelize
from tpmsfem.thermal import HeatModel, geometric_steps


def _mat(k=10.0, cp=500.0, rho=8000.0, latent=0.0, Ts=np.inf, Tl=np.inf):
    th = ThermalProperties.from_si([0.0, 2000.0], [k, k], [cp, cp], rho, latent, Ts, Tl)
    return tf.Material("th", 200_000.0, 0.3, thermal=th)


def test_semi_infinite_slab_fourier_series():
    """Bar 0 <= x <= L, T(0) = 100, other faces insulated, T0 = 0."""
    L, h = 10.0, 0.25
    m = voxelize(box((0, 0, 0), (L, h, h)), (0, 0, 0), (L, h, h), h)
    mat = _mat()
    hm = HeatModel(m, mat, T_initial=0.0)
    hm.fix(np.flatnonzero(m.points[:, 0] < 1e-9), 100.0)
    a = 10.0 / (8000e-12 * 500e6)                         # diffusivity [mm^2/s]
    t_end, n = 20.0, 400
    for _ in range(n):
        hm.step(t_end / n)
    x = m.points[:, 0]
    lam = (2 * np.arange(200) + 1) * np.pi / (2 * L)
    exact = 100.0 * (1 - (4 / ((2 * np.arange(200) + 1) * np.pi) * np.sin(np.outer(x, lam))
                          * np.exp(-lam**2 * a * t_end)).sum(axis=1))
    assert np.abs(hm.T - exact).max() < 1.0               # 1 % of the range (h = 0.25, dt = 0.05 s)


def test_latent_heat_is_conserved():
    """Insulated block, half liquid at 1500 C, half at 20 C: total enthalpy is
    conserved exactly and the final uniform temperature matches the enthalpy
    balance (with the latent heat released during solidification)."""
    m = voxelize(box((0, 0, 0), (2, 1, 1)), (0, 0, 0), (2, 1, 1), 0.25)
    mat = _mat(latent=2.7e5, Ts=1290.0, Tl=1350.0)
    hm = HeatModel(m, mat, T_initial=20.0)
    hm.T[m.points[:, 0] > 1.0 + 1e-9] = 1500.0
    H0 = hm.enthalpy()
    for dt in geometric_steps(200.0, 0.01, 30):
        hm.step(dt)
    assert hm.enthalpy() == pytest.approx(H0, rel=1e-9)
    assert np.ptp(hm.T) < 1e-3
    tt, HH = mat.thermal.enthalpy_table()
    Vn = np.zeros(m.n_points)
    np.add.at(Vn, m.cells.ravel(), np.full(m.cells.size, 0.25**3 / 8))
    assert hm.T.mean() == pytest.approx(np.interp(H0 / Vn.sum(), HH, tt), abs=1e-3)


def test_newton_cooling_lumped_capacitance_and_face_filter():
    m = voxelize(box((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), 0.25)
    mat = _mat(k=1e4)                                     # Biot << 1: uniform temperature
    rc = 8000e-12 * 500e6
    for only_top, area in ((False, 6.0), (True, 1.0)):
        hm = HeatModel(m, mat, T_initial=500.0)
        filt = (lambda c, n: n[:, 2] > 0.5) if only_top else None
        hm.set_surface_losses(h=0.02, T_amb=20.0, face_filter=filt)
        t_end, n = 100.0, 2000
        for _ in range(n):
            hm.step(t_end / n)
        exact = 20.0 + 480.0 * np.exp(-0.02 * area * t_end / (rc * 1.0))
        assert hm.T.mean() == pytest.approx(exact, rel=2e-3)


def test_birth_sets_deposition_temperature_on_new_nodes_only():
    m = voxelize(box((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), 0.5)
    hm = HeatModel(m, _mat(), T_initial=20.0)
    z = m.centroids()[:, 2]
    hm.deactivate(np.flatnonzero(z > 0.5))
    hm.activate(np.flatnonzero(z > 0.5), T_new=1350.0)
    zp = m.points[:, 2]
    assert np.all(hm.T[zp > 0.75] == 1350.0) and np.all(hm.T[zp < 0.75] == 20.0)
