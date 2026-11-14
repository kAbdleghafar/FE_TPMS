"""Free-surface lattice Boltzmann melt pool (tpmsfem.process.freesurface)."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.process import Laser, MeltPoolModel, ScanVector
from tpmsfem.process.freesurface import GAS, INTERFACE, LIQUID, WALL, FreeSurfaceLBM, FreeSurfaceMeltPool


def _walls(flag):
    flag[0, :, :] = flag[-1, :, :] = flag[:, 0, :] = flag[:, -1, :] = flag[:, :, 0] = flag[:, :, -1] = WALL
    return flag


def _droplet(n=24, r=6.0):
    g = np.arange(n) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    fill = np.clip(r + 0.5 - np.sqrt((X - n / 2) ** 2 + (Y - n / 2) ** 2 + (Z - n / 2) ** 2), 0.0, 1.0)
    flag = np.where(fill >= 1.0, LIQUID, np.where(fill > 0.0, INTERFACE, GAS)).astype(np.int8)
    return _walls(flag), fill


def test_laplace_pressure_and_mass():
    flag, fill = _droplet()
    sigma = 0.01
    s = FreeSurfaceLBM(flag, fill, tau=0.8, sigma=sigma)
    m0 = s.total_mass()
    for _ in range(300):
        s.step()
    R = (3.0 * s.total_mass() / (4.0 * np.pi)) ** (1.0 / 3.0)
    core = float(np.asarray(s.rho)[10:14, 10:14, 10:14].mean())
    assert core - 1.0 == pytest.approx(2.0 * sigma / R * 3.0, rel=0.08)       # Laplace: dp = 2 sigma / R
    assert abs(s.total_mass() - m0) / m0 < 1e-12
    assert float(np.sqrt((np.asarray(s.u) ** 2).sum(axis=0)).max()) < 1e-3   # spurious currents


def test_falling_column_conserves_mass_and_moves_the_interface():
    n = (12, 12, 24)
    flag = _walls(np.full(n, GAS, np.int8))
    flag[1:-1, 1:6, 1:16] = LIQUID                     # a liquid column against one wall collapses
    flag[1:-1, 6, 1:16] = INTERFACE
    flag[1:-1, 1:6, 16] = INTERFACE
    fill = np.where(flag != GAS, 1.0, 0.0)
    fill[flag == WALL] = 0.0
    s = FreeSurfaceLBM(flag, fill, tau=0.6, gravity=(0.0, 0.0, -2e-4))
    m0 = s.total_mass()
    for _ in range(400):
        s.step()
    assert abs(s.total_mass() - m0) / m0 < 1e-12
    f = np.asarray(s.fill)
    assert f[:, 8:11, 1:4].sum() > 1.0                 # liquid has spread along the floor


def _const_material():
    th = ThermalProperties.from_si([0.0, 20000.0], [25.0, 25.0], [650.0, 650.0], 8440.0, 2.72e5, 1290.0, 1350.0)
    return tf.Material("const", 200_000.0, 0.3, thermal=th)


def test_heat_matches_conduction_model():
    """Flow frozen: the free-surface model's heat solver against MeltPoolModel on a short track."""
    h, nx, ny, nz = 10e-6, 30, 16, 10
    metal = np.zeros((nx + 2, ny + 2, nz + 4), bool)
    metal[1:-1, 1:-1, 1:nz + 1] = True
    md = FreeSurfaceMeltPool(_const_material(), h, metal, dt=1e-6, T0=25.0, T_max=20000.0)
    y = h + 0.5 * ny * h
    laser = Laser(150.0, 0.35, 40e-6, depth=15e-6)
    res = md.run(laser, [ScanVector(h + 50e-6, y, h + 250e-6, y, 0.8)], frozen_flow=True)
    assert abs(res.energy_balance) < 1e-10
    mp = MeltPoolModel(_const_material(), h * 1e3, length=nx * h * 1e3, width=ny * h * 1e3, depth=nz * h * 1e3,
                       symmetric_y=False, T0=25.0, emissivity=0.0, h_conv=0.0, T_max=20000.0)
    r2 = mp.run(Laser(150.0, 0.35, 0.04, depth=0.015), [ScanVector(0.05, 0.0, 0.25, 0.0, 800.0)])
    T_lbm, T_fd = res.T[1:-1, 1:-1, 1:nz + 1], np.asarray(r2.T)
    hot = T_fd > 200.0
    # different time steps (1e-6 vs the conduction model's own): agree to a few percent
    assert np.sqrt(np.mean((T_lbm - T_fd)[hot] ** 2)) / np.mean(T_fd[hot] - 25.0) < 0.03


def test_surface_shear_gives_couette_profile():
    """A tangential stress on the interface cells of a periodic liquid layer on a no-slip
    floor: steady linear profile u = tau z / mu (free-surface stress boundary)."""
    n = (4, 4, 14)
    flag = np.full(n, GAS, np.int8)
    flag[:, :, 0] = flag[:, :, -1] = WALL
    flag[:, :, 1:9] = LIQUID
    flag[:, :, 9] = INTERFACE
    fill = np.where((flag == LIQUID) | (flag == INTERFACE), 1.0, 0.0)
    s = FreeSurfaceLBM(flag, fill, tau=0.8)
    F = np.zeros((3,) + n)
    F[0][:, :, 9] = 1e-6
    s.force = F
    nu = 0.1
    for _ in range(int(6 * 9 * 9 / nu)):
        s.step()
    u = np.asarray(s.u[0])[0, 0, 1:10]
    z = np.arange(9) + 0.5
    np.testing.assert_allclose(u[:-1], 1e-6 * z[:-1] / nu, rtol=0.01)


def test_marangoni_force_and_quiescent_plate():
    """dsigma/dT * grad T on a flat surface: one cell of force per column, along -grad T for
    dsigma/dT < 0; and a plate at rest stays at rest (walls do not act as surfaces)."""
    h = 5e-6
    metal = np.zeros((20, 6, 12), bool)
    metal[1:-1, 1:-1, 1:7] = True
    md = FreeSurfaceMeltPool(_const_material(), h, metal, dt=1e-7, T0=25.0, T_max=4000.0, dsigma_dT=-1e-4)
    x = (np.arange(20) + 0.5) * h
    md.T = md.xp.asarray(np.broadcast_to((1400.0 + 1e6 * x)[:, None, None], md.shape).copy())
    md._surface_terms()
    Fx = np.asarray(md.lbm.force[0]) / (md.dt ** 2 / (md.rho0 * h))          # N/m^3
    col = Fx[10, 3, :]
    assert col[6] == pytest.approx(-1e-4 * 1e6 / h, rel=1e-6)             # surface cell: tau / dx
    assert np.all(col[:6] == 0.0)
    md2 = FreeSurfaceMeltPool(_const_material(), h, metal, dt=1e-7, T0=1400.0, T_max=4000.0, sigma=1.8,
                              gravity=(0.0, 0.0, 0.0))
    m0 = md2.lbm.total_mass()
    for _ in range(50):
        md2._surface_terms()
        md2._couple()
        md2.lbm.step()
    assert float(np.abs(np.asarray(md2.lbm.u)).max()) < 1e-12
    assert abs(md2.lbm.total_mass() - m0) / m0 < 1e-12


def test_sahoo_surface_tension():
    from tpmsfem.process.freesurface import SahooSurfaceTension

    clean = SahooSurfaceTension(a_s=0.0)
    assert clean.dsigma_dT(np.array([1500.0, 2500.0])) == pytest.approx([-4.3e-4, -4.3e-4])
    m = SahooSurfaceTension(a_s=0.01)
    T = np.linspace(1400.0, 3000.0, 9)
    num = (m.sigma(T + 0.01) - m.sigma(T - 0.01)) / 0.02
    np.testing.assert_allclose(m.dsigma_dT(T), num, rtol=1e-5, atol=1e-9)
    Tc = m.zero_crossing()
    assert 1500.0 < Tc < 2000.0
    assert m.dsigma_dT(np.array(Tc - 100.0)) > 0.0 > m.dsigma_dT(np.array(Tc + 100.0))


def test_energy_conserved_with_flow_and_mass_carried_heat():
    """Marangoni flow with the enthalpy carried by the lattice mass flux: the energy error is
    exactly the residual booked by the bounded transport, and that residual is small."""
    h = 10e-6
    metal = np.zeros((24, 14, 12), bool)
    metal[1:-1, 1:-1, 1:7] = True
    md = FreeSurfaceMeltPool(_const_material(), h, metal, dt=2e-7, T0=25.0, T_max=8000.0, dsigma_dT=-1.1e-4,
                             energy_transport="lbm")
    y = h + 6 * h
    res = md.run(Laser(100.0, 0.35, 60e-6, depth=5e-6), [ScanVector(h + 40e-6, y, h + 180e-6, y, 0.8)])
    assert abs(res.mass_error) < 1e-12
    assert abs(res.energy_balance + res.energy_residual) < 1e-10
    assert abs(res.energy_residual) < 1e-2


def test_numba_kernel_matches_array_version():
    """The fused numba step gives the array version's result (to round-off)."""
    from tpmsfem.process import freesurface_kernels as K

    if not K.available():
        pytest.skip("numba not installed")
    flag, fill = _droplet(n=20, r=5.0)
    out = {}
    for kern in ("array", "numba"):
        s = FreeSurfaceLBM(flag, fill, tau=0.7, sigma=0.01, gravity=(0.0, 0.0, -1e-5))
        s.kernel = kern
        s.h = np.where(fill > 0, 1.0 + np.arange(fill.size).reshape(fill.shape) % 7, 0.0)
        for _ in range(60):
            s.step()
        out[kern] = s
    a, b = out["array"], out["numba"]
    assert (np.asarray(a.flag) == np.asarray(b.flag)).all()
    np.testing.assert_allclose(b.mass, a.mass, atol=1e-10)
    np.testing.assert_allclose(b.u, a.u, atol=1e-12)
    np.testing.assert_allclose(b.h, a.h, atol=1e-9)
