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
