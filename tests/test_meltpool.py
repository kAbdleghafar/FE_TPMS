"""Melt-pool solver: exact moving-source solution, energy balance, solidification records."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.process import Laser, MeltPoolModel, ScanVector
from tpmsfem.process.meltpool import fusion_zone, moving_gaussian_analytic, pool_length


def _mat(latent=0.0, Ts=1e9, Tl=1e9 + 1):
    th = ThermalProperties.from_si([0.0, 5000.0], [20.0, 20.0], [500.0, 500.0], 8000.0, latent, Ts, Tl)
    return tf.Material("const", 200_000.0, 0.3, thermal=th)


def test_moving_gaussian_matches_eagar_tsai():
    """Constant properties, no phase change: quasi-steady field under a moving Gaussian
    source equals the Eagar-Tsai integral (semi-infinite body)."""
    mat = _mat()
    h, v, w, P, A = 0.01, 500.0, 0.1, 100.0, 0.5
    md = MeltPoolModel(mat, h, length=2.4, width=0.6, depth=0.6, emissivity=0.0, h_conv=0.0, T0=20.0)
    laser = Laser(P, A, w)
    res = md.run(laser, [ScanVector(0.3, 0.0, 1.9, 0.0, v)])
    k, rc = 20.0, 8000e-12 * 500e6
    x_beam = 1.9
    for dx, yy, zz in ((0.0, 0.005, -0.005), (-0.1, 0.005, -0.005), (-0.3, 0.005, -0.055), (0.0, 0.105, -0.025)):
        i = int(np.argmin(np.abs(res.x - (x_beam + dx))))
        j = int(np.argmin(np.abs(res.y - yy)))
        kk = int(np.argmin(np.abs(res.z - zz)))
        ref = moving_gaussian_analytic(res.x[i] - x_beam, res.y[j], res.z[kk], A * P * 1e3, v, w, k, rc)
        assert res.T[i, j, kk] - 20.0 == pytest.approx(ref - 20.0, rel=0.03), (dx, yy, zz)
    assert abs(res.energy_balance) < 1e-9


def test_latent_heat_energy_balance_and_solidification_records():
    th = ThermalProperties.from_si([20.0, 870.0], [9.9, 22.8], [410.0, 620.0], 8440.0, 2.72e5, 1290.0, 1350.0)
    mat = tf.Material("in625", 200_000.0, 0.3, thermal=th)
    md = MeltPoolModel(mat, 0.005, length=0.8, width=0.2, depth=0.15, emissivity=0.4)
    res = md.run(Laser(180.0, 0.4, 0.085), [ScanVector(0.1, 0.0, 0.6, 0.0, 800.0)], t_extra=2e-4,
                 probes={"c": (0.4, 0.0)})
    assert abs(res.energy_balance) < 1e-9                     # enthalpy gain = absorbed - radiated
    melted = res.n_melt > 0
    assert melted.any()
    w, d = fusion_zone(res, 0.4)
    assert 0.02 < d < w < 0.3
    ok = melted & np.isfinite(res.t_solid)
    assert np.all(res.t_solid[ok] > res.t_melt[ok])
    assert np.all(res.G[ok] > 0) and np.all(res.cooling_rate[ok] > 0)
    assert np.nanmax(res.probes["c"]) > 1290.0


def test_evaporation_flux_at_boiling_point_and_volumetric_source_energy():
    from tpmsfem.process import Evaporation

    e = Evaporation()
    Tk = e.T_boil + 273.15
    mdot = (1 - e.beta_r) * e.p0 * np.sqrt(e.M / (2 * np.pi * 8.314462618 * Tk))
    assert e.flux(e.T_boil) == pytest.approx(mdot * e.L_v * 1e-3, rel=1e-12)   # p_sat = 1 atm at T_boil
    assert e.flux(e.T_boil + 300) > 3 * e.flux(e.T_boil)
    th = ThermalProperties.from_si([20.0, 870.0], [9.9, 22.8], [410.0, 620.0], 8440.0, 2.72e5, 1290.0, 1350.0)
    mat = tf.Material("in625", 2e5, 0.3, thermal=th)
    md = MeltPoolModel(mat, 0.005, length=0.5, width=0.15, depth=0.12, emissivity=0.4, evaporation=e)
    res = md.run(Laser(180.0, 0.3, 0.085, depth=0.02), [ScanVector(0.1, 0.0, 0.4, 0.0, 800.0)])
    assert res.absorbed_energy == pytest.approx(0.3 * 180.0 * 1e3 * 0.3 / 800.0, rel=1e-9)   # A P t [mJ]
    assert abs(res.energy_balance) < 1e-9
    assert res.T_peak.max() < 4000.0                                          # capped by evaporation
