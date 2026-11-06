"""Tabulated / temperature-dependent plasticity, thermal strains and element birth
at temperature, checked against closed-form solutions."""
import numpy as np
import pytest
from scipy.optimize import brentq

import tpmsfem as tf
from tpmsfem.fem.materials import FlowTable, TemperatureTable
from tpmsfem.geometry import box
from tpmsfem.mesh import voxelize


def _cube(h=0.5):
    return voxelize(box((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), h)


def _symmetry(md, m, axial=True):
    """Symmetry planes x=0, y=0, z=0 (homogeneous uniaxial states)."""
    x, y, z = m.points.T
    md.fix(np.flatnonzero(x < 1e-9), (0,))
    md.fix(np.flatnonzero(y < 1e-9), (1,))
    md.fix(np.flatnonzero(z < 1e-9), (2,))
    return np.flatnonzero(z > 1 - 1e-9)


def test_multilinear_uniaxial_matches_curve():
    flow = FlowTable([0.0, 0.01, 0.03], [300.0, 400.0, 450.0], tail_slope=1_000.0)
    mat = tf.Material("tab", 200_000.0, 0.3, flow=flow)
    m = _cube()
    md = tf.SolidModel(m, mat)
    assert md.tabular
    top = _symmetry(md, m)

    def analytic(eps):
        if eps * mat.E <= 300.0:
            return eps * mat.E
        return brentq(lambda s: s / mat.E + _inv(s) - eps, 300.0, 5_000.0)

    def _inv(s):  # plastic strain on the curve for stress s
        if s <= 450.0:
            return np.interp(s, [300.0, 400.0, 450.0], [0.0, 0.01, 0.03])
        return 0.03 + (s - 450.0) / 1_000.0

    for eps in (0.001, 0.003, 0.008, 0.02, 0.05):     # crosses every segment and the tail
        md.fix(top, (2,), eps)
        md.solve(n_inc=3, solver="direct")
        assert md.stress()[..., 2].mean() == pytest.approx(analytic(eps), rel=1e-8)
        assert np.ptp(md.stress()[..., 2]) < 1e-6 * analytic(eps)


def test_one_point_table_equals_linear_hardening():
    lin = tf.Material("lin", 200_000.0, 0.3, yield_stress=300.0, hardening=5_000.0)
    tab = tf.Material("tab", 200_000.0, 0.3, flow=FlowTable([0.0], [300.0], tail_slope=5_000.0))
    m = _cube()
    out = []
    for mat in (lin, tab):
        md = tf.SolidModel(m, mat)
        top = _symmetry(md, m)
        hist = []
        for eps in (0.004, 0.01, -0.006, 0.002):                     # cyclic path
            md.fix(top, (2,), eps)
            md.solve(n_inc=4, solver="direct")
            hist.append(md.stress()[..., 2].mean())
        out.append(np.array(hist))
    assert np.allclose(out[0], out[1], rtol=1e-10, atol=1e-8)


def _thermo_material(**kw):
    temp = TemperatureTable(T=[20.0, 600.0], E=[200_000.0, 150_000.0], alpha=[1.2e-5, 1.2e-5], T_ref=20.0)
    flow = FlowTable([0.0], [[400.0], [100.0]], T=[20.0, 600.0])        # perfectly plastic, sigma_y(T)
    return tf.Material("thermo", 200_000.0, 0.3, flow=flow, temperature=temp, **kw)


def test_free_thermal_expansion_is_stress_free():
    mat = _thermo_material()
    m = _cube()
    md = tf.SolidModel(m, mat)
    top = _symmetry(md, m)
    md.set_temperature(500.0)
    md.solve(n_inc=2, solver="direct")
    assert np.abs(md.stress()).max() < 1e-8
    assert md.displacement()[top, 2].mean() == pytest.approx(1.2e-5 * 480.0, rel=1e-10)


def test_constrained_heating_yields_then_cools_elastically():
    """Bar clamped axially, free laterally: on heating sigma = -E(T) eps_th
    until the temperature-dependent yield stress is reached; on cooling it
    unloads elastically into tension (total formulation, C = C(T))."""
    mat = _thermo_material()
    m = _cube()
    md = tf.SolidModel(m, mat)
    top = _symmetry(md, m)
    md.fix(top, (2,), 0.0)
    md.set_temperature(600.0)
    md.solve(n_inc=6, solver="direct")
    assert md.stress()[..., 2].mean() == pytest.approx(-100.0, rel=1e-8)          # = -sigma_y(600)
    eth600 = 1.2e-5 * 580.0
    ep = -eth600 + 100.0 / 150_000.0                                               # axial plastic strain
    assert md.plastic_strain()[0][..., 2].mean() == pytest.approx(ep, rel=1e-8)
    md.set_temperature(500.0)
    md.solve(n_inc=4, solver="direct")
    E500 = 200_000.0 - 50_000.0 * 480.0 / 580.0
    assert md.stress()[..., 2].mean() == pytest.approx(E500 * (-1.2e-5 * 480.0 - ep), rel=1e-8)


def test_birth_at_deposition_temperature_and_cooling():
    """An element born at 800 C is stress-free there; free cooling keeps it
    stress-free and shrinks it by eps_th(800) - eps_th(20); the thermal strain
    is frozen above T_cap."""
    temp = TemperatureTable(T=[20.0, 900.0], E=[200_000.0, 120_000.0], alpha=[1.2e-5, 1.6e-5], T_ref=20.0,
                            T_cap=700.0)
    mat = tf.Material("dep", 200_000.0, 0.3, temperature=temp)
    m = _cube()
    md = tf.SolidModel(m, mat)
    top = _symmetry(md, m)
    md.deactivate(np.arange(m.n_cells))
    md.set_temperature(800.0, ramp=False)
    md.activate(np.arange(m.n_cells))
    md.solve(solver="direct")
    assert np.abs(md.stress()).max() < 1e-8 and np.abs(md.displacement()).max() < 1e-14
    md.set_temperature(20.0)
    md.solve(n_inc=3, solver="direct")
    a700 = 1.2e-5 + 0.4e-5 * 680.0 / 880.0
    assert np.abs(md.stress()).max() < 1e-6
    assert md.displacement()[top, 2].mean() == pytest.approx(-a700 * 680.0, rel=1e-10)


def test_anneal_resets_hardening():
    temp = TemperatureTable(T=[20.0, 1000.0], E=[200_000.0, 200_000.0], alpha=[0.0, 0.0])
    mat = tf.Material("ann", 200_000.0, 0.3, flow=FlowTable([0.0], [300.0], tail_slope=5_000.0),
                      temperature=temp, anneal_temperature=800.0)
    m = _cube()
    md = tf.SolidModel(m, mat)
    top = _symmetry(md, m)
    md.fix(top, (2,), 0.01)
    md.solve(n_inc=2, solver="direct")
    assert md.plastic_strain()[1].min() > 0.008
    md.release(top, (2,))
    md.set_temperature(900.0)
    md.solve(solver="direct")
    assert np.abs(md.plastic_strain()[1]).max() == 0.0          # hardening memory lost
    assert np.abs(md.plastic_strain()[0][..., 2].mean()) > 0.008  # permanent strain kept


def test_thermomechanical_layer_build_small_wall():
    """4 super-layers on an elastic-plastic plate: the process runs end to end,
    leaves the classic residual-stress pattern (top in longitudinal tension)
    and a self-equilibrated state after removal from the plate."""
    from tpmsfem.fem.materials import ThermalProperties
    from tpmsfem.residual.thermomech import ThermoMechanicalAM
    from tpmsfem.thermal import HeatModel

    h = 0.25
    wall = box((0, 0, 0), (4, 1, 1))
    plate = box((-0.5, -0.5, -0.5), (4.5, 1.5, 0.0))
    m = voxelize(wall | plate, (-0.5, -0.5, -0.5), (4.5, 1.5, 1.0), h, material_fields=[(1, wall), (2, plate)])
    temp = TemperatureTable(T=[20.0, 900.0], E=[200_000.0, 140_000.0], alpha=[1.3e-5, 1.6e-5], T_cap=1290.0)
    flow = FlowTable([0.0, 0.05], [[700.0, 900.0], [400.0, 500.0]], T=[20.0, 900.0])
    th = ThermalProperties.from_si([20.0, 900.0], [10.0, 22.0], [420.0, 620.0], 8440.0, 2.7e5, 1290.0, 1350.0)
    mat = tf.Material("in", 200_000.0, 0.3, flow=flow, temperature=temp, thermal=th)
    md = tf.SolidModel(m, mat, T_initial=20.0)
    hm = HeatModel(m, mat, T_initial=20.0)
    sub = np.flatnonzero(m.cell_data["material"] == 2)
    tm = ThermoMechanicalAM(md, hm, layer_height=h, T_deposit=1350.0, layer_time=20.0, n_steps=3,
                            first_step=0.05, substrate=sub, solver="direct")
    tm.build(verbose=False)
    assert len(tm.records) == 4 and tm.records[0].T_max_after_first_step < 1350.0
    tm.cool_down(time_total=600.0, n_steps=3, verbose=False)
    assert np.allclose(md.temperature(), 20.0)
    tm.remove_from_plate(verbose=False)
    s = md.stress().mean(axis=1)
    c = m.centroids()
    top = (m.cell_data["material"] == 1) & (c[:, 2] > 0.75) & (np.abs(c[:, 0] - 2.0) < 1.0)
    assert s[top, 0].mean() > 100.0                                     # tensile top layer
    assert np.abs(md.reactions()).max() < 1e-6 * np.abs(s).max()      # 3-2-1 support: no reactions
    sec = (np.abs(c[:, 0] - 2.125) < 1e-9)                              # a full cross-section x = 2.125
    assert abs(s[sec, 0].sum()) * h * h < 1e-2 * np.abs(s[sec, 0]).sum() * h * h


def test_strain_life_round_trip_and_corrections():
    from tpmsfem.post.fatigue import IN625_LPBF_CONCLI2024 as F

    N = np.array([1e2, 1e3, 1e4, 1e5])
    e = F.strain_from_life(N)
    assert np.allclose(F.life(e), N, rtol=1e-8)
    assert np.all(F.life(e, sigma_m=200.0, correction="morrow") < N)     # tensile mean stress shortens life
    assert np.all(F.life(e, sigma_m=-200.0, correction="morrow") > N)
    s = np.array([200.0, 600.0, 900.0])
    assert np.allclose(F.stress_amplitude(F.strain_amplitude(s)), s, rtol=1e-9)
    sn, en = F.neuber(np.array([300.0, 1500.0]))
    assert np.allclose(sn * en, np.array([300.0, 1500.0]) ** 2 / F.E, rtol=1e-9)
    assert sn[1] < 1500.0                                                   # plasticity caps the notch stress


def test_abaqus_export_of_tabulated_thermal_material(tmp_path):
    from tpmsfem.fem.materials import ThermalProperties
    from tpmsfem.io.abaqus import write_inp

    th = ThermalProperties.from_si([20.0, 900.0], [10.0, 22.0], [420.0, 620.0], 8440.0, 2.7e5, 1290.0, 1350.0)
    mat = _thermo_material(thermal=th, anneal_temperature=1000.0)
    md = tf.SolidModel(_cube(), mat)
    _symmetry(md, md.mesh)
    txt = open(write_inp(md, str(tmp_path / "t.inp"))).read()
    for card in ("*ELASTIC", "*EXPANSION, ZERO=20", "*PLASTIC", "*ANNEAL TEMPERATURE", "*CONDUCTIVITY",
                 "*SPECIFIC HEAT", "*DENSITY", "*LATENT HEAT"):
        assert card in txt, card
    assert "1.000000000e+02, 0.000000000e+00, 600" in txt          # sigma_y(600 C) row with temperature


def test_gmg_stays_convergent_when_moduli_stiffen_on_cooling():
    """The multigrid smoother is set up while hot (low E); cooling stiffens the
    material by 2x -- the smoother must be refreshed or PCG stalls."""
    temp = TemperatureTable(T=[20.0, 900.0], E=[200_000.0, 100_000.0], alpha=[1.2e-5, 1.2e-5])
    mat = tf.Material("soft_hot", 200_000.0, 0.3, temperature=temp)
    m = voxelize(box((0, 0, 0), (10, 10, 8)), (0, 0, 0), (10, 10, 8), 0.5)
    md = tf.SolidModel(m, mat, T_initial=900.0)
    assert md.ndof > 20_000                                     # multigrid with >= 2 levels
    md.fix(np.flatnonzero(m.points[:, 2] < 1e-9))
    md.set_temperature(880.0)
    md.solve(solver="gmg")
    md.set_temperature(20.0 + 10.0 * m.points[:, 2])            # cool, with a gradient
    info = md.solve(solver="gmg")
    assert info.converged and info.krylov_iterations < 150
