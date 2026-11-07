"""2-D quantitative phase field (tpmsfem.process.phasefield)."""
import numpy as np
import pytest

from tpmsfem.process import phasefield as PF

# model alloy with comfortable thin-interface conditions; W0 = 2.5 d0 = 25 nm
MODEL = PF.DiluteAlloy(c_inf=1.0, k=0.3, freezing_range=10.0, D=1e-9, gibbs_thomson=1e-7, eps4=0.0)
W0 = 2.5 * MODEL.d0


def model_run(lD, lT, nx, height_W, alloy=MODEL, **kw):
    """lD = D/V and lT = dT0/G in units of W0."""
    V = alloy.D / (lD * W0)
    G = alloy.freezing_range / (lT * W0)
    return PF.DirectionalPF(alloy, G, V, nx * 0.8 * W0, height_W * W0, **kw)


def perturb(m, amp_W, q):
    x = (np.arange(m.nx) + 0.5) * m.dx
    z = (np.arange(m.nz) + 0.5) * m.dx
    m.phi = -np.tanh((z[:, None] - (m.z0 + amp_W * np.cos(q * x))[None, :]) / np.sqrt(2.0))


def mode_amplitude(m):
    h = PF.front_profile(m.phi, m.dx)
    return 2 * np.abs(np.fft.rfft(h)[1]) / m.nx


def test_alloy_from_slope():
    a = PF.NI5NB_GHOSH2017
    assert a.freezing_range == pytest.approx(56.9, abs=0.1)        # Ghosh et al. (2017): 57 K
    assert a.m == pytest.approx(-10.5)
    assert PF.IN625_NB.freezing_range == pytest.approx(10.5 * 4.0 * 0.52 / 0.48, rel=1e-3)
    assert PF.constitutional_velocity(a, 1e7) == pytest.approx(a.D * 1e7 / a.freezing_range)


@pytest.mark.parametrize("kernel", ["array", "numba"])
def test_solute_conserved_in_closed_box(kernel):
    if kernel == "numba" and not PF.numba_available():
        pytest.skip("numba not installed")
    a = PF.DiluteAlloy(**{**MODEL.__dict__, "eps4": 0.03})
    m = model_run(10, 300, 48, 40, alloy=a, noise=0.3, seed=2, kernel=kernel, tilt_deg=15.0)
    s0 = m.solute()
    m.run(400 * m.dt * m.tau0, shift_at=None, record_every=10 ** 9)
    assert abs(m.solute() / s0 - 1.0) < 1e-12


def test_numba_kernel_matches_array_kernel():
    if not PF.numba_available():
        pytest.skip("numba not installed")
    a = PF.DiluteAlloy(**{**MODEL.__dict__, "eps4": 0.05})
    m0 = model_run(10, 300, 40, 40, alloy=a, noise=0.3, seed=3, tilt_deg=20.0, kernel="array")
    m1 = model_run(10, 300, 40, 40, alloy=a, noise=0.3, seed=3, tilt_deg=20.0, kernel="numba")
    for _ in range(200):
        m0.step()
        m1.step()
    assert np.abs(m0.phi - m1.phi).max() < 1e-12
    assert np.abs(m0.U - m1.U).max() < 1e-12


def test_dispersion_relation_limits():
    a = MODEL
    G = 1e6
    q = np.linspace(1e4, 1e8, 2000)
    stable = PF.mullins_sekerka_rate(a, G, 0.5 * PF.constitutional_velocity(a, G), q)
    assert np.nanmax(stable) < 0                                  # below V_cs: no unstable mode
    unstable = PF.mullins_sekerka_rate(a, G, 20 * PF.constitutional_velocity(a, G), q)
    assert np.nanmax(unstable) > 0
    assert unstable[-1] < 0                                        # capillarity kills short waves
    va = PF.absolute_stability_velocity(a)
    assert np.nanmax(PF.mullins_sekerka_rate(a, G, 1.5 * va, q)) < 0


def test_planar_front_stable_below_constitutional_limit():
    m = model_run(20, 10, 42, 120)                                 # V = 0.5 V_cs
    q = 2 * np.pi / (m.nx * m.dx)
    perturb(m, 1.0, q)
    a0 = mode_amplitude(m)
    m.run(3 * (20 * W0) ** 2 / MODEL.D, record_every=10 ** 9)
    assert mode_amplitude(m) < 0.2 * a0
    h = PF.front_profile(m.phi, m.dx).mean() + m.offset
    theta = (h - m.z0 - m.Vt * m.t) / m.lT
    assert abs(theta) < 0.02                                       # steady planar front at T0


def test_growth_rate_matches_mullins_sekerka():
    m = model_run(20, 400, 42, 160, noise=0.0)                     # V = 20 V_cs, near the fastest mode
    q = 2 * np.pi / (m.nx * m.dx)
    w_ms = float(PF.mullins_sekerka_rate(MODEL, m.G, m.V, q / W0)) * m.tau0
    perturb(m, 0.05, q)
    ts, amps = [], []
    for _ in range(24):
        m.run(3.0 / w_ms / 24 * m.tau0, record_every=10 ** 9)
        ts.append(m.t)
        amps.append(mode_amplitude(m))
    ts, amps = np.array(ts), np.array(amps)
    sel = ts > 1.0 / w_ms
    w_pf = np.polyfit(ts[sel], np.log(amps[sel]), 1)[0]
    assert w_pf == pytest.approx(w_ms, rel=0.05)


def test_unstable_front_forms_enriched_cells():
    a = PF.DiluteAlloy(**{**MODEL.__dict__, "eps4": 0.02})
    m = model_run(10, 300, 100, 80, alloy=a, seed=1)               # V = 30 V_cs
    r = m.run(12 * (10 * W0) ** 2 / a.D, record_every=500)
    lam, n = r.spacing()
    assert 2 <= n <= 8 and np.isfinite(lam)
    tips = r.tips()
    assert tips and all(t["cs"] < 1.0 for t in tips)                # tip solid is depleted, c_s* < c_inf
    assert r.c[r.phi < -0.5].max() > 1.0 / a.k                      # intercellular liquid beyond c_inf / k
    assert len(r.front_history) > 2


def test_front_profile_and_tips_with_liquid_groove_to_the_bottom():
    nz, nx, dx = 60, 40, 1.0
    z = (np.arange(nz) + 0.5)[:, None]
    tipz = np.array([30.0 if (j // 10) % 2 == 0 else 20.0 for j in range(nx)])
    phi = -np.tanh((z - tipz[None, :]) / np.sqrt(2.0))
    phi[:, 9] = -1.0                                             # a liquid groove through the whole window
    h = PF.front_profile(phi, dx)
    assert h[9] == 0.0
    assert h[0] == pytest.approx(30.0, abs=0.05) and h[15] == pytest.approx(20.0, abs=0.05)
    r = PF.PFResult(phi, np.ones_like(phi), np.zeros((0, nx)), dx, 0.0, 0, 0.0, params=dict(W0_m=1.0, lT_m=100.0))
    tips = r.tips()                                              # plateaus at 30 (two of them); 20 is >2 % of l_T lower
    assert len(tips) == 2 and all(t["height_m"] == pytest.approx(30.0, abs=0.05) for t in tips)


def test_boundaries_counted_next_to_strongly_enriched_droplets():
    nz, nx = 40, 120
    x = np.arange(nx)
    c_row = 0.8 + 0.6 * np.exp(-0.5 * (((x % 10) - 5) / 1.0) ** 2)    # 12 boundaries, 0.6 c_inf above the cores
    c_row[55] = 6.0                                                    # an Nb-rich liquid droplet on one of them
    c = np.repeat(c_row[None, :], nz, axis=0)
    phi = np.where(np.arange(nz)[:, None] < 30, 1.0, -1.0) * np.ones((1, nx))
    r = PF.PFResult(phi, c, np.zeros((0, nx)), 1.0, 0.0, 0, 0.0, params=dict(W0_m=1.0, lT_m=100.0))
    assert r.boundaries() == 12
    assert r.spacing()[0] == pytest.approx(10.0)
