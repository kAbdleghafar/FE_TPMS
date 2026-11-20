"""Porosity criteria (tpmsfem.process.defects)."""
import numpy as np
import pytest

from tpmsfem.process.defects import (gan_porosity, keyhole_absorptivity, keyhole_aspect_ratio, keyhole_depth,
                                     keyhole_power,
                                     keyhole_fluctuation, keyhole_number, keyhole_porosity_number, keyhole_regime,
                                     keyhole_stability, keyhole_state, lof_criterion, lof_porosity,
                                     normalized_diffusion_length, normalized_energy_density, volumetric_energy_density)

# Gan et al. (2021) Ti-6Al-4V: liquid properties at the liquidus, T_l - T0 = 1630 K (Supplementary Data 1)
TI64 = dict(rho=3920.0, cp=830.0, T_l=1933.0, T0=303.0)
TI64_ALPHA = 33.4 / (3920.0 * 830.0)


def test_lof_criterion():
    assert lof_criterion(100.0, 60.0, 50.0, 30.0) == pytest.approx(0.25 + 0.25)
    assert lof_criterion(50.0, 60.0, 50.0, 30.0) > 1.0                  # hatch = width: no side overlap


def test_lof_porosity_limits():
    # deep, wide pools fuse everything
    assert lof_porosity(200.0, 120.0, 50.0, 30.0) == 0.0
    # very wide, shallow pools: only the strip below the pool depth stays unfused
    assert lof_porosity(1e5, 20.0, 50.0, 30.0, n=300) == pytest.approx(10.0 / 30.0, abs=0.01)
    # pools narrower than the hatch leave gaps even when deep enough
    assert lof_porosity(40.0, 80.0, 50.0, 30.0) > 0.05
    # monotone in hatch
    p = [lof_porosity(80.0, 45.0, h, 30.0) for h in (40.0, 60.0, 80.0, 100.0)]
    assert all(b >= a for a, b in zip(p, p[1:]))
    # criterion and porosity agree on which side of the limit a case falls (no scatter)
    for W, D, h, L in ((90.0, 50.0, 50.0, 30.0), (60.0, 32.0, 60.0, 30.0)):
        assert (lof_criterion(W, D, h, L) <= 1.0) == (lof_porosity(W, D, h, L, n=300) < 0.01)


def test_keyhole_number_scaling():
    base = dict(absorptivity=0.35, radius=40e-6, rho=8000.0, cp=600.0, T_l=1400.0, T0=25.0, alpha=5e-6)
    k1 = keyhole_number(200.0, 1.0, **base)
    assert 1.0 < k1 < 20.0                                              # a typical LPBF track
    assert keyhole_number(400.0, 1.0, **base) == pytest.approx(2.0 * k1)
    assert keyhole_number(200.0, 4.0, **base) == pytest.approx(0.5 * k1)
    assert volumetric_energy_density(200.0, 1.0, 50e-6, 30e-6) / 1e9 == pytest.approx(133.33, rel=1e-3)


def test_keyhole_depth_law():
    assert keyhole_aspect_ratio(1.0) == 0.0 and keyhole_aspect_ratio(1.4) == 0.0
    assert keyhole_aspect_ratio(6.0) == pytest.approx(0.4 * 4.6)
    assert keyhole_depth(11.4, 50e-6) == pytest.approx(4.0 * 50e-6)
    assert keyhole_fluctuation(10.0) == pytest.approx(0.36 * 10.0 ** 0.86)
    assert keyhole_fluctuation(1.0) == 0.0
    assert list(keyhole_regime([1.0, 1.4, 3.0, 6.0, 7.0])) == ["conduction", "conduction", "transition", "transition",
                                                              "keyhole"]
    assert keyhole_regime(20.0) == "keyhole"
    assert list(keyhole_stability([5.0, 16.0, 20.0, 31.0])) == ["stable", "stable", "intermediate", "chaotic"]


def test_keyhole_absorptivity():
    # the diffusivity cancels in Ke_m L_d*, and Ke_m L_d* = Ke(eta_m) * L_d*
    P, v, r0 = 200.0, 0.6, 47.5e-6
    kl = keyhole_porosity_number(P, v, r0, 0.26, **TI64)
    ke_m = keyhole_number(P, v, 0.26, r0, alpha=TI64_ALPHA, **TI64)
    assert kl == pytest.approx(ke_m * normalized_diffusion_length(v, r0, TI64_ALPHA))
    # Eq. 6 tends to 0.7 at high power and is floored at eta_m at low power
    assert keyhole_absorptivity(1e5, v, r0, 0.26, **TI64) == pytest.approx(0.7)
    assert keyhole_absorptivity(1.0, v, r0, 0.26, **TI64) == pytest.approx(0.26)
    assert keyhole_absorptivity(1.0, v, r0, 0.26, floor=False, **TI64) < 0.01
    # Supplementary Data 1 of Gan et al.: their eta column is Eq. 6 with an effective eta_m of 0.2424
    for P, v, r0, eta in ((103.198, 0.4, 47.5e-6, 0.44205), (468.023, 0.6, 47.5e-6, 0.66578),
                          (207.849, 0.7, 70e-6, 0.28758)):
        assert keyhole_absorptivity(P, v, r0, 0.2424, **TI64) == pytest.approx(eta, abs=2e-4)


def test_keyhole_state_against_xray():
    # three Ti-6Al-4V keyholes of Gan et al. Supplementary Data 1 (power, speed, r0, eta, depth [um])
    rows = ((155.523, 0.4, 47.5e-6, 0.5445, 130.4), (363.372, 0.6, 47.5e-6, 0.6328, 310.9),
            (312.5, 1.0, 47.5e-6, 0.4911, 174.2))
    for P, v, r0, eta, e in rows:
        st = keyhole_state(P, v, r0, alpha=TI64_ALPHA, eta_min=0.26, absorptivity=eta, **TI64)
        assert st["regime"] == "keyhole"
        assert 1e6 * st["depth"] == pytest.approx(e, rel=0.25)                    # scatter of the data: RMSE 18 um
        assert st["depth_min"] < st["depth"] < st["depth_max"]
    st = keyhole_state([100.0, 300.0], [[0.5], [1.0]], 50e-6, alpha=TI64_ALPHA, eta_min=0.26, **TI64)
    assert st["Ke"].shape == (2, 2) and st["regime"].shape == (2, 2)
    assert st["Ke"][0, 1] > st["Ke"][0, 0] and st["Ke"][0, 0] > st["Ke"][1, 0]     # up with P, down with v


def test_gan_porosity_terms():
    ned = normalized_energy_density(200.0, 1.0, 100e-6, 30e-6, 0.26, **TI64)
    assert ned == pytest.approx(0.26 * 200.0 / (1.0 * 100e-6 * 30e-6) / (3920.0 * 830.0 * 1630.0))
    assert gan_porosity(50.0, 2.32) == pytest.approx(0.055)                        # both terms at rest
    assert gan_porosity(1.51, -10.0) == pytest.approx(0.10 - 0.01 + 0.055)
    assert gan_porosity(50.0, 10.0) > gan_porosity(50.0, 1.0)                      # keyhole pores
    assert gan_porosity(0.5, 1.0) > gan_porosity(5.0, 1.0)                         # lack of fusion


def test_keyhole_power_inverts_ke():
    kw = dict(eta_min=0.26, alpha=TI64_ALPHA, **TI64)
    st = keyhole_state(200.0, 1.0, 25e-6, **kw)
    P = keyhole_power(float(st["Ke"]), 1.0, 25e-6, **kw)
    assert P == pytest.approx(200.0, rel=1e-9)
    # slowing down at constant Ke needs less power, roughly with sqrt(v) (less as eta saturates)
    P_slow = keyhole_power(float(st["Ke"]), 0.25, 25e-6, **kw)
    assert P_slow < 200.0 * np.sqrt(0.25) * 1.05
    assert float(keyhole_state(P_slow, 0.25, 25e-6, **kw)["Ke"]) == pytest.approx(float(st["Ke"]), rel=1e-9)
