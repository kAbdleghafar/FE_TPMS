"""Porosity criteria (tpmsfem.process.defects)."""
import numpy as np
import pytest

from tpmsfem.process.defects import keyhole_number, lof_criterion, lof_porosity, volumetric_energy_density


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
