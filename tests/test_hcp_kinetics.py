"""Strain-induced HCP kinetics (tpmsfem.post.hcp_kinetics)."""
import numpy as np
import pytest

from tpmsfem.post import hcp_kinetics as HK


@pytest.mark.parametrize("shape", ["exponential", "sigmoidal"])
def test_fit_passes_through_the_point_and_starts_at_f0(shape):
    law = HK.StrainInducedHCP.fit(0.2, 0.3, f0=0.01, shape=shape)
    assert law.fraction(0.2) == pytest.approx(0.3, rel=1e-12)
    assert law.fraction(0.0) == pytest.approx(0.01)
    e = np.linspace(0, 2, 200)
    f = law.fraction(e)
    assert np.all(np.diff(f) >= 0) and f[-1] <= 1.0
    with pytest.raises(ValueError):
        HK.StrainInducedHCP.fit(0.2, 0.005, f0=0.01)


def test_sigmoidal_is_slower_at_small_strain_than_exponential():
    a = HK.femncocr_lpbf("exponential")
    b = HK.femncocr_lpbf("sigmoidal")
    assert b.fraction(0.02) < a.fraction(0.02)
    eps_p = 0.2 - 744.9 / 200e3
    assert a.fraction(eps_p) == pytest.approx(0.281) and b.fraction(eps_p) == pytest.approx(0.281)
    assert a.fraction(0.0) == pytest.approx(0.0094)
