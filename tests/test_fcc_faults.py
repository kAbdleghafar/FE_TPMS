"""Stacking-fault extension in FCC crystals (tpmsfem.post.fcc_faults)."""
import numpy as np
import pytest

from tpmsfem.post import fcc_faults as F

I = np.eye(3)[None]


def test_systems_lie_in_their_planes_and_partials_sum_to_full_dislocations():
    n, b, lead, trail = F.shockley_systems()
    assert n.shape == (24, 3)
    for v in (b, lead, trail):
        assert np.abs((n * v).sum(axis=1)).max() < 1e-12
    assert np.allclose(np.cross(b, lead + trail), 0.0)          # b parallel to the sum of its partials
    nt, t = F.twin_systems()
    assert nt.shape == (12, 3) and np.abs((nt * t).sum(axis=1)).max() < 1e-12
    # the three partials of each plane sum to zero (120 deg apart)
    assert np.allclose(t.reshape(4, 3, 3).sum(axis=1), 0.0)


def test_stacking_sense_of_the_leading_partials():
    """For n = [111] the layer above A sits at A + a/6 [1 1 -2] (B): that shift makes the fault."""
    a = 1.0
    above = a / 2 * np.array([1.0, 1.0, 0.0])                    # an atom one {111} layer up
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    shift = above - (above @ n) * n
    nt, t = F.twin_systems()
    first = t[:3]
    assert np.any(np.all(np.isclose(first, shift / np.linalg.norm(shift)), axis=1))


@pytest.mark.parametrize("axis, tension, compression", [((0, 0, 1), 0.236, 0.471), ((1, 1, 1), 0.314, 0.157),
                                                        ((1, 0, 1), 0.471, 0.236)])
def test_leading_partial_schmid_factors_tension_compression_asymmetry(axis, tension, compression):
    """Twinning Schmid factors of FCC single crystals: [001] twins in compression, [111] in tension."""
    assert F.uniaxial_factor(I, axis, +1)[0] == pytest.approx(tension, abs=1e-3)
    assert F.uniaxial_factor(I, axis, -1)[0] == pytest.approx(compression, abs=1e-3)


def test_free_partials_narrow_in_001_tension_and_widen_in_compression():
    assert F.uniaxial_factor(I, (0, 0, 1), +1, criterion="byun")[0] == pytest.approx(-0.236, abs=1e-3)
    assert F.uniaxial_factor(I, (0, 0, 1), -1, criterion="byun")[0] == pytest.approx(+0.236, abs=1e-3)
    assert F.uniaxial_factor(I, (1, 1, 1), +1, criterion="byun")[0] > 0


def test_capacity_scales_with_stress_and_is_frame_invariant():
    R = F.random_rotations(50, seed=3)
    rng = np.random.default_rng(0)
    S = rng.normal(size=(20, 6)) * 300.0
    g1 = F.fault_energy_capacity(S, R)
    g2 = F.fault_energy_capacity(2.0 * S, R)
    assert np.allclose(g2, 2.0 * g1)
    # rotating stress and crystals together changes nothing
    Q = F.random_rotations(1, seed=9)[0]
    T = F._voigt_to_tensor(S)
    Sq = np.einsum("ij,njk,lk->nil", Q, T, Q)
    Rq = np.einsum("mij,kj->mik", R, Q)                       # crystal = R Q^T (Q sample)
    assert np.allclose(F.fault_energy_capacity(Sq, Rq), g1, atol=1e-9)
    # uniaxial: capacity = Schmid factor * stress * b_p
    s = np.array([[0, 0, 500.0, 0, 0, 0]])
    bp = F.A_CANTOR / np.sqrt(6)
    assert F.fault_energy_capacity(s, I)[0, 0] == pytest.approx(0.236 * 500e6 * bp * 1e3, rel=2e-3)


def test_onset_stress_and_olson_cohen_threshold():
    R = F.random_rotations(4000, seed=1)
    s = F.onset_stress_uniaxial(R, 30.0)
    bp = F.A_CANTOR / np.sqrt(6)
    assert s.min() == pytest.approx(30e-3 / (0.5 * bp) * 1e-6, rel=0.02)       # best grains: Schmid factor 0.5
    assert F.olson_cohen_threshold(30.0, 10.0, 1) == 30.0
    assert F.olson_cohen_threshold(30.0, 10.0) == pytest.approx(10.0)
