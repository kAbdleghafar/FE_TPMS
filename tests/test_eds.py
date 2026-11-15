"""Synthetic EDS maps (tpmsfem.post.eds)."""
import numpy as np
import pytest

from tpmsfem.post import eds


def test_line_selection_follows_overvoltage():
    assert eds.select_line("Nb", 15.0)[0] == "La"
    assert eds.select_line("Mo", 20.0)[0] == "La"
    assert eds.select_line("Ni", 15.0)[0] == "Ka"
    assert eds.select_line("Ni", 10.0)[0] == "La"     # E0/Ec = 1.2 < 1.5
    assert eds.select_line("Mo", 200.0)[0] == "Ka"
    with pytest.raises(ValueError):
        eds.select_line("Al", 1.0)


def test_generation_range_anderson_hasler():
    # Nb L3 edge 2.371 keV, 15 kV, IN625 8.44 g/cm^3: 0.064 (15^1.68 - 2.371^1.68) / 8.44
    assert eds.generation_range_um(15.0, 2.371, 8.44) == pytest.approx(0.064 * (15 ** 1.68 - 2.371 ** 1.68) / 8.44)
    assert 0.6 < eds.generation_range_um(15.0, 2.371, 8.44) < 0.75
    assert eds.generation_range_um(5.0, 2.371, 8.44) < 0.1


def _columns(n=64, nz=120, lam=16, amp=2.0, mean=4.0):
    """A field uniform in z with a sinusoid along x (wavelength lam cells)."""
    x = np.arange(n)
    f = mean + amp * np.cos(2 * np.pi * x / lam)
    return {"Nb": np.broadcast_to(f[None, None, :], (nz, n, n)).copy(),
            "Ni": np.full((nz, n, n), 60.0)}


def test_sem_blur_keeps_the_mean_and_damps_a_sinusoid_as_a_gaussian():
    dx, lam = 10e-9, 64                                      # 640 nm: partly resolved at 15 kV
    m = eds.sem_eds(_columns(n=128, lam=lam), dx, index=110, axis=0, beam_kV=15.0)
    a = m.maps["Nb"]
    assert a.mean() == pytest.approx(4.0, rel=1e-9)
    R = eds.generation_range_um(15.0, eds.select_line("Nb", 15.0)[2], 8.44) * 1e-6
    sig = 0.25 * R / dx
    expected = 2.0 * np.exp(-0.5 * (2 * np.pi * sig / lam) ** 2)
    amp = 0.5 * (a[0].max() - a[0].min())
    assert amp == pytest.approx(expected, rel=0.03)
    assert m.resolution_m["Nb"] == pytest.approx(2.355 * 0.25 * R)
    # 5 kV resolves what 15 kV smears
    m5 = eds.sem_eds(_columns(n=128, lam=lam), dx, index=110, axis=0, beam_kV=5.0)
    assert 0.5 * np.ptp(m5.maps["Nb"][0]) > 3 * amp and 0.5 * np.ptp(m5.maps["Nb"][0]) > 0.95 * 2.0


def test_stem_foil_average_and_counting_noise():
    dx = 10e-9
    f = _columns(lam=16)
    m = eds.stem_eds(f, dx, index=60, axis=0, foil_m=100e-9)
    assert 2e-9 < m.resolution_m["Nb"] < 10e-9               # probe + broadening, 100 nm foil at 200 kV
    assert np.allclose(m.maps["Nb"], f["Nb"][60], atol=0.02)  # uniform in z: the section itself
    noisy = eds.stem_eds(f, dx, index=60, axis=0, foil_m=100e-9, total_counts=2000.0, background=0.1, seed=1)
    expected = 2000.0 * m.maps["Nb"] / 100.0 + 0.1 * 2000.0 * m.maps["Nb"].mean() / 100.0
    assert noisy.maps["Nb"].mean() == pytest.approx(expected.mean(), rel=0.02)


def test_longitudinal_section_and_display():
    dx = 10e-9
    f = _columns(lam=16)
    m = eds.sem_eds(f, dx, index=32, axis=1, beam_kV=10.0, pixel_m=20e-9)
    assert m.axes == ("z", "x") and m.maps["Nb"].shape == (60, 32) and m.pixel_m == pytest.approx(20e-9)
    rgb = m.overlay(["Nb", "Ni"])
    assert rgb.shape == (60, 32, 3) and rgb.min() >= 0.0 and rgb.max() <= 1.0
    d, s = eds.line_profile(m, "Nb")
    assert len(d) == len(s) == 32
