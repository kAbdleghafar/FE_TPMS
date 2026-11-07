"""Surface-topography model: staircase law, rounding, particle statistics."""
import numpy as np
import pytest

from tpmsfem.process.roughness import PowderPSD, SurfaceModel, calibrate_particle_density, inclination, profile_parameters


@pytest.mark.parametrize("theta", [30.0, 60.0, 120.0, 150.0])
def test_sharp_staircase_ra(theta):
    m = SurfaceModel(layer=0.02, bead_radius=0.0, rho_side=0.0)
    s = np.arange(0, 2.0, 0.0002)
    r = profile_parameters(m.staircase(theta, s))
    assert r["Ra"] == pytest.approx(1e3 * 0.02 * abs(np.cos(np.radians(theta))) / 4, rel=0.02)


def test_vertical_wall_without_powder_is_smooth_and_rounding_reduces_ra():
    m = SurfaceModel(layer=0.02, bead_radius=0.05, rho_side=0.0)
    assert m.roughness(90.0, size=(0.4, 0.2))["Sa"] < 1e-6
    sharp = SurfaceModel(layer=0.04, bead_radius=0.0).roughness(45.0, size=(0.6, 0.1))["Ra"]
    round_ = SurfaceModel(layer=0.04, bead_radius=0.05).roughness(45.0, size=(0.6, 0.1))["Ra"]
    assert round_ < sharp


def test_particles_make_downskin_rougher_and_calibration_hits_target():
    m = SurfaceModel(layer=0.03, bead_radius=0.05, psd=PowderPSD(0.015, 0.03, 0.045), rho_side=60.0)
    up, side, down = (m.roughness(t, size=(1.0, 1.0), seed=3)["Sa"] for t in (45.0, 90.0, 135.0))
    assert down > side > 0 and down > up
    cal = calibrate_particle_density(m, 90.0, 6.0, size=(1.0, 1.0), seed=3)
    assert cal.roughness(90.0, size=(1.0, 1.0), seed=3)["Ra"] == pytest.approx(6.0, abs=0.05)


def test_inclination():
    assert np.allclose(inclination([[0, 0, 1], [1, 0, 0], [0, 0, -1]]), [0, 90, 180])
