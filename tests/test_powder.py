"""Powder-bed generation (tpmsfem.process.powder)."""
import numpy as np
import pytest

from tpmsfem.process.powder import particle_sizes, rain_powder, voxelise


def test_rained_bed_is_valid_and_dense():
    ext, floor, layer = (0.0, 0.25e-3, 0.0, 0.25e-3), 50e-6, 90e-6
    b = rain_powder(ext, floor, layer, seed=2)
    c, r = b.centres, b.radii
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert (r[:, None] + r[None] - d).max() < 1e-12                      # no overlaps
    assert np.all(c[:, 2] - r >= floor - 1e-12)                          # on or above the floor
    assert np.all(c[:, 2] + r <= floor + layer + 1e-12)                  # under the recoater
    assert np.all((c[:, 0] - r >= ext[0] - 1e-12) & (c[:, 0] + r <= ext[1] + 1e-12))
    assert 0.40 < b.packing_density < 0.62                               # spread powder
    assert np.all((2 * r >= 15e-6 - 1e-12) & (2 * r <= 45e-6 + 1e-12))


def test_sizes_and_voxelisation():
    d = particle_sizes(5000, d50=30e-6, sigma_ln=0.25, rng=1)
    assert np.median(d) == pytest.approx(30e-6, rel=0.05)
    b = rain_powder((0.0, 0.12e-3, 0.0, 0.12e-3), 10e-6, 40e-6, seed=1)
    dx = 4e-6
    f = voxelise(b, dx, (32, 32, 16))
    assert f.sum() * dx ** 3 == pytest.approx((4.0 / 3.0 * np.pi * b.radii ** 3).sum(), rel=0.02)
    assert f.max() <= 1.0 and f.min() >= 0.0
