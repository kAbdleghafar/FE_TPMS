"""Layer scan paths for implicit parts and the geometry-aware melt-pool substrate."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.geometry.fields import Field, box
from tpmsfem.process import Laser, MeltPoolModel
from tpmsfem.process.scanpath import slice_mask, slice_scan_path


def test_box_slice_hatch_and_contour():
    part = box((0.0, 0.0, -1.0), (1.0, 0.5, 1.0))
    path, st = slice_scan_path(part, 0.0, (-0.2, 1.2, -0.2, 0.7), hatch=0.1, speed=1000.0, contour_offset=0.04)
    lit = [v for v in path if v.power != 0.0]
    assert st["contours"] == 1
    # contour offset 0.04 inward: perimeter of the 0.92 x 0.42 rectangle (rounded corners allowed)
    assert st["contour_length_mm"] == pytest.approx(2 * (0.92 + 0.42), rel=0.03)
    # hatch inside f < -(0.04 + 0.05): lines 0.1 apart across y in [0.09, 0.41], each 0.82 long
    hatch = [v for v in lit if abs(v.y1 - v.y0) < 1e-12 and abs(v.x1 - v.x0) > 0.5]
    assert len(hatch) == st["hatch_vectors"] in (3, 4)
    for v in hatch:
        assert min(v.x0, v.x1) >= 0.09 - 0.003 and max(v.x0, v.x1) <= 0.91 + 0.003
        assert 0.09 - 1e-9 <= v.y0 <= 0.41 + 1e-9
    assert hatch[0].x1 > hatch[0].x0 and hatch[1].x1 < hatch[1].x0          # alternating
    # every lit vector lies inside the part
    for v in lit:
        for s in (0.0, 0.5, 1.0):
            x, y = v.x0 + s * (v.x1 - v.x0), v.y0 + s * (v.y1 - v.y0)
            assert part(np.array(x), np.array(y), np.array(0.0)) < 0.0
    # consecutive lit vectors are joined by jumps with the laser off
    for a, b in zip(path[:-1], path[1:]):
        assert np.hypot(b.x0 - a.x1, b.y0 - a.y1) < 1e-9


def test_rotated_hatch_direction():
    part = box((0.0, 0.0, -1.0), (1.0, 1.0, 1.0))
    path, _ = slice_scan_path(part, 0.0, (-0.1, 1.1, -0.1, 1.1), hatch=0.1, contour=False, angle_deg=90.0)
    lit = [v for v in path if v.power != 0.0]
    assert all(abs(v.x1 - v.x0) < 1e-9 for v in lit)                       # along y


def test_slice_mask_and_part_substrate():
    th = ThermalProperties.from_si([0.0, 3000.0], [20.0, 20.0], [500.0, 500.0], 8000.0, 2.5e5, 1250.0, 1340.0)
    mat = tf.Material("m", 200_000.0, 0.3, thermal=th)
    wall = box((0.2, -1.0, -1.0), (0.4, 1.0, 1.0))                          # a 0.2 mm wall along y
    md = MeltPoolModel(mat, 0.01, length=0.6, width=0.4, depth=0.1, powder=0.03, symmetric_y=False)
    frac = md.set_part_geometry(wall)
    assert frac == pytest.approx(0.2 / 0.6, abs=0.02)
    pm = np.asarray(md.powder_mask)
    assert pm[:, :, md.nz_sub:].all()                                     # new powder layer
    # thermal check on the bare wall top (no powder layer): the substrate geometry alone
    md = MeltPoolModel(mat, 0.01, length=0.6, width=0.4, depth=0.1, symmetric_y=False)
    md.set_part_geometry(wall)
    pm = np.asarray(md.powder_mask)
    i_in = int(np.argmin(np.abs(md.x - 0.3)))
    i_out = int(np.argmin(np.abs(md.x - 0.1)))
    assert not pm[i_in, :, :md.nz_sub].any() and pm[i_out, :, :md.nz_sub].all()
    m = slice_mask(wall, 0.0, md.x, md.y)
    assert m[i_in].all() and not m[i_out].any()
    # heat stays in the wall: a track along the wall heats the powder beside it much less
    md.set_temperature(80.0)
    assert float(np.asarray(md.T).mean()) == pytest.approx(80.0)
    from tpmsfem.process import ScanVector
    res = md.run(Laser(150.0, 0.4, 0.04), [ScanVector(0.3, -0.15, 0.3, 0.15, 800.0)], t_extra=5e-5)
    k = md.nz_sub - 2
    j = int(np.argmin(np.abs(md.y)))
    assert res.T_peak[i_in, j, k] > res.T_liquidus
    assert res.T_peak[i_out, j, k] < 0.5 * res.T_peak[i_in, j, k]
    # a plate instead of the wall loses more heat sideways: the wall gets hotter
    plate = MeltPoolModel(mat, 0.01, length=0.6, width=0.4, depth=0.1, symmetric_y=False)
    rp = plate.run(Laser(150.0, 0.4, 0.04), [ScanVector(0.3, -0.15, 0.3, 0.15, 800.0)], t_extra=5e-5)
    assert res.T_peak[i_in, j, k - 3] > rp.T_peak[i_in, j, k - 3]
    assert abs(res.energy_balance) < 1e-6


def test_gyroid_sheet_thickness():
    """Thin sheets: volume fraction = wall thickness x surface area per volume (3.0915 / L for
    the gyroid), on a 3-D grid."""
    from tpmsfem.geometry.tpms import sheet_with_thickness

    n, L = 160, 4.0
    g = (np.arange(n) + 0.5) * L / n
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    for t in (0.1, 0.2):
        frac = (sheet_with_thickness("gyroid", L, t)(X, Y, Z) < 0).mean()
        assert frac == pytest.approx(t * 3.0915 / L, rel=0.03)
    graded = sheet_with_thickness("gyroid", L, lambda x, y, z: 0.2 + 0.4 * x / L)
    lo = (graded(X[: n // 8], Y[: n // 8], Z[: n // 8]) < 0).mean()
    hi = (graded(X[-n // 8:], Y[-n // 8:], Z[-n // 8:]) < 0).mean()
    assert hi > 2.0 * lo
