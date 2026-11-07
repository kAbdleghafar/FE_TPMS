"""Solidification models: spacing laws, Hunt CET, epitaxial grain growth, ExaCA export."""
import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.process import Laser, MeltPoolModel, ScanVector
from tpmsfem.process import solidification as S


@pytest.fixture(scope="module")
def track():
    th = ThermalProperties.from_si([20.0, 870.0], [9.9, 22.8], [410.0, 620.0], 8440.0, 2.72e5, 1290.0, 1350.0)
    mat = tf.Material("in625", 2e5, 0.3, thermal=th)
    md = MeltPoolModel(mat, 0.005, length=0.7, width=0.15, depth=0.12, emissivity=0.4)
    return md.run(Laser(180.0, 0.35, 0.085), [ScanVector(0.1, 0.0, 0.55, 0.0, 800.0)], t_extra=2e-4)


def test_spacing_laws_scale_and_value():
    c = S.SolidificationConstants()
    G, R = 1e7, 0.5
    kf = S.primary_spacing(G, R, c, "kurz-fisher")
    ref = 4.3 * (c.freezing_range * c.D_liquid * c.gibbs_thomson / c.partition) ** 0.25 * G ** -0.5 * R ** -0.25 * 1e6
    assert kf == pytest.approx(ref)
    assert S.primary_spacing(4 * G, R, c) == pytest.approx(kf / 2)          # lambda ~ G^-1/2
    assert S.primary_spacing(G, 16 * R, c) == pytest.approx(kf / 2)         # lambda ~ R^-1/4
    assert S.primary_spacing(G, R, model="cooling-rate", a=80.0, b=0.33) == pytest.approx(80.0 * (G * R) ** -0.33)


def test_hunt_cet_columnar_at_high_gradient():
    G = np.array([1e8, 1e5, 1e2])
    cls = S.hunt_cet(G, np.full(3, 0.1))
    assert cls[0] == 0 and cls[-1] == 2 and np.all(np.diff(cls) >= 0)


def test_track_gr_and_epitaxial_grains(track, tmp_path):
    ok, G, R, cr = S.gr_fields(track)
    assert ok.sum() > 50
    assert np.allclose(cr[ok], G[ok] * R[ok], rtol=1e-9)                     # cooling rate = G R
    s = S.summary(track)
    assert 1e5 < s["cooling_rate_K_per_s"]["p50"] < 1e8
    gr = S.grow_grains(track, substrate_grain_size=0.02, p_equiaxed=0.0, seed=1)
    fz = gr.grain[gr.melted]
    assert np.all(fz >= 0)
    sub = np.unique(gr.grain[~gr.melted])
    assert np.isin(np.unique(fz), sub).all()                                 # purely epitaxial growth
    st = S.grain_stats(gr, track, x_section=0.4)
    assert st["grains_in_fusion_zone"] >= 1 and st["nucleated"] == 0
    n = S.export_exaca(track, tmp_path / "exaca.csv")
    head = open(tmp_path / "exaca.csv").readline().strip()
    assert head == "x,y,z,tm,tl,cr"
    data = np.loadtxt(tmp_path / "exaca.csv", delimiter=",", skiprows=1)
    assert len(data) == n and np.all(data[:, 4] > data[:, 3]) and np.all(data[:, 5] > 0)


def test_ipf_colors():
    I = np.eye(3)[None]
    assert np.allclose(S.ipf_color(I)[0], [1, 0, 0])                         # [001] red
