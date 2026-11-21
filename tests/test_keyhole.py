"""Keyhole physics: Fresnel absorption, ray tracing, bubbles (tpmsfem.process.keyhole)."""
import numpy as np
import pytest

from tpmsfem.backend import asnumpy

from tpmsfem.process.freesurface import GAS, INTERFACE, LIQUID, WALL
from tpmsfem.process.keyhole import BubbleModel, RayTracer, fresnel_absorptivity, index_from_absorptivity, smooth_fill

pytest.importorskip("numba")


def test_fresnel():
    n, k = index_from_absorptivity(0.26, 1.3)
    assert k == pytest.approx(1.3 * n)
    assert fresnel_absorptivity(1.0, n, k) == pytest.approx(0.26, abs=1e-12)
    n, k = index_from_absorptivity(0.26)                         # k = n (Bramson's limit)
    assert fresnel_absorptivity(1.0, n, k) == pytest.approx(0.26, abs=1e-12)
    assert fresnel_absorptivity(1.0, n, k) == pytest.approx(4 * n / ((n + 1) ** 2 + k ** 2))
    c = np.cos(np.radians(np.linspace(0, 90, 901)))
    A = fresnel_absorptivity(c, n, k)
    assert A[-1] == pytest.approx(0.0, abs=1e-9)                 # grazing incidence reflects all
    i = int(np.argmax(A))
    assert 70 < np.degrees(np.arccos(c[i])) < 89                # the metal's pseudo-Brewster maximum
    assert A.max() == pytest.approx(0.432, abs=2e-3)                # 0.43 at 83 deg for k = n


def _plate(shape, height):
    """Fill levels of a flat plate whose top is at z = height [cells] (walls around)."""
    nz = shape[2]
    z = np.arange(nz)
    col = np.clip(height - z, 0.0, 1.0)
    fill = np.broadcast_to(col, shape).copy()
    fill[[0, -1], :, :] = 0.0
    fill[:, [0, -1], :] = 0.0
    fill[:, :, [0, -1]] = 0.0
    return fill


def test_flat_plate_absorbs_A0():
    shape = (60, 60, 30)
    fill = _plate(shape, 12.4)
    rt = RayTracer(A0=0.3, spacing=0.5)
    dep, st = rt.trace(fill, fill > 0, 30.0, 30.0, 100.0, 8.0)
    assert st["absorbed"] == pytest.approx(0.3, abs=2e-3)
    assert st["escaped_top"] == pytest.approx(0.7, abs=2e-3)
    assert st["mean_bounces"] == pytest.approx(1.0)
    assert dep.sum() == pytest.approx(30.0, rel=1e-2)
    # all of it at the surface (half a cell into the metal: the full cell under the 0.4 one),
    # centred, with the Gaussian width (sigma = r0 / 2)
    assert dep[:, :, 11:13].sum() / dep.sum() > 0.99
    i = np.arange(shape[0]) + 0.5
    px = dep.sum(axis=(1, 2)) / dep.sum()
    mx = (px * i).sum()
    assert mx == pytest.approx(30.0, abs=0.05)
    assert np.sqrt((px * (i - mx) ** 2).sum()) == pytest.approx(4.0, rel=0.05)


def _inclined(shape, z0, slope, sub=8):
    """Fill levels (sub-sampled) of the half-space z < z0 + slope * (x - nx/2)."""
    nx, ny, nz = shape
    s = (np.arange(sub) + 0.5) / sub
    fx = np.zeros((nx, nz))
    for a in s:
        for b in s:
            X = np.arange(nx)[:, None] + a
            Z = np.arange(nz)[None, :] + b
            fx += Z < z0 + slope * (X - nx / 2)
    fx /= sub * sub
    fill = np.repeat(fx[:, None, :], ny, axis=1)
    fill[[0, -1]] = 0.0
    fill[:, [0, -1]] = 0.0
    fill[:, :, [0, -1]] = 0.0
    return fill


@pytest.mark.parametrize("deg", [20.0, 35.0])
def test_inclined_plane_absorbs_fresnel(deg):
    shape = (80, 40, 60)
    fill = _inclined(shape, 25.0, np.tan(np.radians(deg)))
    rt = RayTracer(A0=0.3, spacing=0.5)
    dep, st = rt.trace(smooth_fill(fill), fill > 0, 40.0, 20.0, 1.0, 5.0)
    expected = fresnel_absorptivity(np.cos(np.radians(deg)), rt.n, rt.k)
    assert st["first_cos"] == pytest.approx(np.cos(np.radians(deg)), abs=0.006)     # normal within ~1 deg
    assert st["absorbed"] == pytest.approx(expected, rel=0.02)          # one bounce, then out


def _paraboloid(shape, top, depth, R, sub=4):
    """Plate with a paraboloid cavity (depth, rim radius R [cells]) on the axis: a keyhole."""
    nx, ny, nz = shape
    fill = np.zeros(shape)
    s_ = (np.arange(sub) + 0.5) / sub
    for a in s_:
        for b in s_:
            for c in s_:
                X, Y, Z = np.meshgrid(np.arange(nx) + a, np.arange(ny) + b, np.arange(nz) + c, indexing="ij")
                r2 = (X - nx / 2) ** 2 + (Y - ny / 2) ** 2
                fill += Z < top - depth * np.clip(1.0 - r2 / R ** 2, 0.0, None)
    fill /= sub ** 3
    fill[[0, -1]] = 0.0
    fill[:, [0, -1]] = 0.0
    fill[:, :, [0, -1]] = 0.0
    return fill


def test_keyhole_cavity_traps_light():
    shape = (50, 50, 80)
    rt = RayTracer(A0=0.3)
    out = []
    for depth in (0, 20, 45):                     # aspect ratio depth / rim radius 0, 2, 4.5
        fill = _paraboloid(shape, 60.0, depth, 10.0)
        dep, st = rt.trace(smooth_fill(fill), fill > 0, 25.0, 25.0, 1.0, 4.0)
        tot = st["absorbed"] + st["escaped_top"] + st["escaped_side"] + st["stopped"] + st["not_deposited"]
        assert tot == pytest.approx(1.0, abs=1e-9)                     # every watt accounted for
        assert dep.sum() == pytest.approx(st["absorbed"])
        out.append((st["absorbed"], st["mean_bounces"]))
    assert out[0][0] == pytest.approx(0.3, abs=1e-3)
    assert out[1][0] == pytest.approx(0.52, abs=0.02)                  # multiple reflections
    assert out[2][0] == pytest.approx(0.62, abs=0.02)
    assert out[2][1] > 2.0


def _block_with_bubble(n=24, r=4.0, centre=(12, 12, 9), top=16):
    flag = np.full((n, n, n), GAS, np.int8)
    fill = np.zeros((n, n, n))
    flag[:, :, :top] = LIQUID
    fill[:, :, :top] = 1.0
    g = np.arange(n) + 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    inside = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2 < r * r
    flag[inside] = GAS
    fill[inside] = 0.0
    return flag, fill


def _close(flag):
    """Interface layer around every gas region, and walls."""
    gas = flag == GAS
    nb = np.zeros_like(gas)
    for ax in range(3):
        for d in (-1, 1):
            nb |= np.roll(gas, d, ax)
    flag = flag.copy()
    flag[(flag == LIQUID) & nb] = INTERFACE
    flag[[0, -1]] = WALL
    flag[:, [0, -1]] = WALL
    flag[:, :, [0, -1]] = WALL
    return flag


def test_bubble_pressure_follows_volume():
    flag, fill = _block_with_bubble()
    flag = _close(flag)
    bm = BubbleModel(p_ambient=1e5)
    dp = asnumpy(bm.update(flag, fill))
    assert len(bm.bubbles) == 1 and bm.formed == 1
    b = bm.bubbles[0]
    assert b["pressure"] == pytest.approx(1e5)
    assert np.allclose(dp, 0.0)
    assert b["centroid"] == pytest.approx([12.0, 12.0, 9.0], abs=0.1)
    # frozen in when all the metal around it is below the solidus
    bm_t = BubbleModel(p_ambient=1e5)
    bm_t.update(flag, fill, T=np.full(flag.shape, 1200.0), T_solidus=1600.0)
    assert bm_t.bubbles[0]["frozen"] and bm_t.bubbles[0]["T_wall_max"] == pytest.approx(1200.0)
    T = np.full(flag.shape, 1200.0)
    T[12, 12, 13] = 1700.0                                        # one hot cell on the bubble wall
    bm_t.update(flag, fill, T=T, T_solidus=1600.0)
    assert not bm_t.bubbles[0]["frozen"] and bm_t.bubbles[0]["T_wall_max"] == pytest.approx(1700.0)
    # a void that was never molten is not a pore; one frozen after its walls were molten is
    cold = BubbleModel(p_ambient=1e5)
    cold.update(flag, fill, T=np.full(flag.shape, 1200.0), T_solidus=1600.0)
    assert cold.bubbles[0]["frozen"] and not cold.bubbles[0]["pore"]
    bm_t.update(flag, fill, T=np.full(flag.shape, 1200.0), T_solidus=1600.0)   # the same bubble, now cold
    assert bm_t.bubbles[0]["frozen"] and bm_t.bubbles[0]["pore"]
    assert bm_t.bubbles[0]["T_wall_hist"] == pytest.approx(1700.0)
    # compress it: fill the gas cells of one half (as the melt would), pressure goes up as V0/V
    V0 = b["volume"]
    f2, fl2 = fill.copy(), flag.copy()
    half = (flag == GAS) & (np.arange(24)[:, None, None] < 12) & (np.arange(24)[None, None, :] < 15)
    fl2[half] = LIQUID
    f2[half] = 1.0
    fl2 = _close(np.where(fl2 == INTERFACE, LIQUID, fl2).astype(np.int8))
    dp = asnumpy(bm.update(fl2, f2))
    b = bm.bubbles[0]
    assert b["V0"] == pytest.approx(V0)
    assert b["pressure"] == pytest.approx(1e5 * V0 / b["volume"], rel=1e-9)
    assert b["pressure"] > 1.5e5
    itf_next = (fl2 == INTERFACE) & (dp != 0)
    assert itf_next.sum() > 10 and np.allclose(dp[itf_next], b["pressure"] - 1e5)
    # open it to the atmosphere: the gas is released, no bubble left
    fl3 = fl2.copy()
    fl3[12, 12, 9:17] = GAS
    fl3 = _close(np.where(fl3 == INTERFACE, LIQUID, fl3).astype(np.int8))
    dp = asnumpy(bm.update(fl3, f2))
    assert bm.bubbles == [] and bm.released == 1 and np.allclose(dp, 0.0)


def test_melt_pool_with_rays_and_bubbles():
    """The free-surface melt pool with the ray tracer and the bubble model: mass conserved,
    the energy balance closes on the booked residual, a flat plate absorbs A0 at first, and
    the diagnostics run."""
    import tpmsfem as tf
    from tpmsfem.fem.materials import ThermalProperties
    from tpmsfem.process import Evaporation, Laser, ScanVector
    from tpmsfem.process.freesurface import FreeSurfaceMeltPool
    from tpmsfem.process.keyhole import keyhole_metrics

    th = ThermalProperties.from_si([0.0, 20000.0], [30.0, 30.0], [800.0, 800.0], 4000.0, 2.9e5, 1600.0, 1660.0)
    mat = tf.Material("const", 110_000.0, 0.3, thermal=th)
    h = 5e-6
    metal = np.zeros((30, 20, 24), bool)
    metal[1:-1, 1:-1, 1:15] = True
    md = FreeSurfaceMeltPool(mat, h, metal, dt=5e-8, viscosity=3e-3, sigma=1.6, T0=25.0, T_max=8000.0,
                             dsigma_dT=-2.6e-4, evaporation=Evaporation(T_boil=3290.0, L_v=9.3e6, M=0.048),
                             energy_transport="lbm")
    md.rays = RayTracer(A0=0.26)
    md.ray_every = 5
    md.bubbles = BubbleModel()
    q = md._ray_source(15 * h, 10 * h, 300.0, 25e-6)
    assert md.ray_stats["absorbed"] == pytest.approx(0.26, abs=1e-3)       # the flat plate
    assert float(q.sum()) * h ** 3 == pytest.approx(0.26 * 300.0, rel=1e-3)
    seen = []
    r1 = md.run(Laser(300.0, 0.0, 25e-6), [ScanVector(h + 50e-6, 10 * h, h + 90e-6, 10 * h, 2.0)],
           callback=lambda m, t, x, y: seen.append((m.ray_stats["absorbed"], keyhole_metrics(m, x, y, 25e-6, 15 * h))),
           callback_every=100)
    assert len(seen) == 4
    assert abs(r1.mass_error) < 1e-12 and abs(r1.energy_balance + r1.energy_residual) < 1e-9
    assert r1.absorbed > 0.2 * 300.0 * 20e-6
    assert all(0.2 < a < 0.75 for a, _ in seen)
    assert all(np.isfinite(k["depth"]) and k["n_bubbles"] >= 0 for _, k in seen)
    res = md.run(Laser(300.0, 0.0, 25e-6), [], t_extra=5e-7)
    assert abs(res.mass_error) < 1e-12
    assert abs(res.energy_balance + res.energy_residual) < 1e-9


def test_open_sides_and_breakthrough():
    from tpmsfem.process.keyhole import open_gas_reaches
    n = 16
    flag = np.full((n, n, 24), LIQUID, np.int8)
    fill = np.ones(flag.shape)
    flag[:, :, 18:] = GAS                                      # atmosphere
    flag[:, :, 3:7] = GAS                                      # a gas layer under a slab (the powder's gas)
    fill[flag == GAS] = 0.0
    flag = _close(flag)
    closed = BubbleModel()
    closed.update(flag, fill)
    assert len(closed.bubbles) == 1                            # sealed: one big bubble
    opened = BubbleModel(open_sides=True)
    opened.update(flag, fill)
    assert opened.bubbles == []                                # open through the sides: ambient
    assert open_gas_reaches(flag, 10) == (False, 0)            # keyhole closed
    kh = flag.copy()
    kh[8, 8, 7:18] = GAS                                       # the keyhole breaks through the slab
    kh = _close(np.where(kh == INTERFACE, LIQUID, kh).astype(np.int8))
    reach, cells = open_gas_reaches(kh, 10)
    assert reach and cells > 50


def test_melt_diagnostics_counts_metal_only_and_separates_droplets():
    """T_max ignores stale temperatures of emptied cells; u_max is the pool's, a hot detached
    droplet is reported apart and a cold detached particle (powder) is not a droplet."""
    from types import SimpleNamespace

    from tpmsfem.backend import to_device
    from tpmsfem.process.keyhole import melt_diagnostics

    n = 20
    flag = np.full((n, n, n), GAS, np.int8)
    flag[:, :, :6] = LIQUID                                   # plate with its pool
    flag[:, :, 6] = INTERFACE
    flag[4:6, 4:6, 12:14] = INTERFACE                         # hot droplet, detached
    flag[14:16, 14:16, 12:14] = INTERFACE                     # cold particle, detached
    fill = np.where(flag == LIQUID, 1.0, np.where(flag == INTERFACE, 0.8, 0.0))
    T = np.full(flag.shape, 1000.0)
    T[10, 10, 3] = 2500.0                                     # hottest metal
    T[10, 10, 15] = 9000.0                                    # stale value in a gas cell
    T[4:6, 4:6, 12:14] = 2000.0
    u = np.zeros((3,) + flag.shape)
    u[0, 8, 8, 4] = 0.1                                       # pool: 0.1 lattice -> 10 m/s
    u[2, 4, 4, 12] = 0.3                                      # droplet: 30 m/s
    u[1, 10, 10, 15] = 0.5                                    # gas cell (never counted)
    md = SimpleNamespace(lbm=SimpleNamespace(flag=to_device(flag), fill=to_device(fill), u=to_device(u)),
                         T=to_device(T), dx=1e-6, dt=1e-8, Ts=1600.0)
    d = melt_diagnostics(md)
    assert d["T_max"] == pytest.approx(2500.0)
    assert d["u_max"] == pytest.approx(10.0)
    assert d["drops"] == 1
    assert d["u_max_drops"] == pytest.approx(30.0)
    assert d["drop_volume_um3"] == pytest.approx(8 * 0.8)
    assert d["u_max_all"] == pytest.approx(30.0)
    assert d["u_max_all_at_um"] == pytest.approx([4.5, 4.5, 12.5])
