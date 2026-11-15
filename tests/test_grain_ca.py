"""3-D grain cellular automaton (tpmsfem.process.grain_ca)."""
import numpy as np
import pytest

pytest.importorskip("numba")

from tpmsfem.process import grain_ca as CA  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), np.sin(a), 0.0], [-np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("angle", [0.0, 30.0, 45.0])
def test_envelope_follows_crystal_axes_not_grid(angle):
    """One grain in a uniformly undercooling melt grows as an octahedron along its own <100>
    axes, whatever its tilt to the grid (decentred octahedra), at the prescribed tip speed."""
    n, dx = 73, 1e-6
    grid = CA.CAGrid((0.0, 0.0, 0.0), (n, n, n), dx)
    g0 = np.full((n, n, n), -1, np.int32)
    c = n // 2
    g0[c, c, c] = 0
    R = _rot_z(angle)
    cells = np.flatnonzero(g0.ravel() < 0).astype(np.int64)
    ev = dict(cell=cells, tm=np.zeros(len(cells)), tl=np.zeros(len(cells)), cr=np.full(len(cells), 1e4),
              track=np.zeros(len(cells), np.int32))
    kin = CA.TipKinetics.power_law(1e-3, 2.0, dT_max=50.0)
    t_end = 1e-3                                    # half-diagonal ~33 cells
    r = CA.GrainCA(grid, g0, R[None], kin).run(ev, t_end=t_end).result()
    sol = np.argwhere(r.grain == 0) - c
    u = sol @ R.T                                   # crystal frame
    L = 1e-3 * 1e8 * t_end ** 3 / 3.0 / dx          # analytic half-diagonal [cells]: v = 1e-3 (1e4 t)^2
    assert abs(u[:, 0].max() - L) <= 0.08 * L + 1.0
    # extent along grid x relative to the extent along the crystal axis: max(|cos|, |sin|)
    ratio = sol[:, 0].max() / u[:, 0].max()
    assert abs(ratio - max(abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle))))) < 0.04
    # the grain fills the octahedron |u|_1 <= L
    assert 0.9 < len(sol) / (4.0 / 3.0 * np.abs(u).sum(1).max() ** 3) < 1.1


def test_competitive_growth_selects_aligned_grains():
    """Directional solidification from random grains: misaligned grains are overgrown
    (Walton-Chalmers), so the <100> alignment with the growth direction improves upwards."""
    rng = np.random.default_rng(1)
    nx, ny, nz, dx = 40, 40, 120, 1e-6
    grid = CA.CAGrid((0.0, 0.0, 0.0), (nx, ny, nz), dx)
    from scipy.spatial import cKDTree
    ng = 60
    _, lab = cKDTree(rng.uniform(0, nx, (ng, 2))).query(np.argwhere(np.ones((nx, ny))).astype(float))
    g0 = np.full((nx, ny, nz), -1, np.int32)
    g0[:, :, :3] = lab.reshape(nx, ny)[..., None]
    rot = S.random_rotations(ng, rng)
    G, V = 1e6, 0.05
    cells = np.flatnonzero(g0.ravel() < 0).astype(np.int64)
    z = (cells % nz) * dx
    ev = dict(cell=cells, tm=np.zeros(len(cells)), tl=z / V, cr=np.full(len(cells), G * V),
              track=np.zeros(len(cells), np.int32), dir=np.tile([0.0, 0.0, 1.0], (len(cells), 1)))
    kin = CA.TipKinetics.kgt(4.0, 0.48, -10.5, 3e-9, 3.65e-7)
    r = CA.GrainCA(grid, g0, rot, kin).run(ev).result()
    assert r.stats["liquid_left"] == 0

    def misalign(res, k):
        gg = res.grain[:, :, k].ravel()
        return np.degrees(np.arccos(np.clip(S.alignment(res.orientations[gg], np.array([0, 0, 1.0])), -1, 1))).mean()

    assert misalign(r, 110) < misalign(r, 3) - 10.0
    n_top = len(np.unique(r.grain[:, :, 110]))
    assert n_top < len(np.unique(r.grain[:, :, 3])) / 2
    # heat-flow bias (cellular limit): the orientation penalty along the growth direction is
    # removed, so selection is much weaker
    rc = CA.GrainCA(grid, g0, rot, kin, heat_flow_bias=1.0).run(ev).result()
    assert len(np.unique(rc.grain[:, :, 110])) > n_top
    assert misalign(rc, 110) > misalign(r, 110)


def test_kgt_kinetics():
    """KGT for the Ni-Nb quasi-binary: monotonic, and ~31 K tip undercooling at 0.1 m/s (the
    phase-field tips at those speeds sit 28-40 K below the liquidus)."""
    kin = CA.TipKinetics.kgt(4.0, 0.48, -10.5, 3e-9, 3.65e-7)
    assert np.all(np.diff(kin.v) >= 0)
    assert 27.0 < kin.undercooling(0.1) < 35.0
    assert kin(0.0) == 0.0


def test_track_events_steady_shift():
    """A straight track: every melted cell along the track has the same delays relative to the
    laser, and the two tracks of a raster melt in order; the persistent CA solidifies all."""
    l = np.linspace(-30e-6, 30e-6, 13)
    z = np.linspace(-20e-6, -1e-6, 8)
    Lg, Zg = np.meshgrid(l, z, indexing="ij")
    Tp = 2000.0 - 3e7 * np.hypot(Lg / 2.0, Zg)                 # half-ellipse pool, 17 um deep
    sec = CA.TrackSection(l, z, Tp, -1e-5 + 0 * Lg, 1e-4 + 1e0 * (Zg + 20e-6), 1e6 + 0 * Lg, 1500.0, 1.0)
    grid = CA.CAGrid((0.0, -40e-6, -24e-6), (40, 80, 24), 1e-6)
    tracks = [CA.Track((-1e-3, 0.0), (1e-3, 0.0), 0.0, 0.0), CA.Track((1e-3, 20e-6), (-1e-3, 20e-6), 0.0, 3e-3)]
    ev = CA.track_events(grid, sec, tracks)
    assert np.all(np.diff(ev["tm"]) >= 0)
    t0 = ev["track"] == 0
    i = ev["cell"][t0] // (80 * 24)
    x = i * 1e-6
    # delay of the liquidus after the laser: tl - (x - x_start)/v is a function of (y, z) only
    rel = ev["tl"][t0] - (x + 1e-3) / 1.0
    jk = ev["cell"][t0] % (80 * 24)
    for key in np.unique(jk)[:20]:
        assert np.ptp(rel[jk == key]) < 1e-12
    assert ev["tm"][ev["track"] == 1].min() > ev["tl"][t0].max()
    g0 = np.zeros((40, 80, 24), np.int32)                      # one substrate grain everywhere
    ca = CA.GrainCA(grid, g0, np.eye(3)[None], CA.TipKinetics.power_law(1e-3, 2.0))
    for tr in range(2):                                         # one call per track: persistent state
        sel = ev["track"] == tr
        ca.run({k: v[sel] for k, v in ev.items()})
    r = ca.result()
    assert r.stats["liquid_left"] == 0
    assert r.melted.sum() > 1000
    assert np.all(r.grain[r.melted] == 0)                      # epitaxial regrowth
