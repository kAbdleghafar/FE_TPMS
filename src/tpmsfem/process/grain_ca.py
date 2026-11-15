"""3-D cellular automaton of grain growth in laser melting (decentred octahedra).

Model (Gandin & Rappaz, Acta Mater. 45 (1997) 2187; the formulation of ExaCA, Rolchigo et
al., Comput. Mater. Sci. 214 (2022) 111692):

* **Cells.** Cubic cells of size ``dx``. Each solid cell carries a grain id; each grain has a
  cubic orientation (rotation matrix whose rows are the crystal <100> axes in the sample frame,
  as in :mod:`tpmsfem.process.solidification`).
* **Thermal history.** Per cell, a sequence of melting events (t_m, t_l, cr): the cell turns
  liquid at t_m, passes the liquidus at t_l and then undercools linearly,
  dT(t) = cr (t - t_l).
* **Growth envelopes.** Every solid cell next to the liquid carries an octahedral envelope
  aligned with its grain's <100> axes. The half-diagonal grows at the dendrite-tip velocity
  v(dT) (:class:`TipKinetics`, e.g. the KGT model of the alloy).
* **Capture.** A liquid neighbour (26-neighbourhood) below the liquidus whose centre lies
  inside an envelope joins that grain. If several envelopes reach it in the same step, the
  deepest one wins, so the result does not depend on the cell order. The captured cell gets
  a decentred envelope: same orientation, sharing the capturing corner (the dendrite tip) of
  its parent, with a size set by the projection of the cell centre on the parent's face
  (capped at sqrt(3) dx). Growth therefore follows the crystal axes and not the grid.
* **Undercooling of an envelope.** The cell's own dT if it melted in the same event as the
  liquid next to it; otherwise (unmelted substrate at the fusion boundary, or a cell that
  solidified in an earlier track) the largest dT of its liquid neighbours. An envelope grows
  only while one of its liquid neighbours is below the liquidus, and its half-diagonal is
  capped at 8 dx, where it already contains all neighbours.
* **Nucleation.** Heterogeneous nucleation in the undercooled liquid (Rappaz 1989; Gaussian
  nucleation undercooling): sites with density N [1/m^3]. A site still liquid when its
  undercooling passes its threshold becomes a new grain of random orientation.
* **Epitaxy.** Unmelted cells next to the melt get envelopes of zero size, so the melt pool
  grows from the substrate grains, and earlier tracks and layers seed later ones.
* **Heat-flow bias** (optional, ``heat_flow_bias`` = alpha in [0, 1]). The classical model
  (alpha = 0) treats dendrites, which grow along <100>. Cells, as in LPBF, grow in a direction
  between the crystal axis and the heat flow (e.g. Deschamps, Georgelin & Pocheau, PRE 78
  (2008) 011605). Here each envelope's frame is the grain's, rotated so that its <100>
  nearest to the growth direction (grad t_l) turns towards it by alpha times their angle:
  * alpha = 0: dendritic;
  * alpha = 1: every grain grows along the heat flow and orientation selection vanishes.
  This is a phenomenological interpolation, to be calibrated against measured grain maps.

Thermal input for scan strategies (:func:`track_events`). The steady-state temperature
history of one track (:class:`TrackSection`, from a MeltPoolModel run) is shifted along every
scan vector, with arbitrary direction, hatch and layer offsets. Two simplifications follow:
* thermal interaction between tracks (preheating) is not included;
* remelting, and epitaxial regrowth across tracks and layers, are included.

Units: SI (m, s, K). Speed: a serial numba kernel over the active cells (the interface),
~10^7 cell updates per second.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

EMPTY, LIQUID, SOLID, ACTIVE = 0, 1, 2, 3


# ---------------------------------------------------------------- tip kinetics
@dataclass(frozen=True)
class TipKinetics:
    """Dendrite-tip velocity v(dT) [m/s] as a table on a uniform undercooling grid [K];
    linear interpolation, constant beyond the last point."""
    dT: np.ndarray
    v: np.ndarray

    def __call__(self, dT):
        return np.interp(dT, self.dT, self.v, left=0.0)

    def undercooling(self, v):
        """Inverse: tip undercooling [K] at velocity v [m/s]."""
        return np.interp(v, self.v, self.dT)

    @classmethod
    def kgt(cls, c0: float, k: float, m: float, D: float, gibbs_thomson: float, dT_max: Optional[float] = None,
            n: int = 2001) -> "TipKinetics":
        """KGT model (Kurz, Giovanola & Trivedi, Acta Metall. 34 (1986) 823) at low Peclet
        number, solutal undercooling plus curvature, marginal stability sigma* = 1/(4 pi^2),
        thermal gradient neglected: for Peclet number Pe,

            Omega = Iv(Pe) = Pe e^Pe E1(Pe),  c_l* = c0 / (1 - (1 - k) Omega),
            r = Gamma / (sigma* |m| 2 Pe (1 - k) c_l*),  V = 2 D Pe / r,
            dT = |m| (c_l* - c0) + 2 Gamma / r.

        c0 [wt%], k, m [K/wt%], D [m^2/s], Gamma [K m]. Tabulated up to ``dT_max`` (default:
        99 % of the freezing range |m| c0 (1 - k)/k, where the simple model diverges)."""
        from scipy.special import exp1

        m = abs(m)
        sig = 1.0 / (4.0 * np.pi ** 2)
        Pe = np.logspace(-7, 1.0, 4000)
        Iv = Pe * np.exp(Pe) * exp1(Pe)
        cl = c0 / (1.0 - (1.0 - k) * Iv)
        r = gibbs_thomson / (sig * m * 2.0 * Pe * (1.0 - k) * cl)
        V = 2.0 * D * Pe / r
        dT = m * (cl - c0) + 2.0 * gibbs_thomson / r
        ok = np.isfinite(dT) & np.isfinite(V)
        dT, V = dT[ok], V[ok]
        order = np.argsort(dT)
        dT, V = dT[order], np.maximum.accumulate(V[order])
        top = 0.99 * m * c0 * (1.0 - k) / k if dT_max is None else dT_max
        grid = np.linspace(0.0, min(top, dT.max()), n)
        return cls(grid, np.interp(grid, dT, V, left=0.0))

    @classmethod
    def power_law(cls, a: float, n_exp: float, dT_max: float = 100.0, n: int = 2001) -> "TipKinetics":
        """v = a dT^n."""
        grid = np.linspace(0.0, dT_max, n)
        return cls(grid, a * grid ** n_exp)


# ---------------------------------------------------------------- thermal input
@dataclass
class TrackSection:
    """Steady-state thermal history of one track in its cross-section, relative to the
    laser: for lateral offset l and height z (relative to the top of the layer) [m], the
    peak temperature, the delays tau_m (melting) and tau_l (liquidus on cooling) after the
    laser passes [s], and the cooling rate at the liquidus [K/s]."""
    l: np.ndarray
    z: np.ndarray
    T_peak: np.ndarray           # (nl, nz)
    tau_m: np.ndarray
    tau_l: np.ndarray
    cooling_rate: np.ndarray
    T_melt: float                # level above which a cell counts as melted (default: solidus)
    speed: float                 # scan speed of the source run [m/s]

    @classmethod
    def from_meltpool(cls, res, x_section: float, speed: float, x_start: float, t_start: float = 0.0,
                      level: Optional[float] = None) -> "TrackSection":
        """From a single-track :class:`~tpmsfem.process.meltpool.MeltPoolResult` (mm units,
        track along +x starting at ``x_start`` [mm] at ``t_start``) and its steady section
        at ``x_section`` [mm]; ``speed`` in mm/s as in the model. A half-symmetric model is
        mirrored. Undefined delays near the fusion line (cells between solidus and liquidus)
        take the nearest defined value."""
        from scipy.ndimage import distance_transform_edt

        i = int(np.argmin(np.abs(res.x - x_section)))
        t_laser = t_start + (res.x[i] - x_start) / speed
        fields = [res.T_peak[i], res.t_melt[i] - t_laser, res.t_solid[i] - t_laser, res.cooling_rate[i]]
        y = res.y
        if res.symmetric_y:
            y = np.r_[-res.y[::-1], res.y]
            fields = [np.vstack([f[::-1], f]) for f in fields]
        Tp, tm, tl, cr = fields
        out = []
        for f in (tm, tl, cr):
            bad = ~np.isfinite(f)
            if bad.all():
                raise ValueError("no melted cells in the section")
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            out.append(f[tuple(idx)])
        tm, tl, cr = out
        tm = np.minimum(tm, tl)
        return cls(y * 1e-3, res.z * 1e-3, Tp, tm, tl, cr, res.T_solidus if level is None else level, speed * 1e-3)

    def sample(self, l, z, gradient: bool = False):
        """Interpolated (melted, tau_m, tau_l, cr) at points (l, z) [m]; with ``gradient``
        also d tau_l / dl and d tau_l / dz [s/m]."""
        from scipy.interpolate import RegularGridInterpolator as RGI

        pts = np.column_stack([np.clip(np.ravel(l), self.l[0], self.l[-1]), np.clip(np.ravel(z), self.z[0], self.z[-1])])
        inside = (np.ravel(l) >= self.l[0]) & (np.ravel(l) <= self.l[-1]) & (np.ravel(z) >= self.z[0]) \
            & (np.ravel(z) <= self.z[-1] + 0.5 * (self.z[-1] - self.z[-2]))
        Tp = RGI((self.l, self.z), self.T_peak)(pts)
        arrays = [self.tau_m, self.tau_l, self.cooling_rate]
        if gradient:
            arrays += list(np.gradient(self.tau_l, self.l, self.z))
        f = [RGI((self.l, self.z), a)(pts) for a in arrays]
        return ((Tp >= self.T_melt) & inside,) + tuple(f)


@dataclass(frozen=True)
class Track:
    """One scan vector: start and end (x, y) [m], the top of its layer z_top [m], the time the
    laser leaves the start t0 [s] and the speed [m/s] (default: that of the section)."""
    start: tuple
    end: tuple
    z_top: float
    t0: float
    speed: Optional[float] = None


@dataclass
class CAGrid:
    """Regular grid: cell centres at origin + (i, j, k) dx; shape (nx, ny, nz)."""
    origin: tuple
    shape: tuple
    dx: float

    def centres(self, axis):
        return self.origin[axis] + np.arange(self.shape[axis]) * self.dx

    @property
    def size(self):
        return int(np.prod(self.shape))


def track_events(grid: CAGrid, section: TrackSection, tracks: Sequence[Track]):
    """Melting events (cell, t_m, t_l, cr, track id, growth direction) of all tracks, sorted
    by t_m. A track's history at a point is the section's at the point's lateral offset and
    height, delayed by the time the laser takes to reach the point along the track (steady
    state). The growth direction is grad t_l / |grad t_l|."""
    x, y, z = (grid.centres(a) for a in range(3))
    cells, tms, tls, crs, ids, dirs = [], [], [], [], [], []
    half = 0.5 * (section.l[-1] - section.l[0])
    z_hi = section.z[-1] + 0.5 * (section.z[-1] - section.z[-2])          # top face of the section grid
    for n, tr in enumerate(tracks):
        v = tr.speed or section.speed
        p0, p1 = np.asarray(tr.start, float), np.asarray(tr.end, float)
        length = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / length
        nrm = np.array([-d[1], d[0]])
        # candidate box: within the section's half width of the track
        lo = np.minimum(p0, p1) - half
        hi = np.maximum(p0, p1) + half
        ix = np.flatnonzero((x >= lo[0]) & (x <= hi[0]))
        iy = np.flatnonzero((y >= lo[1]) & (y <= hi[1]))
        iz = np.flatnonzero((z - tr.z_top >= section.z[0]) & (z - tr.z_top <= z_hi))
        if not (len(ix) and len(iy) and len(iz)):
            continue
        I, J, K = np.meshgrid(ix, iy, iz, indexing="ij")
        px, py, pz = x[I].ravel(), y[J].ravel(), z[K].ravel()
        s = (px - p0[0]) * d[0] + (py - p0[1]) * d[1]
        lat = (px - p0[0]) * nrm[0] + (py - p0[1]) * nrm[1]
        keep = (s >= 0.0) & (s <= length)
        melted, tm, tl, cr, dl, dz = section.sample(lat[keep], pz[keep] - tr.z_top, gradient=True)
        sel = np.flatnonzero(keep)[melted]
        if not len(sel):
            continue
        t_pass = tr.t0 + s[sel] / v
        cid = (I.ravel()[sel] * grid.shape[1] + J.ravel()[sel]) * grid.shape[2] + K.ravel()[sel]
        cells.append(cid)
        tms.append(t_pass + tm[melted])
        tls.append(t_pass + tl[melted])
        crs.append(cr[melted])
        # growth direction: the liquidus isotherm passes later along grad t_l
        g = (np.outer(np.full(int(melted.sum()), 1.0 / v), [d[0], d[1], 0.0])
             + np.outer(dl[melted], [nrm[0], nrm[1], 0.0]) + np.outer(dz[melted], [0.0, 0.0, 1.0]))
        dirs.append(g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30))
        ids.append(np.full(len(sel), n, np.int32))
    if not cells:
        return dict(cell=np.zeros(0, np.int64), tm=np.zeros(0), tl=np.zeros(0), cr=np.zeros(0), track=np.zeros(0, np.int32),
                    dir=np.zeros((0, 3)))
    cell, tm, tl, cr, tid, gd = (np.concatenate(a) for a in (cells, tms, tls, crs, ids, dirs))
    o = np.argsort(tm, kind="stable")
    return dict(cell=cell[o].astype(np.int64), tm=tm[o], tl=tl[o], cr=cr[o].astype(np.float64), track=tid[o],
                dir=np.ascontiguousarray(gd[o]))


# ---------------------------------------------------------------- substrate
def voronoi_substrate(grid: CAGrid, grain_size: float, rng, mask=None, orientations=None):
    """Voronoi grains with mean spacing ``grain_size`` [m] (one seed per grain_size^3) in the
    cells of ``mask`` (default: all). Orientations: random, or drawn from ``orientations``
    (n, 3, 3). Returns (grain id per cell, -1 outside the mask; orientations)."""
    from scipy.spatial import cKDTree
    from .solidification import random_rotations

    ext = np.array(grid.shape) * grid.dx
    n = max(1, int(round(np.prod(ext) / grain_size ** 3)))
    seeds = np.asarray(grid.origin) + rng.uniform(0, 1, (n, 3)) * ext
    x, y, z = (grid.centres(a) for a in range(3))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    _, near = cKDTree(seeds).query(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    g = near.reshape(grid.shape).astype(np.int32)
    if mask is not None:
        g[~mask] = -1
    if orientations is None:
        rot = random_rotations(n, rng)
    else:
        rot = np.asarray(orientations)[rng.integers(0, len(orientations), n)]
    return g, rot


# ---------------------------------------------------------------- CA
@dataclass
class CAResult:
    grain: np.ndarray            # (nx, ny, nz) grain id, -1 = no material
    orientations: np.ndarray     # (n_grains, 3, 3)
    nucleated: np.ndarray        # bool per grain id
    melted: np.ndarray           # (nx, ny, nz) bool: melted at least once
    grid: CAGrid
    t_end: float
    steps: int
    seconds: float
    stats: dict = field(default_factory=dict)


class GrainCA:
    """Decentred-octahedron CA (see the module docstring).

    ``grain``: initial grain id per cell (-1: no material, e.g. powder or gas: such cells only
    take part once they melt); ``orientations``: (n, 3, 3) for those ids; ``nucleation``:
    (sites per m^3, mean undercooling [K], standard deviation [K]) or None. Time step: the
    envelopes grow by at most ``cfl`` dx and the liquid undercools by at most ``dT_step``
    (99th percentile of the cooling rates) per step.

    The state persists: :meth:`run` can be called repeatedly with the events of successive
    tracks or layers (later than the earlier ones), which keeps the event lists small."""

    def __init__(self, grid: CAGrid, grain: np.ndarray, orientations: np.ndarray, kinetics: TipKinetics,
                 nucleation: Optional[tuple] = None, seed: int = 0, cfl: float = 0.25, dT_step: float = 0.5,
                 heat_flow_bias: float = 0.0):
        self.grid, self.kin, self.cfl, self.dT_step = grid, kinetics, cfl, dT_step
        self.alpha = float(heat_flow_bias)
        self.rng = np.random.default_rng(seed)
        self.nucleation = nucleation
        N = grid.size
        self.grain = np.ascontiguousarray(grain, dtype=np.int32).ravel().copy()
        self.state = np.where(self.grain >= 0, SOLID, EMPTY).astype(np.int8)
        self.rot = np.ascontiguousarray(np.asarray(orientations, float).reshape(-1, 3, 3))
        self.n_sub = len(self.rot)
        self.n_grains = self.n_sub
        self.tl = np.full(N, np.inf)
        self.cr = np.zeros(N)
        self.gd = np.zeros((N, 3))
        self.mid = np.full(N, -1, np.int32)
        self.oc = np.zeros((N, 3))
        self.L = np.zeros(N)
        self.ever = np.zeros(N, np.bool_)
        self.work = (np.empty(N, np.int64), np.zeros(N, np.bool_), np.empty(N, np.int64), np.full(N, -1.0),
                     np.zeros(N, np.int64))
        self.n_act = 0
        self.n_liq = 0
        self.t = -np.inf
        self.track_offset = 0
        self.steps = 0
        self.seconds = 0.0
        self.stats = dict(nucleated=0, captured=0, sites=0)

    def run(self, events: dict, t_end: Optional[float] = None, dt_max: Optional[float] = None) -> "GrainCA":
        """Advance through ``events`` (from :func:`track_events`) until all liquid has
        solidified, or to ``t_end``."""
        import time

        from .solidification import random_rotations

        g = self.grid
        if self.nucleation is not None and self.nucleation[0] > 0 and len(events["cell"]):
            dens, mu, sd = self.nucleation
            melted_cells = np.unique(events["cell"])
            n_site = self.rng.binomial(len(melted_cells), min(dens * g.dx ** 3, 1.0))
            site_cell = self.rng.choice(melted_cells, n_site, replace=False).astype(np.int64)
            site_dT = np.maximum(self.rng.normal(mu, sd, n_site), 0.0)
        else:
            site_cell, site_dT = np.zeros(0, np.int64), np.zeros(0)
        n_tracks = int(events["track"].max()) + 1 if len(events["track"]) else 1
        need = self.n_grains + len(site_cell) * min(n_tracks, 8) + 16
        if need > len(self.rot):
            self.rot = np.ascontiguousarray(np.concatenate([self.rot, random_rotations(need - len(self.rot), self.rng)]))
        t_end = np.inf if t_end is None else t_end
        if dt_max is None:                      # undercooling steps of at most dT_step
            cr99 = float(np.percentile(events["cr"], 99)) if len(events["cr"]) else 1.0
            dt_max = self.dT_step / max(cr99, 1e-30)
        ids = (events["track"].astype(np.int64) + self.track_offset).astype(np.int32)
        self.track_offset += n_tracks
        t0 = time.perf_counter()
        gdir = events.get("dir")
        gdir = np.zeros((len(events["cell"]), 3)) if gdir is None else np.ascontiguousarray(gdir, dtype=np.float64)
        out = _kernel()(np.array(g.shape, np.int64), self.state, self.grain, self.tl, self.cr, self.mid, self.oc,
                        self.L, self.ever, self.rot, events["cell"].astype(np.int64), events["tm"].astype(np.float64),
                        events["tl"].astype(np.float64), events["cr"].astype(np.float64), ids, self.gd, gdir, self.alpha,
                        site_cell, site_dT.astype(np.float64), np.ascontiguousarray(self.kin.dT, dtype=np.float64),
                        np.ascontiguousarray(self.kin.v, dtype=np.float64), g.dx, self.cfl, dt_max, t_end,
                        self.n_grains, *self.work, self.n_act, self.n_liq, self.t)
        self.t, self.n_grains, steps, n_nuc, n_capt, self.n_liq, self.n_act = out
        self.steps += int(steps)
        self.seconds += time.perf_counter() - t0
        self.stats["nucleated"] += int(n_nuc)
        self.stats["captured"] += int(n_capt)
        self.stats["sites"] += int(len(site_cell))
        return self

    def result(self) -> CAResult:
        nucleated = np.zeros(int(self.n_grains), bool)
        nucleated[self.n_sub:] = True
        return CAResult(self.grain.reshape(self.grid.shape).copy(), self.rot[:self.n_grains].copy(), nucleated,
                        self.ever.reshape(self.grid.shape).copy(), self.grid, float(self.t), self.steps, self.seconds,
                        dict(self.stats, liquid_left=int(self.n_liq)))


_KERNEL = {}


def _kernel():
    if "k" in _KERNEL:
        return _KERNEL["k"]
    import numba

    S3, S2 = np.sqrt(3.0), np.sqrt(2.0)
    L_CAP = 8.0                     # [dx]: > 3 (farthest neighbour) + the largest decentring offset

    @numba.njit(cache=True)
    def vtip(dT, kin_dT, kin_v):
        if dT <= 0.0:
            return 0.0
        h = kin_dT[1] - kin_dT[0]
        f = dT / h
        i = int(f)
        if i >= len(kin_dT) - 1:
            return kin_v[-1]
        w = f - i
        return kin_v[i] * (1.0 - w) + kin_v[i + 1] * w

    @numba.njit(cache=True)
    def frame(R, n0, n1, n2, alpha, F):
        """Envelope frame: the grain's axes R, rotated so that the <100> nearest to the growth
        direction n turns towards n by alpha times their angle (Rodrigues)."""
        for a in range(3):
            for b in range(3):
                F[a, b] = R[a, b]
        nn = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
        if alpha <= 0.0 or nn < 1e-12:
            return
        n0, n1, n2 = n0 / nn, n1 / nn, n2 / nn
        best, amax, sg = 0, -1.0, 1.0
        for a in range(3):
            d = R[a, 0] * n0 + R[a, 1] * n1 + R[a, 2] * n2
            if abs(d) > amax:
                amax, best, sg = abs(d), a, (1.0 if d >= 0.0 else -1.0)
        if amax > 1.0 - 1e-9:
            return
        a0, a1, a2 = sg * R[best, 0], sg * R[best, 1], sg * R[best, 2]
        k0, k1, k2 = a1 * n2 - a2 * n1, a2 * n0 - a0 * n2, a0 * n1 - a1 * n0
        kn = np.sqrt(k0 * k0 + k1 * k1 + k2 * k2)
        k0, k1, k2 = k0 / kn, k1 / kn, k2 / kn
        phi = alpha * np.arccos(min(amax, 1.0))
        cp, sp = np.cos(phi), np.sin(phi)
        for a in range(3):
            v0, v1, v2 = R[a, 0], R[a, 1], R[a, 2]
            kd = k0 * v0 + k1 * v1 + k2 * v2
            F[a, 0] = v0 * cp + (k1 * v2 - k2 * v1) * sp + k0 * kd * (1.0 - cp)
            F[a, 1] = v1 * cp + (k2 * v0 - k0 * v2) * sp + k1 * kd * (1.0 - cp)
            F[a, 2] = v2 * cp + (k0 * v1 - k1 * v0) * sp + k2 * kd * (1.0 - cp)

    @numba.njit(cache=True)
    def decentre(oc, c, Lp, R, x0, x1, x2, ocn, n_out):
        """New envelope for the cell at (x0, x1, x2) captured by the envelope (centre oc[c],
        half-diagonal Lp, axes R): returns its half-diagonal, writes its centre to ocn[n_out]."""
        d0, d1, d2 = x0 - oc[c, 0], x1 - oc[c, 1], x2 - oc[c, 2]
        s = np.empty(3)
        for a in range(3):
            u = R[a, 0] * d0 + R[a, 1] * d1 + R[a, 2] * d2
            s[a] = 1.0 if u >= 0.0 else -1.0
        # corners of the capturing face and the face normal
        C = np.empty((3, 3))
        nv = np.zeros(3)
        for a in range(3):
            for b in range(3):
                C[a, b] = oc[c, b] + s[a] * Lp * R[a, b]
                nv[b] += s[a] * R[a, b] / S3
        q = (x0 - C[0, 0]) * nv[0] + (x1 - C[0, 1]) * nv[1] + (x2 - C[0, 2]) * nv[2]
        P0, P1, P2 = x0 - q * nv[0], x1 - q * nv[1], x2 - q * nv[2]
        Ic = np.empty(3)
        best, kmin = 1e300, 0
        for a in range(3):
            dd = np.sqrt((P0 - C[a, 0]) ** 2 + (P1 - C[a, 1]) ** 2 + (P2 - C[a, 2]) ** 2)
            if dd < best:
                best, kmin = dd, a
            Ic[a] = min(dd, S3)
        Lnew = S2 * max(0.5 * (Ic[0] + Ic[1]), 0.5 * (Ic[0] + Ic[2]), 0.5 * (Ic[1] + Ic[2]))
        for b in range(3):
            ocn[n_out, b] = C[kmin, b] - Lnew * s[kmin] * R[kmin, b]
        return Lnew

    @numba.njit(cache=True)
    def run(shape, state, grain, tl, cr, mid, oc, L, ever, rot, ev_cell, ev_tm, ev_tl, ev_cr, ev_id, gd, ev_dir, alpha,
            site_cell, site_dT, kin_dT, kin_v, dx, cfl, dt_max, t_end, n_grains,
            act, in_act, capt, best_p, best_src, n_act, n_liq, t):
        nx, ny, nz = shape[0], shape[1], shape[2]
        N = nx * ny * nz
        offs = np.empty((26, 3), np.int64)
        m = 0
        for a in range(-1, 2):
            for b in range(-1, 2):
                for cc in range(-1, 2):
                    if a != 0 or b != 0 or cc != 0:
                        offs[m, 0], offs[m, 1], offs[m, 2] = a, b, cc
                        m += 1
        n_ev = len(ev_cell)
        n_rot = rot.shape[0]
        F = np.empty((3, 3))
        fsrc = np.empty(nx * ny * nz, np.int64)    # growth-direction source cell of each active cell
        e = 0
        if n_ev > 0 and n_liq <= 0:
            t = max(t, ev_tm[0])
        dt = dt_max
        steps = 0
        n_nuc = 0
        n_capt_tot = 0
        while True:
            # melting events up to t
            while e < n_ev and ev_tm[e] <= t:
                c = ev_cell[e]
                if state[c] != 1:
                    n_liq += 1
                state[c] = 1
                grain[c] = -1
                tl[c] = ev_tl[e]
                cr[c] = ev_cr[e]
                mid[c] = ev_id[e]
                gd[c, 0], gd[c, 1], gd[c, 2] = ev_dir[e, 0], ev_dir[e, 1], ev_dir[e, 2]
                ever[c] = True
                i, j, k = c // (ny * nz), (c // nz) % ny, c % nz
                for q in range(26):
                    ii, jj, kk = i + offs[q, 0], j + offs[q, 1], k + offs[q, 2]
                    if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                        continue
                    nb = (ii * ny + jj) * nz + kk
                    if state[nb] == 2:
                        state[nb] = 3
                        L[nb] = 0.0
                        oc[nb, 0], oc[nb, 1], oc[nb, 2] = ii, jj, kk
                        if not in_act[nb]:
                            act[n_act] = nb
                            n_act += 1
                            in_act[nb] = True
                e += 1
            if n_liq <= 0:
                if e >= n_ev:
                    break
                t = ev_tm[e]
                continue
            if t > t_end:
                break
            # nucleation
            for s_ in range(len(site_cell)):
                c = site_cell[s_]
                if state[c] == 1 and t >= tl[c] and cr[c] * (t - tl[c]) >= site_dT[s_] and n_grains < n_rot:
                    grain[c] = n_grains
                    n_grains += 1
                    state[c] = 3
                    L[c] = 0.0
                    oc[c, 0], oc[c, 1], oc[c, 2] = c // (ny * nz), (c // nz) % ny, c % nz
                    n_liq -= 1
                    n_nuc += 1
                    if not in_act[c]:
                        act[n_act] = c
                        n_act += 1
                        in_act[c] = True
            if n_act == 0:
                # nothing grows: only nucleation can still start solidification
                site_left = False
                for s_ in range(len(site_cell)):
                    if state[site_cell[s_]] == 1:
                        site_left = True
                        break
                if not site_left:
                    if e >= n_ev:
                        break
                    t = ev_tm[e]
                    continue
            # growth of the envelopes and capture candidates
            vmax = 0.0
            n_capt = 0
            for a_ in range(n_act):
                c = act[a_]
                if state[c] != 3:
                    continue
                i, j, k = c // (ny * nz), (c // nz) % ny, c % nz
                has_liq = False
                own = False
                dT_nb = 0.0
                nb_best = c
                for q in range(26):
                    ii, jj, kk = i + offs[q, 0], j + offs[q, 1], k + offs[q, 2]
                    if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                        continue
                    nb = (ii * ny + jj) * nz + kk
                    if state[nb] == 1:
                        has_liq = True
                        dTn = cr[nb] * (t - tl[nb])
                        if dTn > dT_nb:
                            dT_nb = dTn
                            nb_best = nb
                        if mid[nb] == mid[c]:
                            own = True
                if not has_liq:
                    state[c] = 2
                    continue
                dT = cr[c] * (t - tl[c]) if own else dT_nb
                # grow only while some liquid neighbour is below the liquidus (else the envelope
                # would run ahead of the liquid it can capture), and not beyond L_CAP, where it
                # already contains all 26 neighbours
                src = c if own else nb_best
                frame(rot[grain[c]], gd[src, 0], gd[src, 1], gd[src, 2], alpha, F)
                fsrc[c] = src
                if dT > 0.0 and dT_nb > 0.0 and L[c] < L_CAP:
                    v = vtip(dT, kin_dT, kin_v)
                    if v > vmax:
                        vmax = v
                    L[c] = min(L[c] + v * dt / dx, L_CAP)
                if L[c] <= 0.0:
                    continue
                for q in range(26):
                    ii, jj, kk = i + offs[q, 0], j + offs[q, 1], k + offs[q, 2]
                    if ii < 0 or ii >= nx or jj < 0 or jj >= ny or kk < 0 or kk >= nz:
                        continue
                    nb = (ii * ny + jj) * nz + kk
                    if state[nb] != 1 or t < tl[nb]:
                        continue
                    d0, d1, d2 = ii - oc[c, 0], jj - oc[c, 1], kk - oc[c, 2]
                    dist = (abs(F[0, 0] * d0 + F[0, 1] * d1 + F[0, 2] * d2)
                            + abs(F[1, 0] * d0 + F[1, 1] * d1 + F[1, 2] * d2)
                            + abs(F[2, 0] * d0 + F[2, 1] * d1 + F[2, 2] * d2))
                    if dist <= L[c]:
                        p = L[c] - dist
                        if best_p[nb] < 0.0:
                            capt[n_capt] = nb
                            n_capt += 1
                        if p > best_p[nb]:
                            best_p[nb] = p
                            best_src[nb] = c
            # captures (deepest envelope wins)
            for q_ in range(n_capt):
                nb = capt[q_]
                src = best_src[nb]
                best_p[nb] = -1.0
                g = grain[src]
                grain[nb] = g
                state[nb] = 3
                n_liq -= 1
                ii, jj, kk = nb // (ny * nz), (nb // nz) % ny, nb % nz
                s2 = fsrc[src]
                frame(rot[g], gd[s2, 0], gd[s2, 1], gd[s2, 2], alpha, F)
                L[nb] = decentre(oc, src, L[src], F, float(ii), float(jj), float(kk), oc, nb)
                if not in_act[nb]:
                    act[n_act] = nb
                    n_act += 1
                    in_act[nb] = True
            n_capt_tot += n_capt
            w = 0
            for a_ in range(n_act):
                c = act[a_]
                if state[c] == 3:
                    act[w] = c
                    w += 1
                else:
                    in_act[c] = False
            n_act = w
            dt = dt_max if vmax <= 0.0 else min(dt_max, cfl * dx / vmax)
            t += dt
            steps += 1
        return t, n_grains, steps, n_nuc, n_capt_tot, n_liq, n_act

    _KERNEL["k"] = run
    return run
