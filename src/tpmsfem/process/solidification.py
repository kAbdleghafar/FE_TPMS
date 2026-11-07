"""Solidification microstructure from melt-pool thermal histories.

Three levels, all driven by a :class:`~tpmsfem.process.meltpool.MeltPoolResult`:

1. **G-R analysis** -- thermal gradient G and solidification velocity R at the last
   liquidus crossing of every melted cell; cooling rate G*R; morphology indicator G/R;
   columnar-to-equiaxed transition with Hunt's (1984) criterion.
2. **Cell / dendrite spacing** -- primary arm spacing from Kurz-Fisher (1981) or Hunt
   (1979) (``lambda_1 = C (dT0 D Gamma / k)^(1/4) G^(-1/2) R^(-1/4)``), or a power law of the
   cooling rate ``lambda_1 = a (G R)^(-b)``.
3. **Grain structure** -- a reduced cellular-automaton model of epitaxial, competitive
   columnar growth with optional equiaxed nucleation (below), and an exporter of the
   thermal history in the input format of ExaCA (ORNL's open-source GPU grain-growth CA).

Grain model. Unmelted substrate cells carry random grains (Voronoi, random cube
orientations). Melted cells re-solidify in the order of their solidification time; each
takes the grain of the already-solid neighbour whose nearest <100> axis is best aligned
with the local growth direction (grad T at solidification, pointing into the liquid), so
favourably oriented grains outgrow the others. Where Hunt's criterion predicts equiaxed
growth a new randomly oriented grain nucleates with probability ``p_equiaxed``. This is the
competitive-growth rule of decentred-octahedron CA models without the envelope kinetics; use
the ExaCA export for a full CA.

All material constants below are inputs. The defaults are generic Ni-base superalloy values
(the CET constants are those of CMSX-4, Gaumann et al. 2001) and are *not* calibrated for IN625.

Units: lengths mm, times s, temperatures degC/K, G in K/mm, R in mm/s (as in MeltPoolResult);
functions that use SI constants say so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SolidificationConstants:
    freezing_range: float = 60.0      # dT0 [K]  (liquidus - solidus of the property table)
    D_liquid: float = 3.0e-9          # solute diffusivity in the liquid [m^2/s]
    gibbs_thomson: float = 3.65e-7    # Gamma [K m]
    partition: float = 0.48           # k (Nb in Ni)
    # Hunt CET: nucleation density [1/m^3], nucleation undercooling [K], tip growth law R = a dT^n;
    # defaults: values for the Ni-base superalloy CMSX-4 (Gaumann et al., Acta Mater. 49 (2001) 1051)
    N0: float = 2.0e15
    dT_nuc: float = 2.5
    growth_a: float = 1.25e-6         # m/s / K^n
    growth_n: float = 3.4
    note: str = "generic Ni-base superalloy values (CET: CMSX-4, Gaumann 2001), not calibrated for IN625"


def gr_fields(res):
    """Melted cells and their G [K/m], R [m/s], cooling rate [K/s] (SI)."""
    ok = (res.n_melt > 0) & np.isfinite(res.t_solid) & (res.G > 0)
    return ok, res.G * 1e3, res.R * 1e-3, res.cooling_rate


def primary_spacing(G, R, c: SolidificationConstants = SolidificationConstants(), model: str = "kurz-fisher",
                    a: float = 80.0, b: float = 0.33):
    """Primary dendrite / cell arm spacing [um]. G [K/m], R [m/s] (SI).

    ``kurz-fisher``: 4.3 (dT0 D Gamma / k)^0.25 G^-0.5 R^-0.25;
    ``hunt``: 2.83 (k dT0 D Gamma)^0.25 G^-0.5 R^-0.25;
    ``cooling-rate``: a (G R)^-b  [um, K/s] -- an empirical fit; ``a``, ``b`` must come from
    the user's own measurements."""
    G, R = np.asarray(G, float), np.asarray(R, float)
    if model == "kurz-fisher":
        lam = 4.3 * (c.freezing_range * c.D_liquid * c.gibbs_thomson / c.partition) ** 0.25 * G ** -0.5 * R ** -0.25
        return lam * 1e6
    if model == "hunt":
        lam = 2.83 * (c.partition * c.freezing_range * c.D_liquid * c.gibbs_thomson) ** 0.25 * G ** -0.5 * R ** -0.25
        return lam * 1e6
    if model == "cooling-rate":
        return a * (G * R) ** -b
    raise ValueError(model)


def hunt_cet(G, R, c: SolidificationConstants = SolidificationConstants()):
    """Hunt (1984): equiaxed volume fraction proxy and classification.

    Fully columnar if G > 0.617 (100 N0)^(1/3) (1 - dTn^3/dTc^3) dTc, fully equiaxed if
    G < 0.617 N0^(1/3) (1 - dTn^3/dTc^3) dTc, with the tip undercooling dTc = (R/a)^(1/n).
    Returns an array: 0 columnar, 1 mixed, 2 equiaxed. G [K/m], R [m/s]."""
    G, R = np.asarray(G, float), np.asarray(R, float)
    dTc = (np.maximum(R, 0.0) / c.growth_a) ** (1.0 / c.growth_n)
    fac = np.maximum(1.0 - (c.dT_nuc / np.maximum(dTc, 1e-12)) ** 3, 0.0) * dTc
    G_col = 0.617 * (100.0 * c.N0) ** (1.0 / 3.0) * fac
    G_eq = 0.617 * c.N0 ** (1.0 / 3.0) * fac
    out = np.ones(G.shape, np.int8)
    out[G > G_col] = 0
    out[G < G_eq] = 2
    return out


def summary(res, c: SolidificationConstants = SolidificationConstants(), region=None):
    """Statistics of G, R, cooling rate, spacing and CET class over the re-solidified cells
    (optionally restricted to a boolean ``region``)."""
    ok, G, R, cr = gr_fields(res)
    if region is not None:
        ok &= region
    g, r, q = G[ok], R[ok], cr[ok]
    lam_kf = primary_spacing(g, r, c, "kurz-fisher")
    cls = hunt_cet(g, r, c)

    def pct(v):
        return {f"p{p}": float(np.percentile(v, p)) for p in (5, 50, 95)}
    return dict(cells=int(ok.sum()), G_K_per_m=pct(g), R_m_per_s=pct(r), cooling_rate_K_per_s=pct(q),
                G_over_R=pct(g / r), spacing_kurz_fisher_um=pct(lam_kf),
                cet_fraction=dict(columnar=float((cls == 0).mean()), mixed=float((cls == 1).mean()),
                                  equiaxed=float((cls == 2).mean())), constants=c.note)


# ---------------------------------------------------------------- orientations
def random_rotations(n, rng):
    """n uniformly random rotation matrices (via quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def alignment(rot, d):
    """cos of the angle between direction(s) d (sample frame) and the nearest <100> of each
    grain (rows of ``rot`` are the crystal axes in the sample frame)."""
    return np.max(np.abs(np.einsum("...ij,...j->...i", rot, d)), axis=-1)


def ipf_color(rot, axis=(0.0, 0.0, 1.0)):
    """Inverse-pole-figure RGB of the sample ``axis`` for cubic grains ([001] red, [101]
    green, [111] blue), for visualisation."""
    v = np.abs(np.einsum("...ij,j->...i", rot, np.asarray(axis, float)))
    v = np.sort(v, axis=-1)[..., ::-1]                     # h >= k >= l
    h, k, l_ = v[..., 0], v[..., 1], v[..., 2]
    rgb = np.stack([h - k, k - l_, l_ * np.sqrt(3.0)], -1)
    return rgb / np.maximum(rgb.max(axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- grain growth
@dataclass
class GrainResult:
    grain: np.ndarray            # (nx, ny, nz) grain id (-1 = never solid)
    rotations: np.ndarray        # (n_grains, 3, 3)
    new_grain: np.ndarray        # bool per grain: nucleated in the melt (equiaxed)
    melted: np.ndarray


def grow_grains(res, substrate_grain_size: float = 0.03, p_equiaxed: float = 0.0,
                c: SolidificationConstants = SolidificationConstants(), seed: int = 0) -> GrainResult:
    """Reduced CA of epitaxial competitive growth (see module docstring)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = res.T.shape
    h = res.h
    X, Y, Z = np.meshgrid(res.x, res.y, res.z, indexing="ij")
    # substrate grains: Voronoi seeds with mean spacing substrate_grain_size
    vol = (np.ptp(res.x) + h) * (np.ptp(res.y) + h) * (np.ptp(res.z) + h)
    n_seed = max(1, int(round(vol / substrate_grain_size ** 3)))
    seeds = np.column_stack([rng.uniform(res.x.min(), res.x.max(), n_seed), rng.uniform(res.y.min(), res.y.max(), n_seed),
                             rng.uniform(res.z.min(), res.z.max(), n_seed)])
    from scipy.spatial import cKDTree

    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    _, nearest = cKDTree(seeds).query(pts)
    grain = nearest.reshape(nx, ny, nz).astype(np.int64)
    rots = [random_rotations(n_seed, rng)]
    newg = [np.zeros(n_seed, bool)]
    n_grains = n_seed
    melted = (res.n_melt > 0) & np.isfinite(res.t_solid)
    grain[melted] = -1
    order = np.argsort(res.t_solid[melted], kind="stable")
    idx = np.array(np.nonzero(melted)).T[order]
    tsol = res.t_solid
    ok_G, G_si, R_si, _ = gr_fields(res)
    cls = np.full(res.T.shape, 0, np.int8)
    if p_equiaxed > 0:
        cls[ok_G] = hunt_cet(G_si[ok_G], R_si[ok_G], c)
    rot_all = rots[0]
    offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])
    offn = offs / np.linalg.norm(offs, axis=1, keepdims=True)
    for (i, j, k) in idx:
        t0 = tsol[i, j, k]
        d = res.grad_dir[i, j, k]
        if cls[i, j, k] == 2 and rng.random() < p_equiaxed:
            rot_all = np.concatenate([rot_all, random_rotations(1, rng)])
            newg.append(np.ones(1, bool))
            grain[i, j, k] = n_grains
            n_grains += 1
            continue
        nb = np.array([i, j, k]) + offs
        inside = np.all((nb >= 0) & (nb < [nx, ny, nz]), axis=1)
        # mirror at the symmetry plane y = 0
        if res.symmetric_y:
            nb[:, 1] = np.abs(nb[:, 1] + (nb[:, 1] < 0)) * (nb[:, 1] >= 0) + (nb[:, 1] < 0) * (-nb[:, 1] - 1)
            inside = np.all((nb >= 0) & (nb < [nx, ny, nz]), axis=1)
        nb, on = nb[inside], offn[inside]
        g_nb = grain[nb[:, 0], nb[:, 1], nb[:, 2]]
        solid = g_nb >= 0
        if melted.any():
            t_nb = tsol[nb[:, 0], nb[:, 1], nb[:, 2]]
            solid &= ~np.isfinite(t_nb) | (t_nb <= t0)
        if not solid.any():
            continue
        g_nb, on = g_nb[solid], on[solid]
        if np.all(np.isfinite(d)):
            # growth comes from the neighbour lying "upstream" (opposite to grad T); weight its
            # crystal alignment with the growth direction
            up = np.clip(-on @ d, 0.0, None)
            score = alignment(rot_all[g_nb], d) * (0.25 + up)
            grain[i, j, k] = g_nb[int(np.argmax(score))]
        else:
            grain[i, j, k] = g_nb[0]
    return GrainResult(grain, rot_all, np.concatenate(newg), melted)


def grain_stats(gr: GrainResult, res, x_section: Optional[float] = None, build_axis=(0.0, 0.0, 1.0)):
    """Grains in the fusion zone: count, mean transverse width [um] (equivalent diameter in
    the cross-section), and the fraction of fusion-zone volume whose nearest <100> lies within
    15 deg of the build axis and of the scan axis (texture)."""
    fz = gr.melted
    ids = gr.grain[fz]
    uniq, counts = np.unique(ids, return_counts=True)
    i = None if x_section is None else int(np.argmin(np.abs(res.x - x_section)))
    widths = []
    if i is not None:
        sec_ids = gr.grain[i][fz[i]]
        su, sc = np.unique(sec_ids, return_counts=True)
        mult = 2.0 if res.symmetric_y else 1.0
        widths = 2.0 * np.sqrt(sc * res.h * res.h * mult / np.pi) * 1e3
    cos15 = np.cos(np.radians(15.0))
    a_build = alignment(gr.rotations[ids], np.asarray(build_axis, float))
    a_scan = alignment(gr.rotations[ids], np.array([1.0, 0.0, 0.0]))
    return dict(grains_in_fusion_zone=int(len(uniq)), nucleated=int(gr.new_grain[uniq].sum()),
                section_grains=int(len(widths)), section_mean_width_um=float(np.mean(widths)) if len(widths) else None,
                frac_100_build_15deg=float((a_build >= cos15).mean()), frac_100_scan_15deg=float((a_scan >= cos15).mean()))


# ---------------------------------------------------------------- ExaCA export
def export_exaca(res, path, mirror_y: bool = True):
    """Thermal history of the melted cells in ExaCA's temperature-data format: CSV with header
    ``x,y,z,tm,tl,cr`` (SI: m, s, K/s) -- melting time, liquidus (solidification) time and
    cooling rate of the last melting event. A half-symmetric domain is mirrored to y < 0."""
    ok = (res.n_melt > 0) & np.isfinite(res.t_solid) & np.isfinite(res.t_melt)
    i, j, k = np.nonzero(ok)
    x, y, z = res.x[i], res.y[j], res.z[k]
    tm, tl, cr = res.t_melt[ok], res.t_solid[ok], res.cooling_rate[ok]
    if mirror_y and res.symmetric_y:
        x, y, z = np.r_[x, x], np.r_[y, -y], np.r_[z, z]
        tm, tl, cr = np.r_[tm, tm], np.r_[tl, tl], np.r_[cr, cr]
    data = np.column_stack([x * 1e-3, y * 1e-3, z * 1e-3, tm, tl, cr])
    np.savetxt(path, data, delimiter=",", header="x,y,z,tm,tl,cr", comments="", fmt="%.9e")
    return len(data)
