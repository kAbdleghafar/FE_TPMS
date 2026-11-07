"""AM-Bench 2018-02 case B: 3-D version of the EBSD grain test.

The 2-D test (nist_ambench2018_tracks_ebsd.py) regrew the measured transverse section from
the substrate *in that plane*. In a moving melt pool the growth direction (grad T at
solidification) also points forward along the scan, so a grain seen in the section may have
started on the fusion boundary behind it. Here every melt-pool pixel of the measured section
is traced backwards through the simulated *3-D* steady-state growth-direction field of case B
until it leaves the melt pool: that point is where its grain started.

* origin within +-delta of the section plane (delta = half the measured median substrate grain
  size): the pixel gets the measured substrate orientation at the origin's (y, z);
* origin further upstream, where NIST did not measure the substrate: it gets the orientation
  of a statistically equivalent substrate -- a 3-D Voronoi tessellation with the measured
  median grain size whose grains draw orientations from the measured substrate (several
  realisations).

The falsifiable prediction is that pixels with an in-plane origin match the measurement much
better than pixels with an upstream origin, and that the in-plane fraction bounds the
point-wise agreement any 2-D regrowth can reach.

    python benchmarks/literature/nist_ambench2018_tracks_ebsd3d.py      # ~5 min
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nist_ambench2018_tracks_ebsd as EB2  # noqa: E402

from tpmsfem.process import ebsd as E  # noqa: E402

OUT = "benchmarks/literature/results/ambench2018_tracks_ebsd3d.json"
X0 = 0.7                         # section position [mm] (steady state of the case-B run)
DS = 0.25e-3                     # trace step [mm]
REALISATIONS = 5


def full_fields(res):
    """Mirror the half-symmetric fields to the full width: T_peak, grad_dir (x, y, z)."""
    y = np.r_[-res.y[::-1], res.y]
    Tp = np.concatenate([res.T_peak[:, ::-1], res.T_peak], axis=1)
    gd = np.nan_to_num(res.grad_dir)
    gdm = gd[:, ::-1] * np.array([1.0, -1.0, 1.0])
    G = np.concatenate([gdm, gd], axis=1)
    return y, Tp, G


def trace_back(starts, res, max_len=0.8):
    """Integrate dp/ds = -grad_dir(p) from each start point (mm, model frame) until the peak
    temperature drops below the solidus (left the melt pool). Returns end points, path
    lengths and a 'reached' flag."""
    y, Tp, G = full_fields(res)
    axes = (res.x, y, res.z)
    fT = RGI(axes, Tp, bounds_error=False, fill_value=0.0)
    fG = [RGI(axes, G[..., k], bounds_error=False, fill_value=0.0) for k in range(3)]
    p = starts.copy()
    active = np.ones(len(p), bool)
    length = np.zeros(len(p))
    for _ in range(int(max_len / DS)):
        if not active.any():
            break
        q = p[active]
        d = np.column_stack([f(q) for f in fG])
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(n > 1e-6, d / np.maximum(n, 1e-12), np.array([[0.0, 0.0, 1.0]]))   # fallback: straight down
        q = q - DS * d
        p[active] = q
        length[active] += DS
        out = fT(q) < res.T_solidus
        idx = np.flatnonzero(active)
        active[idx[out]] = False
    return p, length, ~active


def main():
    EB2.fetch()
    m = E.read_ctf(EB2.CTF)
    g, valid, step = EB2.binned(m)
    ny, nx = valid.shape
    lab = E.segment_grains(g, valid, threshold=10.0, min_size=10)
    top = np.array([np.argmax(valid[:, j]) if valid[:, j].any() else 0 for j in range(nx)], float)
    top = median_filter(top, size=21, mode="nearest")
    xc, _ = EB2.fan_centre(lab, step, top)
    res, i = EB2.simulate_section()
    X = (np.arange(nx) + 0.5) * step
    Ypx = (np.arange(ny) + 0.5) * step
    depth = Ypx[:, None] - top[None, :] * step
    ym = (X[None, :] - xc) * 1e-3 * np.ones((ny, 1))
    zm = -depth * 1e-3
    y, Tp, G = full_fields(res)
    fT = RGI((res.x, y, res.z), Tp, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([np.full(ny * nx, X0), ym.ravel(), np.clip(zm.ravel(), res.z[0], res.z[-1])])
    Tsec = fT(pts).reshape(ny, nx)
    fz_geom = (Tsec >= res.T_solidus) & (depth >= 0)            # smooth model fusion line (for plots)
    fz = fz_geom & valid
    sub = valid & ~fz & (depth >= 0)
    print(f"section: fusion zone {fz.sum()} px, substrate {sub.sum()} px, centre x = {xc:.1f} um", flush=True)

    # substrate grain size (in-plane) -> delta and the equivalent 3-D substrate
    sub_lab = E.segment_grains(g, sub, threshold=10.0, min_size=10)
    sh = E.grain_shapes(sub_lab, (step, step))
    big = sh["area"] > 4 * step * step
    d_med = float(np.median(sh["d_eq"][big]))
    d_aw = float((sh["d_eq"][big] * sh["area"][big]).sum() / sh["area"][big].sum())   # area-weighted
    delta = 0.5 * d_med * 1e-3

    # 3-D back-trace of every melt-pool pixel of the section
    rows, cols = np.nonzero(fz)
    starts = np.column_stack([np.full(len(rows), X0), ym[rows, cols], zm[rows, cols]])
    ends, length, reached = trace_back(starts, res)
    dx = (X0 - ends[:, 0]) * 1e3                                      # um behind the section
    print(f"traces: {reached.mean() * 100:.1f} % reached the fusion boundary; origin behind the section: "
          f"median {np.median(dx):.1f} um, 90th pct {np.percentile(dx, 90):.1f} um; "
          f"{100 * (np.abs(dx) <= delta * 1e3).mean():.1f} % within +-{delta * 1e3:.1f} um of the plane", flush=True)

    # orientation lookup: measured substrate at the origin's (y, z) for in-plane origins
    sub_px = np.argwhere(sub)
    sub_y = (X[sub_px[:, 1]] - xc) * 1e-3
    sub_z = -depth[sub_px[:, 0], sub_px[:, 1]] * 1e-3
    tree = cKDTree(np.column_stack([sub_y, sub_z]))
    _, j = tree.query(np.column_stack([ends[:, 1], ends[:, 2]]))
    g_inplane = g[sub_px[j, 0], sub_px[j, 1]]
    inplane = np.abs(dx) <= delta * 1e3
    gm = g[rows, cols]

    def agree(gp, sel=None):
        mis = E.misorientation(gm if sel is None else gm[sel], gp if sel is None else gp[sel])
        return float((mis < 10).mean())

    rng = np.random.default_rng(0)
    sub_orients = g[sub]

    def realise(d_grain, n_real, keep=False):
        """Equivalent 3-D substrate: Voronoi grains of size d_grain [um] with orientations drawn
        from the measured substrate; in-plane origins keep the measured orientation."""
        real, g3 = [], None
        for r in range(n_real):
            lo = ends.min(axis=0) - 5 * delta
            hi = ends.max(axis=0) + 5 * delta
            nseed = int(np.prod(hi - lo) / (d_grain * 1e-3) ** 3) + 1
            seeds = lo + rng.random((nseed, 3)) * (hi - lo)
            so = sub_orients[rng.integers(0, len(sub_orients), nseed)]
            _, k = cKDTree(seeds).query(ends)
            gp = np.where(inplane[:, None, None], g_inplane, so[k])
            real.append(dict(all=agree(gp), inplane=agree(gp, inplane), upstream=agree(gp, ~inplane)))
            if r == 0:
                g3 = g.copy()
                g3[rows, cols] = gp
        return {k: (float(np.mean([q[k] for q in real])), float(np.std([q[k] for q in real]))) for k in real[0]}, g3

    # parameter-free check: give every pixel the measured in-plane substrate orientation at its
    # origin's (y, z), and bin the agreement by how far behind the section the origin lies
    mis_ip = E.misorientation(gm, g_inplane)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 1e9])
    bins = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        sel = (np.abs(dx) >= a_) & (np.abs(dx) < b_)
        if sel.mean() >= 0.01:
            bins.append(dict(dx_lo=float(a_), dx_hi=float(min(b_, 99.0)), pixels=int(sel.sum()),
                             frac_pixels=float(sel.mean()), agree=float((mis_ip[sel] < 10).mean())))
    print("agreement vs origin distance:", [(b["dx_lo"], round(100 * b["frac_pixels"], 1), round(100 * b["agree"], 1))
                                           for b in bins], flush=True)
    rnd, g3d = realise(d_med, REALISATIONS)
    rnd_aw, g3d_aw = realise(d_aw, REALISATIONS)
    # the 2-D trace restricted to the same pixel groups, for comparison
    d2 = EB2  # noqa: F841
    out = dict(source=EB2.URL, section_x_mm=X0, step_um=step, centre_um=xc, fusion_zone_px=int(fz.sum()),
               substrate_median_d_eq_um=d_med, substrate_area_weighted_d_eq_um=d_aw, delta_um=delta * 1e3,
               traces=dict(reached=float(reached.mean()), dx_median_um=float(np.median(dx)),
                           dx_p90_um=float(np.percentile(dx, 90)), frac_inplane=float(inplane.mean()),
                           path_median_um=float(np.median(length) * 1e3)),
               agreement_vs_origin_distance=bins,
               agreement_3d=dict(all=rnd["all"], inplane=rnd["inplane"], upstream=rnd["upstream"]),
               agreement_3d_area_weighted_grains=dict(all=rnd_aw["all"], inplane=rnd_aw["inplane"],
                                                      upstream=rnd_aw["upstream"]))
    print(json.dumps(out["agreement_3d"]), json.dumps(out["agreement_3d_area_weighted_grains"]), flush=True)
    stats = {}
    for name, gg in (("measured", g), (f"tpmsfem 3-D trace, substrate grains {d_med:.1f} um (median)", g3d),
                     (f"tpmsfem 3-D trace, substrate grains {d_aw:.1f} um (area-weighted)", g3d_aw)):
        lab2 = E.segment_grains(gg, fz, threshold=10.0, min_size=10)
        s2 = E.grain_shapes(lab2, (step, step))
        a = s2["area"]
        stats[name] = dict(grains=int(lab2.max() + 1), area_weighted_d_eq_um=float((s2["d_eq"] * a).sum() / a.sum()))
    out["grain_stats"] = stats
    rnd_or = E.misorientation(gm, S_random(len(gm), rng))
    out["random_frac_lt10"] = float((rnd_or < 10).mean())
    print(json.dumps(stats), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    figure(g, valid, fz_geom, g3d, inplane, rows, cols, dx, step, xc, ends, res, out)


def S_random(n, rng):
    from tpmsfem.process import solidification as S
    return S.random_rotations(n, rng)


def figure(g, valid, fz, g3d, inplane, rows, cols, dx, step, xc, ends, res, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tpmsfem.process import solidification as S

    SURF, INK2, BLUE, ORANGE = "#fcfcfb", "#52514e", "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "figure.facecolor": SURF, "axes.facecolor": SURF, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    ny, nx = valid.shape
    ext = [0, nx * step, ny * step, 0]
    fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05])
    build = (0.0, -1.0, 0.0)
    for k, (gg, title) in enumerate(((g, "Measured (NIST EBSD), IPF of the build direction"),
                                     (g3d, f"tpmsfem 3-D trace: {100 * out['agreement_3d']['all'][0]:.0f} % of melt-pool "
                                           f"pixels within 10° (one substrate realisation)"))):
        ax = fig.add_subplot(gs[k, :])
        rgb = S.ipf_color(gg, build)
        rgb[~valid] = 1.0
        ax.imshow(rgb, extent=ext, interpolation="nearest")
        ax.contour((np.arange(nx) + 0.5) * step, (np.arange(ny) + 0.5) * step, fz.astype(float), levels=[0.5],
                   colors=["#0b0b0b"], linewidths=0.7)
        ax.set_xlim(xc - 85, xc + 85)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_ylabel("depth [µm]")
    # where the grains of the section come from
    ax = fig.add_subplot(gs[2, 0])
    img = np.full((ny, nx), np.nan)
    img[rows, cols] = np.clip(dx, 0, None)
    im = ax.imshow(img, extent=ext, cmap="Blues", vmin=0, vmax=float(np.nanpercentile(dx, 95)))
    ax.contour((np.arange(nx) + 0.5) * step, (np.arange(ny) + 0.5) * step, fz.astype(float), levels=[0.5],
               colors=["#0b0b0b"], linewidths=0.7)
    ax.set_xlim(xc - 85, xc + 85)
    ax.set_title("Origin of each pixel's grain:\ndistance behind the section [µm]", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = fig.add_subplot(gs[2, 1])
    bins = out["agreement_vs_origin_distance"]
    xs = [0.5 * (b["dx_lo"] + min(b["dx_hi"], 12.0)) for b in bins]
    ws = [min(b["dx_hi"], 12.0) - b["dx_lo"] for b in bins]
    ax.bar(xs, [100 * b["frac_pixels"] for b in bins], width=ws, color="#e4e3df", edgecolor=SURF, label="share of pixels")
    ax.plot(xs, [100 * b["agree"] for b in bins], "-o", color=BLUE, lw=2, ms=4,
            label="within 10° of measured,\nusing the in-plane substrate")
    ax.axhline(100 * out["random_frac_lt10"], color=ORANGE, lw=1, ls="--", label="random orientations")
    ax.axvline(out["substrate_median_d_eq_um"], color=INK2, lw=0.8, ls=":")
    ax.text(out["substrate_median_d_eq_um"] + 0.2, 30, "median substrate\ngrain size", fontsize=7, color=INK2)
    ax.set_xlabel("grain origin behind the section [µm]")
    ax.set_ylabel("%")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 75)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("Agreement falls with the origin's distance\nfrom the measured plane", fontsize=8.5, loc="left")
    fig.suptitle("AM-Bench 2018-02 case B: 3-D origin of the melt-pool grains", fontsize=10, x=0.02, ha="left")
    fig.savefig("docs/img/ebsd_caseB_3d.png", dpi=200)


if __name__ == "__main__":
    main()
