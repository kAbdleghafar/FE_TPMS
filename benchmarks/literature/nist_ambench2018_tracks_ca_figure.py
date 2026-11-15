"""Figure for nist_ambench2018_tracks_ca.py."""
import json

import numpy as np

import nist_ambench2018_tracks_ca as M
from tpmsfem.process import solidification as S

INK, MUTED, GRID, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"


def _measured_ipf():
    g, valid, step, top, xc = M.ebsd_frame()
    sec = M.section()
    ny, nx = valid.shape
    X = (np.arange(nx) + 0.5) * step
    Yp = (np.arange(ny) + 0.5) * step
    depth = Yp[:, None] - top[None, :] * step
    melted, *_ = sec.sample((X[None, :] - xc) * 1e-6 * np.ones((ny, 1)), -depth * 1e-6)
    fz = melted.reshape(ny, nx) & (depth >= 0) & valid
    ipf = np.ones((ny, nx, 3))
    ipf[valid] = S.ipf_color(g[valid], axis=(0.0, -1.0, 0.0))
    return ipf, fz, valid, step


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_erosion

    out = json.load(open(M.OUT))
    f = dict(np.load(M.FIELDS))
    ipf_m, fz, valid, step = _measured_ipf()
    edge = fz & ~binary_erosion(fz)
    rows = np.flatnonzero(fz.any(axis=1))
    cols = np.flatnonzero(fz.any(axis=0))
    r0, r1 = 0, min(rows.max() + 25, fz.shape[0])
    c0, c1 = max(cols.min() - 25, 0), min(cols.max() + 25, fz.shape[1])
    ext = [0, (c1 - c0) * step, (r1 - r0) * step, 0]

    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05], hspace=0.42, wspace=0.18, left=0.04, right=0.98,
                          top=0.93, bottom=0.06)
    fig.set_size_inches(14, 10)

    def section(ax, img, title, mask_fz=True):
        im = img.astype(float).copy()
        if mask_fz:
            shown = np.where(fz[..., None], im, 0.55 * ipf_m + 0.45)       # substrate faded
        else:
            shown = im
        shown = shown.copy()
        shown[edge] = 0.0
        shown[~valid] = 1.0
        ax.imshow(shown[r0:r1, c0:c1], extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("um", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)

    section(fig.add_subplot(gs[0, 0]), ipf_m, "(a) measured (NIST EBSD, case-B track)\nIPF along the build direction")
    r = out["extruded"]
    section(fig.add_subplot(gs[0, 1]), f["extruded_section_ipf"],
            f"(b) CA, measured substrate extended along the scan\n{100 * r['frac_lt10']:.1f} % of pixels within 10 deg, "
            f"{r['stats']['CA']['grains']} grains (measured {r['stats']['measured']['grains']})")
    h = out["hybrid 0"]
    section(fig.add_subplot(gs[0, 2]), f["hybrid_section_ipf"],
            f"(c) CA, measured substrate in the plane only, 3-D Voronoi elsewhere\n{100 * h['frac_lt10']:.1f} % within 10 deg, "
            f"{h['stats']['CA']['grains']} grains")
    nz = out["extruded, nucleation 1e15/m3 at 30 +- 2 K"]
    section(fig.add_subplot(gs[1, 0]), f["nuc_section_ipf"],
            f"(d) CA as (b) with nucleation (1e15 /m3 at 30 K)\n{100 * nz['nucleated_fraction']:.0f} % of the pool from new grains, "
            f"{100 * nz['frac_lt10']:.1f} % within 10 deg")

    # longitudinal section and plan view of the extruded run (model frame, IPF along z)
    dx = M.DX * 1e6
    lg = f["hybrid_long_ipf"].astype(float)                  # (nx, nz)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.transpose(lg, (1, 0, 2))[::-1], extent=[0, lg.shape[0] * dx, -lg.shape[1] * dx, 0], interpolation="nearest",
              aspect="auto")
    ax.axvline(40.0, color=INK, lw=0.8, ls="--")
    ax.text(40.6, -43, "EBSD plane", fontsize=7, color=INK)
    ax.set_title("(e) CA as (c), longitudinal section through the track centre\n(laser moving to the right; IPF along z)",
                 fontsize=9, loc="left")
    ax.set_xlabel("x [um]", fontsize=8)
    ax.set_ylabel("z [um]", fontsize=8)
    ax.tick_params(labelsize=7, colors=MUTED)
    pl = f["hybrid_plan_ipf"].astype(float)                  # (nx, ny)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.transpose(pl, (1, 0, 2)), extent=[0, pl.shape[0] * dx, -75, 75], origin="lower", interpolation="nearest",
              aspect="auto")
    ax.set_title("(f) CA as (c), plan view 5 um below the surface\n(IPF along z)", fontsize=9, loc="left")
    ax.set_xlabel("x [um]", fontsize=8)
    ax.set_ylabel("y [um]", fontsize=8)
    ax.tick_params(labelsize=7, colors=MUTED)

    # agreement bar chart
    ax = fig.add_axes([0.28, 0.06, 0.38, 0.26])
    prev = json.load(open("benchmarks/literature/results/ambench2018_tracks_ebsd.json"))["agreement"]
    prev3 = json.load(open("benchmarks/literature/results/ambench2018_tracks_ebsd3d.json"))
    hyb = [out[f"hybrid {s}"]["frac_lt10"] for s in range(5)]
    hyb8 = [v["frac_lt10"] for k, v in out.items() if "um grains" in k]
    bars = [("random orientations", 100 * prev["random orientations"]["frac_lt10"], None, MUTED),
            ("copy nearest fusion-line grain (2-D)", 100 * prev["normal epitaxy"]["frac_lt10"], None, MUTED),
            ("reduced CA, competitive rule (2-D)", 100 * prev["in-plane"]["frac_lt10"], None, MUTED),
            ("gradient trace (2-D)", 100 * prev["gradient trace"]["frac_lt10"], None, MUTED),
            ("3-D gradient trace, hybrid substrate, 2.7 um grains", 100 * prev3["agreement_3d"]["all"][0],
             100 * prev3["agreement_3d"]["all"][1], MUTED),
            ("3-D CA, hybrid substrate, 2.7 um grains (5 runs)", 100 * np.mean(hyb), 100 * np.std(hyb), BLUE),
            ("3-D CA, hybrid substrate, 7.9 um grains (3 runs)", 100 * np.mean(hyb8), 100 * np.std(hyb8), BLUE),
            ("3-D CA, extruded substrate, fast tip kinetics", 100 * out["extruded, fast kinetics (5 K at 0.1 m/s)"]["frac_lt10"],
             None, ORANGE),
            ("3-D CA, extruded substrate, nucleation 1e15 /m3", 100 * nz["frac_lt10"], None, ORANGE),
            ("3-D CA, extruded substrate, reversed scan", 100 * out["extruded, reversed scan"]["frac_lt10"], None, ORANGE),
            ("3-D CA, extruded substrate, 0.25 um cells", 100 * out["extruded, 0.25 um cells"]["frac_lt10"], None, ORANGE),
            ("3-D CA, extruded substrate", 100 * out["extruded"]["frac_lt10"], None, ORANGE)]
    bars = [b for b in bars if np.isfinite(b[1])]
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], xerr=[b[2] or 0 for b in bars], color=[b[3] for b in bars], height=0.62,
            error_kw=dict(lw=1, capsize=2, ecolor=INK))
    for yi, b in zip(y, bars):
        ax.text(b[1] + (b[2] or 0) + 0.4, yi, f"{b[1]:.1f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=7.5)
    ax.set_xlabel("melt-pool pixels within 10 deg of the measured orientation [%]", fontsize=8.5)
    ax.set_title("(g) point-wise agreement with the EBSD section", fontsize=9, loc="left")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.set_xlim(0, 32)

    # grain size / count
    ax = fig.add_axes([0.745, 0.06, 0.2, 0.26])
    names = ["measured", "extruded", "hybrid 2.7 um", "hybrid 7.9 um"]
    cnt = [out["extruded"]["stats"]["measured"]["grains"], out["extruded"]["stats"]["CA"]["grains"],
           np.mean([out[f"hybrid {s}"]["stats"]["CA"]["grains"] for s in range(5)]),
           np.mean([v["stats"]["CA"]["grains"] for k, v in out.items() if "um grains" in k])]
    dd = [out["extruded"]["stats"]["measured"]["area_weighted_d_eq_um"], out["extruded"]["stats"]["CA"]["area_weighted_d_eq_um"],
          np.mean([out[f"hybrid {s}"]["stats"]["CA"]["area_weighted_d_eq_um"] for s in range(5)]),
          np.mean([v["stats"]["CA"]["area_weighted_d_eq_um"] for k, v in out.items() if "um grains" in k])]
    xb = np.arange(4)
    ax.bar(xb - 0.2, cnt, 0.38, color=[INK, ORANGE, BLUE, BLUE], label="grains in the pool section")
    ax2 = ax.twinx()
    ax2.plot(xb + 0.2, dd, "D", color=AQUA, ms=7, label="area-weighted size [um]")
    ax.set_xticks(xb)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("grains (10 deg)", fontsize=8)
    ax2.set_ylabel("area-weighted equivalent diameter [um]", fontsize=8, color=AQUA)
    ax2.set_ylim(0, 18)
    ax.set_title("(h) grain count (bars) and size (diamonds)", fontsize=9, loc="left")
    for a in (ax, ax2):
        a.tick_params(labelsize=7.5, colors=MUTED)
        a.spines["top"].set_visible(False)
    fig.suptitle("3-D grain cellular automaton (decentred octahedra, KGT kinetics) regrowing the AM-Bench 2018-02 "
                 "case-B melt pool, compared with NIST's EBSD section", fontsize=10.5, x=0.02, ha="left")
    fig.savefig(M.FIG, dpi=150)
    print("wrote", M.FIG)
