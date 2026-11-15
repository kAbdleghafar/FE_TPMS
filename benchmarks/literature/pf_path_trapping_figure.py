"""Figure for pf_path_trapping.py."""
import glob
import json

import numpy as np

import pf_path_trapping as M
from tpmsfem.process import phasefield as PF

INK, MUTED, GRID, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _sp(r):
    """Primary spacing from the counted cell boundaries (as in 5.3); active tips if no
    boundaries are resolved (weakly segregated cells near absolute stability)."""
    v = r["spacing_um"]
    return v if np.isfinite(v) and v < 0.9 * r["width_um"] else r["spacing_tips_um"]


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    store = json.load(open(M.STORE))
    old = json.load(open(M.PATH_STORE))
    pts = M.path_points()
    z0 = min(p["z_um"] for p in pts)
    jobs = M.jobs()
    main = [(jobs[t][0], store[t]) for t in jobs if t.endswith("|trap") and t in store]
    main.sort()
    h = np.array([z - z0 for z, _ in main])
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.38, wspace=0.3, left=0.06, right=0.98, top=0.91,
                          bottom=0.07)

    # (a) path conditions and the stability limits
    ax = fig.add_subplot(gs[0, 0])
    hp = np.array([p["z_um"] - z0 for p in pts])
    R = np.array([p["R"] for p in pts])
    ax.plot(hp, R, "o-", color=BLUE, lw=1.6, ms=4, label="growth rate R")
    vabs_eq = PF.absolute_stability_velocity(M.ALLOY)
    vabs_tr = [PF.absolute_stability_velocity(M.ALLOY.at_velocity(r, M.V_D)) for r in R]
    ax.axhline(vabs_eq, color=MUTED, ls="--", lw=1)
    ax.text(hp[0], vabs_eq * 1.04, "absolute stability, equilibrium k", fontsize=7.5, color=MUTED)
    ax.plot(hp, vabs_tr, "--", color=ORANGE, lw=1.4, label="absolute stability with trapping")
    ax.set_xlabel("height above the fusion boundary [um]", fontsize=8.5)
    ax.set_ylabel("velocity [m/s]", fontsize=8.5)
    ax.set_title("(a) growth rate up the case-B centreline", fontsize=9, loc="left")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax2 = ax.twinx()
    ax2.semilogy(hp, [p["G"] for p in pts], "s:", color=AQUA, ms=3.5, lw=1)
    ax2.set_ylabel("G [K/m] (squares)", fontsize=8, color=AQUA)
    ax2.tick_params(labelsize=7.5, colors=AQUA)
    _style(ax)

    # (b) spacing
    ax = fig.add_subplot(gs[0, 1])
    ax.axhspan(*M.MEASURED["spacing_um"], color="#d6d5d0", lw=0)
    ax.axhspan(*M.MEASURED["pf_spacing_um"], facecolor="none", edgecolor="#8f8e89", hatch="///", lw=0)
    sp = np.array([_sp(r) for _, r in main])
    ax.plot(h, sp, "o-", color=BLUE, lw=1.6, ms=5, label="trapping, D/V = 5 W0")
    oz = [(p["z_um"] - z0, old["pf|z=%.1f" % p["z_um"]]) for p in pts if ("pf|z=%.1f" % p["z_um"]) in old]
    ax.plot([a for a, _ in oz], [r["spacing_um"] for _, r in oz], "o--", color=MUTED, ms=4, lw=1,
            label="no trapping, W0 = 10 nm (5.3)")
    conv = sorted([(store[t]["W0_nm"], store[t]) for t in store if t.startswith("z=-22.5|trap")])
    for i, (W, r) in enumerate(conv):
        ax.plot(r["z_um"] - z0 + 0.8, _sp(r), "D", color=ORANGE, ms=5, mfc="none")
        ax.annotate(f"W0 {W:.1f} nm", (r["z_um"] - z0 + 0.8, _sp(r)), xytext=(12.5, 0.05 + 0.065 * i), fontsize=6.5,
                    color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
    ax.plot(h, [r["hunt_um"] for _, r in main], ":", color=AQUA, lw=1.4, label="Hunt (with k(V))")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("height above the fusion boundary [um]", fontsize=8.5)
    ax.set_ylabel("primary cell spacing [um]", fontsize=8.5)
    ax.set_title("(b) cell spacing (diamonds: W0 convergence)", fontsize=9, loc="left")
    hnd, lab = ax.get_legend_handles_labels()
    hnd += [Patch(color="#d6d5d0"), Patch(facecolor="none", edgecolor="#8f8e89", hatch="///")]
    lab += ["measured (SEM, Ghosh et al. 2018)", "Ghosh et al. 2018 phase field"]
    ax.legend(hnd, lab, fontsize=7, frameon=False, loc="upper right")
    _style(ax)

    # (c) segregation
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(M.ALLOY.c_inf, color=MUTED, lw=1, ls="--")
    ax.text(h.max(), M.ALLOY.c_inf + 0.05, "alloy (4 wt%)", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.plot(h, [r["cs_tip_wt"] for _, r in main], "o-", color=BLUE, lw=1.6, ms=5, label="cell core c_s*, trapping")
    ax.plot(h, [r["cl_tip_wt"] for _, r in main], "s-", color=ORANGE, lw=1.6, ms=5, label="liquid at the tip c_l*, trapping")
    ax.plot([a for a, _ in oz], [r["cs_tip_wt"] for _, r in oz], "o--", color=MUTED, ms=4, lw=1, label="c_s*, no trapping (5.3)")
    ax.plot([a for a, _ in oz], [r["cl_tip_wt"] for _, r in oz], "s--", color=MUTED, ms=4, lw=1, label="c_l*, no trapping (5.3)")
    eq = [t for t in store if t.endswith("|eq")]
    for t in eq:
        r = store[t]
        ax.plot([r["z_um"] - z0], [r["cs_tip_wt"]], "o", mfc="white", mec=INK, ms=6)
        ax.plot([r["z_um"] - z0], [r["cl_tip_wt"]], "s", mfc="white", mec=INK, ms=6)
        ax.annotate("same W0,\nno trapping", (r["z_um"] - z0, r["cs_tip_wt"]), xytext=(r["z_um"] - z0 - 6.5, 3.05),
                    fontsize=6.5, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    ax.axhline(M.MEASURED["pf_cs_wt"], color=INK, lw=0.8, ls=":")
    ax.text(0, M.MEASURED["pf_cs_wt"] + 0.05, "Ghosh 2018 cores, 3.7 wt%", fontsize=7, color=INK)
    ax.set_xlabel("height above the fusion boundary [um]", fontsize=8.5)
    ax.set_ylabel("Nb [wt%]", fontsize=8.5)
    ax.set_title("(c) Nb partitioning at the cell tips", fontsize=9, loc="left")
    ax.legend(fontsize=6.8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_ylim(2.8, 6.4)
    _style(ax)

    # (d, e) composition near the tips: lowest and highest point
    files = {f: f for f in glob.glob(M.FIELDS.replace(".npz", "_*.npz"))}
    tags = [t for t in (f"z={main[0][0]:.1f}|trap", f"z={main[-1][0]:.1f}|trap") if
            M.FIELDS.replace(".npz", f"_{M._safe(t)}.npz") in files]
    for i, t in enumerate(tags):
        d = np.load(M.FIELDS.replace(".npz", f"_{M._safe(t)}.npz"))
        c = d["c"].astype(float)
        dx = float(d["dx"]) * 1e6
        ax = fig.add_subplot(gs[1, 0:2] if i == 0 else gs[1, 2])
        im = ax.imshow(c, origin="lower", cmap="magma", vmin=2.5, vmax=7.0, extent=[0, c.shape[1] * dx, 0, c.shape[0] * dx],
                       aspect="equal")
        r = store[t]
        ax.set_title(f"({'d' if i == 0 else 'e'}) Nb near the tips, {r['z_um'] - z0:.0f} um up: R = {r['R']:.3f} m/s,\n"
                     f"k(V) = {r['k']:.2f}, W0 = {r['W0_nm']:.1f} nm", fontsize=9, loc="left")
        ax.set_xlabel("um", fontsize=8)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
        cb.set_label("Nb [wt%]", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Phase field up the AM-Bench case-B melt pool with solute trapping (continuous growth model, V_D = 1 m/s), "
                 "W0 scaled so that D/V = 5 W0", fontsize=10.5, x=0.02, ha="left")
    fig.savefig(M.FIG, dpi=150)
    print("wrote", M.FIG)
