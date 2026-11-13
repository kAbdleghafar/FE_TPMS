"""Figure for the powder-track comparison (tools/lbm/freesurface_powder_track.py)."""
import json

import numpy as np

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
OUT = "docs/img/freesurface_powder_track.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.load(open("tools/lbm/powder_track.json"))
    d = {c: np.load(f"tools/lbm/powder_track_{c}.npz") for c in ("powder", "plate") if c in s}
    h = float(d["powder"]["h"]) * 1e6
    pt = float(d["powder"]["plate_top"])
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28, left=0.05, right=0.97, top=0.9, bottom=0.08)

    def hmap(ax, srf, title, vmin=-20, vmax=60):
        a = (srf[1:-1, 1:-1] - pt) * 1e6
        im = ax.imshow(a.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                       extent=[0, a.shape[0] * h, 0, a.shape[1] * h], aspect="equal")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("x [um]", fontsize=8.5)
        ax.set_ylabel("y [um]", fontsize=8.5)
        ax.tick_params(labelsize=8, colors=MUTED)
        return im

    p = s["powder"]
    hmap(fig.add_subplot(gs[0, 0]), d["powder"]["surface0"],
         f"(a) powder layer before the laser\n{p['particles']} particles, packing {p['packing']:.2f}, 40 um recoater")
    im = hmap(fig.add_subplot(gs[0, 1]), d["powder"]["surface"], "(b) after one track on powder\n200 W, 0.8 m/s, left to right")
    if "plate" in d:
        hmap(fig.add_subplot(gs[0, 2]), d["plate"]["surface"], "(c) the same track on the bare plate")
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.012, pad=0.01)
    cb.set_label("surface height above the plate [um]", fontsize=8)
    # (d) centreline height
    ax = fig.add_subplot(gs[1, 0:2])
    for c, col, lab in (("powder", ORANGE, "on powder"), ("plate", BLUE, "on the bare plate")):
        if c not in d:
            continue
        srf = d[c]["surface"]
        j = srf.shape[1] // 2
        x = (np.arange(srf.shape[0] - 2) + 0.5) * h
        ax.plot(x, (srf[1:-1, j] - pt) * 1e6, color=col, lw=1.8, label=lab)
        if c == "powder":
            ax.plot(x, (d[c]["surface0"][1:-1, j] - pt) * 1e6, color=ORANGE, lw=0.8, ls=":", label="powder before")
    ax.axvspan(100, 500, color=GRID, lw=0, zorder=0)
    ax.text(105, 52, "laser on", fontsize=7.5, color=MUTED)
    ax.axhline(40, color=MUTED, lw=0.8, ls="--")
    ax.text(5, 42, "recoater (40 um)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("x along the track [um]", fontsize=8.5)
    ax.set_ylabel("surface height on the track centre [um]", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("(d) track profile along its centre", fontsize=9.5, loc="left")
    # (e) cross-section of the final metal, middle of the track
    ax = fig.add_subplot(gs[1, 2])
    for c, col in (("powder", ORANGE), ("plate", BLUE)):
        if c not in d:
            continue
        f = d[c]["fill"].astype(float)
        i = int(round(300 / h)) + 1
        sec = f[i, 1:-1, 1:-1]
        y = (np.arange(sec.shape[0]) + 0.5) * h
        z = (np.arange(sec.shape[1]) + 0.5) * h - pt * 1e6 + h
        ax.contour(y, z, sec.T, [0.5], colors=[col], linewidths=1.6)
        Tp = d[c]["T_peak"][i, 1:-1, 1:-1]
        ax.contour(y, z, Tp.T, [1290.0], colors=[col], linewidths=1.0, linestyles="--")
    ax.plot([], [], color=MUTED, lw=1.6, label="metal surface")
    ax.plot([], [], color=MUTED, lw=1.0, ls="--", label="melted (peak T = solidus)")
    ax.set_xlim(40, 210)
    ax.set_ylim(-60, 70)
    ax.set_aspect("equal")
    ax.set_xlabel("y [um]", fontsize=8.5)
    ax.set_ylabel("z above the plate [um]", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("(e) cross-section at x = 300 um\norange: powder, blue: plate", fontsize=9.5, loc="left")
    fig.suptitle("One LPBF track on a 40 um powder layer: free-surface lattice Boltzmann melt pool (IN625, 5 um cells)",
                 fontsize=11, x=0.02, ha="left")
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
