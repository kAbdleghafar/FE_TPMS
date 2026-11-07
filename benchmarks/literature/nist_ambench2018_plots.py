"""Figures for the AM-Bench 2018 comparison (docs/img/ambench2018_*.png)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OUT = "results/literature/ambench2018"
IMG = "docs/img"
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# diverging: blue (compression) <-> neutral grey <-> red (tension)
DIV = LinearSegmentedColormap.from_list("bgr", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9c2424"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})


def contour_maps():
    r = json.load(open(os.path.join(OUT, "analysis_plate.json")))["kappa=0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True, constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-600, vmax=600)
    for ax, key, title in ((axes[0], "measured", "Measured (contour method, UC Davis)"),
                           (axes[1], "predicted", "tpmsfem (calibrated on deflection only)")):
        for pts in (r["profile"], r["contour_plate_points"]):
            y, z, v = np.array(pts["y"]), np.array(pts["z"]), np.array(pts[key], float)
            ok = np.isfinite(v)
            sc = ax.scatter(y[ok] + 4.5, z[ok] + 12.85, c=v[ok], cmap=DIV, norm=norm, s=7, marker="s", linewidths=0)
        ax.axhline(12.85, color=INK2, lw=0.6, ls=(0, (3, 3)))
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Y [mm]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Z [mm]  (plate 0-12.85, part above)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, label="σ_xx [MPa]")
    cb.outline.set_visible(False)
    fig.suptitle("AM-Bench 2018 IN625 bridge, section x = 31 mm: residual stress σ_xx", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "ambench2018_contour.png"), dpi=200)


def neutron_lines():
    r = json.load(open(os.path.join(OUT, "analysis_plate.json")))["kappa=0"]["neutron_points"]
    x, z = np.array(r["x"]), np.array(r["z"])
    levels = [11.248, 6.856, 1.0]
    fig, axes = plt.subplots(len(levels), 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    for ax, zl in zip(axes, levels):
        sel = np.abs(z - zl) < 1e-6
        o = np.argsort(x[sel])
        xs, ps = x[sel][o], 1e6 * np.array(r["exx_pred"], float)[sel][o]
        gap = np.r_[False, np.diff(xs) > 2.0]          # do not draw the prediction across unmeasured gaps
        xs_b = np.insert(xs, np.flatnonzero(gap), np.nan)
        ps_b = np.insert(ps, np.flatnonzero(gap), np.nan)
        ax.plot(xs_b, ps_b, color=BLUE, lw=2, marker="o", ms=3, label="tpmsfem (predicted)")
        ax.plot(x[sel][o], 1e6 * np.array(r["exx_meas"])[sel][o], "o", color=ORANGE, ms=4.5,
                markeredgecolor=SURF, markeredgewidth=0.8, label="neutron diffraction (NIST)")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylabel("ε_xx [µε]")
        ax.set_title(f"z = {zl:g} mm above the plate", fontsize=9, loc="left")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[-1].set_xlabel("x along the bridge [mm]")
    fig.suptitle("Longitudinal elastic strain: prediction vs neutron diffraction", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "ambench2018_neutron.png"), dpi=200)


def deflection():
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    xr = [0.5 + 7 * k for k in range(11)]
    for v, color, label in (("plate", BLUE, "elastic plate strip, h = 0.5 mm"),
                            ("rigid", ORANGE, "rigid plate, h = 0.25 mm")):
        d = json.load(open(os.path.join(OUT, f"analysis_{v}.json")))["kappa=0"]["deflection"]
        ax.plot(xr, d, "-o", color=color, lw=2, ms=4, label=f"tpmsfem, {label}")
    ax.plot([0.5], [1.276], "D", color=INK, ms=7, label="measured, ridge 1 (calibration point)")
    ax.set_xlabel("ridge position x [mm]")
    ax.set_ylabel("deflection after leg cut [mm]")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("Ridge deflection after EDM separation of the 12 legs", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "ambench2018_deflection.png"), dpi=200)


def tm_comparison(tag="base"):
    """Thermo-mechanical (no calibration) vs elastic inherent strain (calibrated) vs measurement:
    neutron eps_xx along the legs line and the body lines, and the ridge deflections."""
    path = os.path.join(OUT, "tm", f"analysis_{tag}.json")
    if not os.path.exists(path):
        return
    tm = json.load(open(path))
    ism = json.load(open(os.path.join(OUT, "analysis_plate.json")))["kappa=0"]
    x, z = np.array(tm["neutron_points"]["x"]), np.array(tm["neutron_points"]["z"])
    levels = [11.248, 6.856, 1.0]
    fig, axes = plt.subplots(len(levels) + 1, 1, figsize=(7.2, 8.4), constrained_layout=True)
    for ax, zl in zip(axes, levels):
        sel = np.flatnonzero(np.abs(z - zl) < 1e-6)
        o = sel[np.argsort(x[sel])]
        xs = x[o]
        gap = np.r_[False, np.diff(xs) > 2.0]
        for res, color, label in ((tm, BLUE, "tpmsfem thermo-mechanical (no calibration)"),
                                  (ism, "#8a8984", "tpmsfem elastic inherent strain (calibrated)")):
            ps = 1e6 * np.array(res["neutron_points"]["exx_pred"], float)[o]
            ax.plot(np.insert(xs, np.flatnonzero(gap), np.nan), np.insert(ps, np.flatnonzero(gap), np.nan),
                    color=color, lw=2 if res is tm else 1.4, marker="o", ms=3 if res is tm else 0, label=label)
        ax.plot(xs, 1e6 * np.array(tm["neutron_points"]["exx_meas"], float)[o], "o", color=ORANGE, ms=4.5,
                markeredgecolor=SURF, markeredgewidth=0.8, label="neutron diffraction (NIST)")
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_ylabel("ε_xx [µε]")
        ax.set_title(f"z = {zl:g} mm above the plate", fontsize=9, loc="left")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7.5)
    axes[len(levels) - 1].set_xlabel("x along the bridge [mm]")
    ax = axes[-1]
    xr = [0.5 + 7 * k for k in range(11)]
    ax.plot(xr, tm["deflection"], "-o", color=BLUE, lw=2, ms=4, label="thermo-mechanical (no calibration)")
    ax.plot(xr, ism["deflection"], "-", color="#8a8984", lw=1.4, label="elastic inherent strain (calibrated on ridge 1)")
    ax.plot([0.5], [1.276], "D", color=INK, ms=7, label="measured, ridge 1")
    ax.set_xlabel("ridge position x [mm]")
    ax.set_ylabel("deflection after leg cut [mm]")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("AM-Bench 2018 IN625 bridge: thermo-mechanical prediction", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "ambench2018_tm.png"), dpi=200)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    contour_maps()
    neutron_lines()
    deflection()
    tm_comparison()
    print("written:", sorted(f for f in os.listdir(IMG) if f.startswith("ambench")))
