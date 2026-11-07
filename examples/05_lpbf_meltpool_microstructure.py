"""Laser scan -> melt pool -> solidification -> grains, for IN625 (AM-Bench 2018-02 settings).

1. single track, case B (179.2 W, 800 mm/s, D4sigma 170 um) with the absorptivity and the
   absorption depth calibrated on NIST's case-B cross-section
   (benchmarks/literature/nist_ambench2018_tracks.py --calibrate);
2. G-R solidification map, cell/dendrite spacing, Hunt CET;
3. reduced-CA epitaxial grain growth from a random substrate, inverse-pole-figure maps;
4. export of the thermal history for ExaCA;
5. a 5-track bidirectional hatch on a 20 um powder layer (the AM-Bench layer thickness):
   remelting of neighbouring tracks, grains of the new layer.

    python examples/05_lpbf_meltpool_microstructure.py      # ~15 min on 4 CPU cores
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "literature"))
import in625_material as IM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector, raster  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402
from tpmsfem.process.meltpool import fusion_zone  # noqa: E402

OUT = "results/process"
IMG = "docs/img"
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
MAT = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())


def calibration():
    d = json.load(open("benchmarks/literature/results/ambench2018_tracks.json"))
    return d["calibration"]


def single_track(cal):
    md = MeltPoolModel(MAT, cal["h"], length=1.2, width=0.2, depth=0.12, T0=25.0,
                       liquid_k_factor=(1.0, 1.0, cal["f_z"]), emissivity=0.4, evaporation=Evaporation())
    x1 = 0.95
    res = md.run(Laser(179.2, cal["A"], 0.085, depth=cal.get("delta", 0.0)), [ScanVector(0.15, 0.0, x1, 0.0, 800.0)], t_extra=3e-4,
                 snapshot_times=[(x1 - 0.15) / 800.0 - 1e-9])
    return res


def hatch(cal):
    md = MeltPoolModel(MAT, cal["h"], length=0.9, width=0.7, depth=0.12, powder=0.02, symmetric_y=False, T0=25.0,
                       liquid_k_factor=(1.0, 1.0, cal["f_z"]), emissivity=0.4, evaporation=Evaporation())
    path = raster(0.15, 0.75, -0.2, 0.2, hatch=0.1, speed=800.0, power=179.2, along="x")
    return md.run(Laser(179.2, cal["A"], 0.085, depth=cal.get("delta", 0.0)), path, t_extra=3e-4)


def figures(res, gr, hres, hgr, summ):
    os.makedirs(IMG, exist_ok=True)
    i = int(np.argmin(np.abs(res.x - 0.7)))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    # (a) temperature snapshot, top view + fusion boundary
    ax = axes[0, 0]
    Tsnap = res.snapshots[0][1]
    yy = np.r_[-res.y[::-1], res.y]
    top = np.c_[Tsnap[:, ::-1, -1], Tsnap[:, :, -1]].T
    im = ax.imshow(top, origin="lower", extent=[res.x[0], res.x[-1], yy[0], yy[-1]], cmap="inferno", vmin=25,
                   vmax=3000, aspect="equal")
    ax.contour(res.x, yy, top, levels=[res.T_solidus], colors=["#ffffff"], linewidths=0.8)
    ax.set_xlim(0.4, 0.98)
    ax.set_ylim(-0.12, 0.12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("(a) Surface temperature; white line: solidus", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7, label="T [°C]", location="bottom")
    # (b) cross-section: peak temperature, fusion line vs NIST width/depth
    ax = axes[0, 1]
    Tp = np.c_[res.T_peak[i][::-1].T, res.T_peak[i].T] if False else np.vstack([res.T_peak[i][::-1], res.T_peak[i]]).T
    ax.contourf(1e3 * yy, 1e3 * res.z, Tp, levels=[25, 400, 800, 1290, 2000, 3500], cmap="magma")
    ax.contour(1e3 * yy, 1e3 * res.z, Tp, levels=[res.T_solidus], colors=[BLUE], linewidths=1.6)
    w, d = 123.5, 36.0
    ax.add_patch(plt.Rectangle((-w / 2, -d), w, d, fill=False, ls="--", lw=1.0, ec=ORANGE))
    ax.set_xlim(-120, 120)
    ax.set_ylim(-80, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("y [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title("(b) Cross-section: fusion line (blue),\nNIST measured width × depth (dashed)", fontsize=8.5, loc="left")
    # (c) G-R map
    ax = axes[1, 0]
    ok, G, R, cr = S.gr_fields(res)
    ax.loglog(R[ok], G[ok], ".", color=BLUE, ms=2, alpha=0.4)
    gg = np.logspace(5, 8.5, 50)
    for q in (1e5, 1e6, 1e7):
        ax.plot(q / gg, gg, color=GRID, lw=0.8)
        ax.text(q / gg[0] * 0.9, gg[0] * 1.3, f"{q:.0e} K/s", fontsize=6.5, color=MUTED, ha="right")
    ax.set_xlabel("solidification velocity R [m/s]")
    ax.set_ylabel("thermal gradient G [K/m]")
    ax.set_title(f"(c) G and R at solidification\n(median cooling rate {summ['cooling_rate_K_per_s']['p50']:.1e} K/s)",
                 fontsize=8.5, loc="left")
    # (d) grains: IPF (build direction) in the cross-section
    ax = axes[1, 1]
    g = gr.grain[i]                                       # (ny, nz)
    rgb = S.ipf_color(gr.rotations[np.maximum(g, 0)], (0, 0, 1))
    rgb[~np.isfinite(rgb).all(-1)] = 1
    full = np.concatenate([rgb[::-1], rgb], axis=0).transpose(1, 0, 2)
    ax.imshow(full, origin="lower", extent=[1e3 * yy[0], 1e3 * yy[-1], 1e3 * res.z[0], 1e3 * res.z[-1]],
              interpolation="nearest")
    ax.contour(1e3 * yy, 1e3 * res.z, Tp, levels=[res.T_solidus], colors=[INK], linewidths=0.8)
    ax.set_xlim(-120, 120)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("y [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title("(d) Grains (IPF colour, build direction);\nblack line: fusion line", fontsize=8.5, loc="left")
    fig.suptitle("Laser track, AM-Bench 2018-02 case B (179 W, 800 mm/s), IN625", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "meltpool_track_B.png"), dpi=200)

    # hatch: top view of grains and number of melting events
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    kp = int(np.argmin(np.abs(hres.z - 0.0075)))          # inside the (former) powder layer
    ks = int(np.argmin(np.abs(hres.z + 0.0025)))          # first plate cell below the powder
    rgb = S.ipf_color(hgr.rotations[np.maximum(hgr.grain[:, :, kp], 0)], (0, 0, 1))
    rgb[hres.n_melt[:, :, kp] == 0] = 1.0                 # powder that never melted
    axes[0].imshow(rgb.transpose(1, 0, 2), origin="lower", extent=[hres.x[0], hres.x[-1], hres.y[0], hres.y[-1]],
                   interpolation="nearest")
    axes[0].set_title("New layer, 7.5 µm above the plate: grains (IPF z);\nwhite = unmelted powder", fontsize=8.5,
                      loc="left")
    im = axes[1].imshow(hres.n_melt[:, :, ks].T, origin="lower", extent=[hres.x[0], hres.x[-1], hres.y[0], hres.y[-1]],
                        cmap="Blues", vmin=0, vmax=max(1, int(hres.n_melt.max())))
    axes[1].set_title("Plate 2.5 µm below the powder:\nnumber of melting events", fontsize=8.5, loc="left")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.suptitle("5-track bidirectional hatch, 100 µm spacing, 20 µm powder layer", fontsize=10, x=0.02, ha="left")
    fig.savefig(os.path.join(IMG, "meltpool_hatch.png"), dpi=200)


def main():
    os.makedirs(OUT, exist_ok=True)
    cal = calibration()
    print("calibration", cal, flush=True)
    res = single_track(cal)
    w, d = fusion_zone(res, 0.7)
    summ = S.summary(res)
    gr = S.grow_grains(res, substrate_grain_size=0.03, seed=2)
    gst = S.grain_stats(gr, res, x_section=0.7)
    n = S.export_exaca(res, os.path.join(OUT, "exaca_track_B.csv"))
    print(f"track B: width {1e3 * w:.1f} um, depth {1e3 * d:.1f} um, {res.steps} steps, {res.seconds:.0f}s", flush=True)
    print("solidification", json.dumps(summ, indent=1), flush=True)
    print("grains", gst, f"ExaCA rows {n}", flush=True)
    hres = hatch(cal)
    hgr = S.grow_grains(hres, substrate_grain_size=0.03, seed=3)
    hst = S.grain_stats(hgr, hres)
    hsumm = S.summary(hres)
    print(f"hatch: {hres.steps} steps, {hres.seconds:.0f}s, max melting events {int(hres.n_melt.max())}", flush=True)
    print("hatch grains", hst, flush=True)
    json.dump(dict(track_B=dict(width_um=1e3 * w, depth_um=1e3 * d, solidification=summ, grains=gst, exaca_rows=n),
                   hatch=dict(solidification=hsumm, grains=hst, max_melt_events=int(hres.n_melt.max()))),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    figures(res, gr, hres, hgr, summ)


if __name__ == "__main__":
    main()
