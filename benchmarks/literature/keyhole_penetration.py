"""Closed keyhole 3 (map): where does the keyhole break through the solid under a layer of the
CoCrFeMnNi gyroid into the powder below?

In welding terms the keyhole of an LPBF track is "closed" (blind, partial penetration): its tip
ends in metal. On the down-facing side of a lattice wall, the solid under the new layer is
thin: there the keyhole can reach through it into the loose powder below ("open", full
penetration). The melt then wets the powder (dross, sagging) and the keyhole vents into the
powder's gas, which feeds pores and spatter.

Method. The graded gyroid sheet of the study (4 mm cell, walls 0.2 -> 0.6 mm along x), sliced
in 30 um layers. For every point of a layer that is scanned (inside the sheet at mid-layer), the
solid below the layer is the distance down to the first point outside the sheet (5 um
resolution). The keyhole reaches the powder there if its depth below the top of the powder
layer exceeds the layer plus that solid:
* mean depth e from Gan et al.'s laws (tpmsfem.process.defects), at the study's 200 W,
  0.75 m/s, r0 35 um, eta_m 0.33, CoCrFeMnNi liquid properties, part temperature T0;
* deepest excursion e + r0 Delta e* / 2 (the fluctuation law): intermittent break-through;
* for comparison, the 54 um pool of the conduction model (no keyhole, process_models.md §7).
Spot radii 35 and 45 um, part temperatures 80 and 300 C. Heights: 16 layers through one
unit cell (z = 0.5 ... 4.25 mm), so every wall orientation appears.

    python benchmarks/literature/keyhole_penetration.py          # ~2 min
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyhole_maps as KM  # noqa: E402
from cantor_gyroid_solidification import L_CELL, LAYER, gyroid  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "results", "keyhole_penetration.json")
FIG = os.path.join(HERE, "..", "..", "docs", "img", "keyhole_penetration.png")
H = 0.005                                  # mm, lateral and vertical sampling
DMAX = 0.6                                 # mm, deepest solid looked at below a layer
HEIGHTS = np.arange(0.5, 4.5, 0.25)        # mm, top of the new layer
CONDUCTION_DEPTH = 0.054                   # mm, conduction-model pool (from the powder top), §7


def solid_below(gy, z_top, x, y):
    """For the layer whose top is at z_top: scanned mask (inside at mid-layer) and the solid
    thickness below the layer [mm] (DMAX where it reaches deeper)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    scanned = np.asarray(gy(X, Y, np.full_like(X, z_top - 0.5 * LAYER))) < 0.0
    below = np.full(X.shape, DMAX)
    open_ = np.ones(X.shape, bool)                 # still in solid going down
    zb = z_top - LAYER
    for k, u in enumerate(np.arange(0.5 * H, DMAX, H)):
        if zb - u <= 0:                            # the build plate
            break
        inside = np.asarray(gy(X, Y, np.full_like(X, zb - u))) < 0.0
        hit = open_ & ~inside
        below[hit] = u - 0.5 * H
        open_ &= inside
        if not open_[scanned].any():
            break
    return scanned, below


def keyhole_cases():
    cases = {}
    for r0 in (35e-6, 45e-6):
        for T0 in (80.0, 300.0):
            st = KM.state("Cantor", 200.0, 0.75, r0, T0, 0.33)
            e = float(st["depth"]) * 1e3
            e_max = float(st["depth_max"]) * 1e3
            cases[f"r{r0 * 1e6:.0f}_T{T0:.0f}"] = dict(r0_um=r0 * 1e6, T0=T0, Ke=float(st["Ke"]), e_mm=e, e_max_mm=e_max)
    return cases


def run():
    gy = gyroid()
    x = np.arange(0.5 * H, L_CELL, H)
    cases = keyhole_cases()
    thick = 0.2 + 0.4 * np.clip(x / L_CELL, 0.0, 1.0)         # graded wall thickness along x
    bins = np.linspace(0.2, 0.6, 5)
    rows, maps = [], {}
    for z in HEIGHTS:
        scanned, below = solid_below(gy, z, x, x)
        reach = LAYER + below                                   # powder this far below the powder top
        row = dict(z=float(z), scanned_mm2=float(scanned.sum() * H * H),
                   over_powder=float((scanned & (below < H)).sum() / scanned.sum()))
        for name, c in cases.items():
            for lab, depth in (("mean", c["e_mm"]), ("max", c["e_max_mm"])):
                pen = scanned & (reach < depth)
                row[f"{name}_{lab}"] = float(pen.sum() / scanned.sum())
                # by wall thickness (x bands)
                row[f"{name}_{lab}_by_t"] = [float((pen & (np.digitize(thick, bins) == b + 1)[:, None]).sum()
                                                   / max((scanned & (np.digitize(thick, bins) == b + 1)[:, None]).sum(), 1))
                                             for b in range(len(bins) - 1)]
        row["conduction"] = float((scanned & (reach < CONDUCTION_DEPTH)).sum() / scanned.sum())
        rows.append(row)
        if abs(z - 1.3) < 0.13 or abs(z - 2.5) < 0.13:
            maps[f"{z:.2f}"] = dict(scanned=scanned, below=below)
        print(f"z {z:.2f} mm: over powder {100 * row['over_powder']:.1f} %, keyhole reaches powder "
              f"{100 * row['r35_T80_mean']:.1f} % (mean), {100 * row['r35_T80_max']:.1f} % (deepest); "
              f"conduction pool {100 * row['conduction']:.1f} %", flush=True)
    tot = {}
    for name in cases:
        for lab in ("mean", "max"):
            k = f"{name}_{lab}"
            tot[k] = float(np.average([r[k] for r in rows], weights=[r["scanned_mm2"] for r in rows]))
            tot[k + "_by_t"] = [float(np.average([r[k + "_by_t"][b] for r in rows], weights=[r["scanned_mm2"] for r in rows]))
                                for b in range(len(bins) - 1)]
    tot["conduction"] = float(np.average([r["conduction"] for r in rows], weights=[r["scanned_mm2"] for r in rows]))
    tot["over_powder"] = float(np.average([r["over_powder"] for r in rows], weights=[r["scanned_mm2"] for r in rows]))
    out = dict(cases=cases, thickness_bins_mm=bins.tolist(), layers=rows, total=tot, layer_mm=LAYER, h_mm=H)
    json.dump(out, open(STORE, "w"), indent=1)
    np.savez_compressed(STORE.replace(".json", ".npz"), x=x,
                        **{f"{k}_{a}": v[a] for k, v in maps.items() for a in ("scanned", "below")})
    print("total (area-weighted):", {k: round(100 * v, 2) for k, v in tot.items() if not k.endswith("_by_t")})
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    INK, MUTED, GRID, BLUE, ORANGE, RED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#c8302c"
    s = json.load(open(STORE))
    z = np.load(STORE.replace(".json", ".npz"))
    x = z["x"]
    c = s["cases"]["r35_T80"]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.05, 1.05]))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.8, bottom=0.2, wspace=0.32)
    cmap = ListedColormap([RED, ORANGE, "#f6d7b0", "#d9e6f5"])
    for ax, key, lab in ((axs[0], "1.25", "(a)"), (axs[1], "2.50", "(b)")):
        sc_, be = z[f"{key}_scanned"], z[f"{key}_below"]
        reach = (LAYER + be)
        cls = np.full(be.shape, np.nan)
        cls[sc_] = 3
        cls[sc_ & (reach < c["e_max_mm"])] = 2
        cls[sc_ & (reach < c["e_mm"])] = 1
        cls[sc_ & (be < H)] = 0
        ax.imshow(cls.T, origin="lower", extent=[0, L_CELL, 0, L_CELL], cmap=cmap, vmin=-0.5, vmax=3.5,
                  interpolation="nearest")
        ax.set_title(f"{lab} layer at z = {float(key):.2f} mm", fontsize=9.5, loc="left")
        ax.set_xlabel("x [mm] (wall 0.2 → 0.6 mm)", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
    from matplotlib.patches import Patch
    hs = [Patch(color=RED, label="over powder (no solid below)"),
          Patch(color=ORANGE, label=f"keyhole reaches powder ({1e3 * c['e_mm']:.0f} µm mean)"),
          Patch(color="#f6d7b0", label=f"only its deepest excursions ({1e3 * c['e_max_mm']:.0f} µm)"),
          Patch(color="#d9e6f5", label="closed: tip ends in solid")]
    fig.legend(handles=hs, fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=4)
    ax = axs[2]
    L = s["layers"]
    zz = [r["z"] for r in L]
    for name, col, ls, lab in (("r35_T80", INK, "-", "r0 35 µm, 80 °C"), ("r35_T300", RED, "-", "r0 35 µm, 300 °C"),
                               ("r45_T80", BLUE, "-", "r0 45 µm, 80 °C")):
        ax.plot(zz, [100 * r[f"{name}_mean"] for r in L], ls, color=col, lw=1.5, label=f"{lab}, mean depth")
        ax.plot(zz, [100 * r[f"{name}_max"] for r in L], ":", color=col, lw=1.1)
    ax.plot(zz, [100 * r["conduction"] for r in L], "--", color=MUTED, lw=1.0, label="conduction pool (54 µm)")
    ax.plot(zz, [100 * r["over_powder"] for r in L], color=ORANGE, lw=0.8, label="scanned over powder")
    ax.set_xlabel("build height of the layer [mm]", fontsize=8.5)
    ax.set_ylabel("scanned area where the melt / keyhole reaches powder [%]", fontsize=8.5)
    ax.set_title("(c) through one unit cell (dotted: deepest excursion)", fontsize=9.5, loc="left")
    ax.set_ylim(0, 75)
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", ncol=2)
    ax.grid(True, color=GRID, lw=0.6)
    ax = axs[3]
    b = np.array(s["thickness_bins_mm"])
    mid = 0.5 * (b[1:] + b[:-1])
    w = 0.018
    for k, (name, col) in enumerate((("r35_T80", INK), ("r35_T300", RED), ("r45_T80", BLUE))):
        ax.bar(mid + (k - 1) * w, [100 * v for v in s["total"][f"{name}_mean_by_t"]], width=w, color=col,
               label=name.replace("r", "r0 ").replace("_T", " µm, ") + " °C")
    ax.set_xticks(mid)
    ax.set_xticklabels([f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(b[:-1], b[1:])])
    ax.set_xlabel("wall thickness [mm]", fontsize=8.5)
    ax.set_ylabel("scanned area with an open keyhole [%] (mean depth)", fontsize=8.5)
    ax.set_title("(d) by wall thickness, whole cell", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(True, color=GRID, lw=0.6, axis="y")
    t = s["total"]
    fig.suptitle("Closed keyhole 3: where the keyhole breaks through the solid under a layer of the CoCrFeMnNi gyroid "
                 f"(200 W, 0.75 m/s, 30 µm layers; Gan et al. depth law)\nWhole unit cell: keyhole reaches the powder under "
                 f"{100 * t['r35_T80_mean']:.1f} % of the scanned area at r0 35 µm and 80 °C ({100 * t['r35_T80_max']:.1f} % "
                 f"at its deepest), against {100 * t['conduction']:.1f} % for a conduction pool; "
                 f"{100 * t['over_powder']:.1f} % is scanned directly over powder", fontsize=10, x=0.02, ha="left")
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "run":
        run()
    figure()
