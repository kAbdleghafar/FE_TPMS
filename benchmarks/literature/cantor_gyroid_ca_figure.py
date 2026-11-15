"""Figure for cantor_gyroid_ca.py."""
import json

import numpy as np

import ca_multilayer_figure as MF
import cantor_gyroid_ca as M

INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = json.load(open(M.OUT))
    names = [n for n in ("thin", "thick") if n in store]
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.36, wspace=0.32, left=0.05, right=0.97, top=0.91,
                          bottom=0.07)
    dx = M.DX * 1e3                                                  # mm
    for col, name in enumerate(names):
        f = np.load(M.FIELDS.format(name=name))
        x0, x1, y0, y1 = store[name]["bounds_mm"]
        pl = f["plan_bd"].astype(float)
        part = f["part_plan"]
        img = np.where((pl.min(axis=-1) < 0.999)[..., None] | part[..., None], pl, 0.93)
        ax = fig.add_subplot(gs[0, 2 * col])
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[x0, x1, y0, y1], interpolation="nearest")
        ax.contour(np.linspace(x0, x1, part.shape[0]), np.linspace(y0, y1, part.shape[1]), part.T.astype(float), [0.5],
                   colors=[INK], linewidths=0.6)
        ax.set_title(f"({'a' if col == 0 else 'c'}) {name} walls: plan view, layer {M.N_LAYERS}\n"
                     f"(IPF along BD; black: nominal wall)", fontsize=9, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        xz = f["xz_bd"].astype(float)
        pxz = f["part_xz"]
        img = np.where((xz.min(axis=-1) < 0.999)[..., None] | pxz[..., None], xz, 0.93)
        ax = fig.add_subplot(gs[0, 2 * col + 1])
        zb = M.Z_START - M.SUB * 1e3
        ax.imshow(np.transpose(img, (1, 0, 2))[::-1], extent=[x0, x1, zb, zb + xz.shape[1] * dx], interpolation="nearest",
                  aspect="auto")
        ax.axhline(M.Z_START, color="white", lw=0.8, ls="--")
        ax.set_title(f"({'b' if col == 0 else 'd'}) {name} walls: vertical section\n(x-z, window centre; dashed: layer 1)",
                     fontsize=9, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("z [mm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
    colors = {"thin": BLUE, "thick": ORANGE}
    ax = fig.add_subplot(gs[1, 0])
    for name in names:
        L = store[name]["layers"]
        ax.plot([l_["layer"] for l_ in L], [100 * l_["frac_100_bd_15"] for l_ in L], "o-", color=colors[name], lw=1.6, ms=3.5,
                label=f"{name} walls")
    if "thin_a1" in store:
        L = store["thin_a1"]["layers"]
        ax.plot([l_["layer"] for l_ in L], [100 * l_["frac_100_bd_15"] for l_ in L], ":", color=BLUE, lw=1.4,
                label="thin walls, cellular growth (alpha = 1)")
    ax.axhline(100 * MF.RANDOM_100, color=MUTED, lw=1, ls="--")
    ax.text(M.N_LAYERS, 100 * MF.RANDOM_100 + 0.4, "random", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("layer", fontsize=8.5)
    ax.set_ylabel("<100> within 15 deg of BD [% of volume]", fontsize=8.5)
    ax.set_title("(e) <100> fibre along the build direction", fontsize=9, loc="left")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    MF._style(ax)
    ax = fig.add_subplot(gs[1, 1])
    for name in names:
        L = store[name]["layers"]
        ax.plot([l_["layer"] for l_ in L], [l_["width_area_weighted_um"] for l_ in L], "o-", color=colors[name], lw=1.6, ms=3.5)
    if "thin_a1" in store:
        L = store["thin_a1"]["layers"]
        ax.plot([l_["layer"] for l_ in L], [l_["width_area_weighted_um"] for l_ in L], ":", color=BLUE, lw=1.4)
    ax.axhline(store[names[0]]["substrate"]["width_area_weighted_um"], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("layer", fontsize=8.5)
    ax.set_ylabel("grain width, area-weighted [um]", fontsize=8.5)
    ax.set_title("(f) grain width normal to the build direction", fontsize=9, loc="left")
    ax.set_ylim(0, None)
    MF._style(ax)
    for col, name in enumerate(names):
        f = np.load(M.FIELDS.format(name=name))
        D, e = MF.pole_density(f["top_rot"].astype(float))
        ax = fig.add_subplot(gs[1, 2 + col])
        im = ax.imshow(D.T, origin="lower", extent=[e[0], e[-1], e[0], e[-1]], cmap="magma", vmin=0, vmax=3.0)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=INK, lw=0.8)
        ax.text(np.sqrt(2) * 1.04, 0, "x", fontsize=8, va="center")
        ax.text(0, np.sqrt(2) * 1.04, "y", fontsize=8, ha="center")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"({'g' if col == 0 else 'h'}) {{100}} pole figure, {name} walls\nlayers 11-20, BD at the centre "
                     f"(equal area)", fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("multiples of random", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("CoCrFeMnNi graded gyroid, 20 layers (0.7-1.3 mm build height) from the 3-D grain CA: 200 W, 0.75 m/s, "
                 "30 um layers, 85 um hatch, contour + hatch, no rotation", fontsize=10.5, x=0.02, ha="left")
    fig.savefig(M.FIG, dpi=150)
    print("wrote", M.FIG)
