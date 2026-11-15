"""Figure for ca_multilayer.py."""
import json

import numpy as np

import ca_multilayer as M

INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
RANDOM_100 = 6 * (1 - np.cos(np.radians(15.0))) / 2          # random cubic: <100> within 15 deg of an axis


def pole_density(rot, n=41):
    """{100} pole density on an equal-area (Lambert) projection of the upper hemisphere around
    the build direction, in multiples of random (MRD)."""
    p = np.concatenate([rot[:, 0], rot[:, 1], rot[:, 2]])
    p = np.where(p[:, 2:3] < 0, -p, p)
    r = np.sqrt(np.clip(2.0 * (1.0 - p[:, 2]), 0, None))
    phi = np.arctan2(p[:, 1], p[:, 0])
    X, Y = r * np.cos(phi), r * np.sin(phi)
    e = np.linspace(-np.sqrt(2), np.sqrt(2), n + 1)
    H, _, _ = np.histogram2d(X, Y, [e, e])
    c = 0.5 * (e[1:] + e[:-1])
    inside = (c[:, None] ** 2 + c[None, :] ** 2) <= 2.0
    from scipy.ndimage import gaussian_filter
    H = gaussian_filter(H, 1.0)
    Hm = H[inside].mean()
    D = np.where(inside, H / Hm, np.nan)
    return D, e


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = json.load(open(M.OUT))
    rots = [k for k in ("rot90_side480_dx2", "rot67_side480_dx2") if k in store]
    checks = [k for k in store if k not in rots]
    fig = plt.figure(figsize=(14, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.38, wspace=0.3, left=0.05, right=0.97, top=0.91,
                          bottom=0.07)
    dx = M.DX * 1e6
    for col, key in enumerate(rots):
        f = np.load(M.FIELDS.format(key=key))
        deg = store[key]["rot_deg"]
        xz = f["xz_bd"].astype(float)                                    # (nx, nz)
        ax = fig.add_subplot(gs[0, 2 * col])
        ax.imshow(np.transpose(xz, (1, 0, 2))[::-1], extent=[0, xz.shape[0] * dx, 0, xz.shape[1] * dx],
                  interpolation="nearest")
        ax.axhline(M.SUB * 1e6, color="white", lw=0.8, ls="--")
        ax.set_title(f"({'a' if col == 0 else 'c'}) {deg:.0f} deg rotation: vertical section (x-z)\nIPF along the build "
                     f"direction; dashed: top of the substrate", fontsize=9, loc="left")
        ax.set_xlabel("x [um]", fontsize=8)
        ax.set_ylabel("z [um]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        pl = f["plan_bd"].astype(float)
        ax = fig.add_subplot(gs[0, 2 * col + 1])
        ax.imshow(np.transpose(pl, (1, 0, 2)), extent=[0, pl.shape[0] * dx, 0, pl.shape[1] * dx], origin="lower",
                  interpolation="nearest")
        ax.set_title(f"({'b' if col == 0 else 'd'}) {deg:.0f} deg: plan view in the last layer\nIPF along the build direction",
                     fontsize=9, loc="left")
        ax.set_xlabel("x [um]", fontsize=8)
        ax.set_ylabel("y [um]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)

    colors = {"rot90_side480_dx2": BLUE, "rot67_side480_dx2": ORANGE}

    def label(k):
        r = store[k]
        return (f"{r['rot_deg']:.0f} deg, {r['side_um']:.0f} um block, {r['dx_um']:g} um cells"
                + (f", alpha {r['heat_flow_bias']:g}" if r.get("heat_flow_bias") else ""))
    ax = fig.add_subplot(gs[1, 0])
    for key in rots:
        L = store[key]["layers"]
        ax.plot([l_["layer"] for l_ in L], [100 * l_["frac_100_bd_15"] for l_ in L], "o-", color=colors[key], lw=1.6, ms=4,
                label=label(key))
    for key in checks:
        L = store[key]["layers"]
        c = BLUE if store[key]["rot_deg"] == 90 else ORANGE
        ax.plot([l_["layer"] for l_ in L], [100 * l_["frac_100_bd_15"] for l_ in L], ":", color=c, lw=1.0,
                label=label(key) + " (check)")
    ax.axhline(100 * RANDOM_100, color=MUTED, lw=1, ls="--")
    ax.text(10.2, 100 * RANDOM_100 + 0.4, "random", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("layer", fontsize=8.5)
    ax.set_ylabel("<100> within 15 deg of BD [% of volume]", fontsize=8.5)
    ax.set_title("(e) <100> fibre along the build direction", fontsize=9, loc="left")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    _style(ax)
    ax = fig.add_subplot(gs[1, 1])
    for key in rots:
        L = store[key]["layers"]
        ax.plot([l_["layer"] for l_ in L], [l_["width_area_weighted_um"] for l_ in L], "o-", color=colors[key], lw=1.6, ms=4)
    for key in checks:
        L = store[key]["layers"]
        c = BLUE if store[key]["rot_deg"] == 90 else ORANGE
        ax.plot([l_["layer"] for l_ in L], [l_["width_area_weighted_um"] for l_ in L], ":", color=c, lw=1.0)
    sub = store[rots[0]]["substrate"]["width_area_weighted_um"]
    ax.axhline(sub, color=MUTED, lw=1, ls="--")
    ax.text(10.2, sub + 1, "substrate", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("layer", fontsize=8.5)
    ax.set_ylabel("grain width, area-weighted [um]", fontsize=8.5)
    ax.set_title("(f) grain width normal to the build direction", fontsize=9, loc="left")
    ax.set_ylim(0, None)
    _style(ax)
    for col, key in enumerate(rots):
        f = np.load(M.FIELDS.format(key=key))
        D, e = pole_density(f["top_rot"].astype(float))
        ax = fig.add_subplot(gs[1, 2 + col])
        im = ax.imshow(D.T, origin="lower", extent=[e[0], e[-1], e[0], e[-1]], cmap="magma", vmin=0, vmax=3.0)
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color=INK, lw=0.8)
        ax.text(np.sqrt(2) * 1.04, 0, "x", fontsize=8, va="center")
        ax.text(0, np.sqrt(2) * 1.04, "y", fontsize=8, ha="center")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"({'g' if col == 0 else 'h'}) {{100}} pole figure, {store[key]['rot_deg']:.0f} deg\n"
                     f"layers 6-10, BD at the centre (equal area)", fontsize=9, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("multiples of random", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Grain structure of a 10-layer IN625 block from the 3-D grain CA (AM-Bench 2018 settings: 195 W, 800 mm/s, "
                 "100 um hatch, 20 um layers; 480 um block, 2 um cells; dotted: resolution and size checks)", fontsize=10.5, x=0.02, ha="left")
    fig.savefig(M.FIG, dpi=150)
    print("wrote", M.FIG)


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
