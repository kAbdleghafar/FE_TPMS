"""Figure for porosity_maps.py."""
import numpy as np

import porosity_maps as M

FIG = M.FIG
INK, MUTED, ORANGE, BLUE, AQUA = "#0b0b0b", "#52514e", "#eb6834", "#2a78d6", "#1baf7a"


def panel(ax, name, radius_um, layer, hatch, title, valid=None):
    import matplotlib.pyplot as plt

    m = M.maps(name, radius_um, layer, hatch)
    P = np.array(M.POWERS, float)
    v = np.array(M.SPEEDS, float) / 1000.0
    V, PP = np.meshgrid(v, P)
    por = np.clip(m["por"], 0.0, None)
    im = ax.pcolormesh(v, P, por, cmap=plt.get_cmap("Blues"), vmin=0.0, vmax=15.0, shading="gouraud")
    c1 = ax.contour(V, PP, m["crit"], [1.0], colors=[INK], linewidths=1.6)
    ax.clabel(c1, fmt={1.0: "fusion limit"}, fontsize=7)
    ck = ax.contour(V, PP, m["Ke"], [6.0, 16.0, 30.0], colors=[ORANGE], linewidths=[1.0, 1.6, 1.0],
                    linestyles=["--", "-", ":"])
    ax.clabel(ck, fmt={6.0: "Ke 6", 16.0: "Ke 16", 30.0: "Ke 30"}, fontsize=7)
    if valid:
        for pts, mk, col, lab in ((valid["lof"], "v", BLUE, "lack of fusion (reported)"),
                                  (valid["keyhole"], "s", ORANGE, "keyhole pores (reported)"),
                                  (valid["dense"], "o", AQUA, "~99.9 % dense (reported)")):
            ax.scatter([q[1] / 1000 for q in pts], [q[0] for q in pts], marker=mk, s=46, facecolor=col,
                       edgecolor="white", linewidth=0.8, zorder=5, label=lab)
        ax.legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_xlim(0.33, 2.57)
    ax.set_ylim(92, 308)
    ax.set_xlabel("scan speed [m/s]", fontsize=8.5)
    ax.set_ylabel("laser power [W]", fontsize=8.5)
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(labelsize=8, colors=MUTED)
    return im, m


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = M.LOCATELLI
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    fig.subplots_adjust(left=0.05, right=0.93, top=0.82, bottom=0.12, wspace=0.28)
    im, _ = panel(axs[0], "IN625", 30, L["layer"], L["hatch"],
                  "(a) IN625, spot radius 30 um\n30 um layer, 50 um hatch, plate 80 C", valid=L)
    panel(axs[1], "IN625", 40, L["layer"], L["hatch"],
          "(b) IN625, spot radius 40 um\nsame layer and hatch", valid=L)
    _, mc = panel(axs[2], "Cantor", 35, 0.030, 0.085,
                  "(c) CoCrFeMnNi, spot radius 35 um\n30 um layer, 85 um hatch (prediction, not validated)")
    i, j = M.POWERS.index(200), M.SPEEDS.index(750)
    axs[2].scatter([0.75], [200], marker="*", s=180, facecolor="#f2c94c", edgecolor=INK, linewidth=0.8, zorder=6,
                   label=f"gyroid study (criterion {mc['crit'][i, j]:.2f})")
    axs[2].legend(fontsize=6.5, loc="upper right", frameon=True, framealpha=0.9, edgecolor="none")
    cax = fig.add_axes([0.945, 0.15, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("lack-of-fusion porosity [%] (geometric)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Porosity process maps from the calibrated conduction melt-pool model: lack of fusion below the black "
                 "line, keyhole above the orange lines (Gan et al. 2021: transition > 6, pores likely > 16-30). "
                 "Points: Locatelli et al. 2026", fontsize=9.5, x=0.02, ha="left")
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)
