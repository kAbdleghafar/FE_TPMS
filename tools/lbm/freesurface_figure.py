"""Figure for docs/freesurface_lbm.md: AM-Bench case B with the free-surface LBM melt pool."""
import numpy as np

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
NPZ = "tools/lbm/caseB_freesurface_physics_{}_10um.npz"
OUT = "docs/img/freesurface_caseB.png"
TS = 1290.0


def pool_size(Tp, nz_metal, h=10.0, x_sec=150 + 0.75 * 700):
    """Width at the surface and depth on the centreline [um], interpolated at the solidus."""
    i = int(round(x_sec / h)) + 1
    sec = Tp[i, 1:-1, 1:nz_metal + 1]
    col = sec[sec.shape[0] // 2]
    z = (np.arange(nz_metal) + 0.5) * h - nz_metal * h
    k = np.nonzero(col >= TS)[0].min()
    depth = -(z[k] - (col[k] - TS) / (col[k] - col[k - 1]) * h)
    surf = sec[:, -1]
    y = (np.arange(len(surf)) + 0.5) * h
    m = np.nonzero(surf >= TS)[0]
    a, b = m.min(), m.max()
    yl = y[a] - (surf[a] - TS) / (surf[a] - surf[a - 1]) * h
    yr = y[b] + (surf[b] - TS) / (surf[b] - surf[b + 1]) * h
    return float(yr - yl), float(depth)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = {v: np.load(NPZ.format(v)) for v in ("flat", "marangoni")}
    h = 10.0                                                     # um
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[1, 1.5, 1.2]))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14, wspace=0.32)
    nz_metal = 20
    # (a) cross-section: melted region
    ax = axs[0]
    for v, col, lab in (("flat", BLUE, "no Marangoni"), ("marangoni", ORANGE, "Marangoni")):
        Tp = d[v]["T_peak"]
        i = int(round((150 + 0.75 * 700) / h)) + 1
        sec = Tp[i, 1:-1, 1:nz_metal + 4]
        y = (np.arange(sec.shape[0]) + 0.5) * h - 150.0
        z = (np.arange(sec.shape[1]) + 0.5) * h - nz_metal * h
        ax.contour(y, z, sec.T, [TS], colors=[col], linewidths=2)
        w, dp = pool_size(Tp, nz_metal)
        ax.plot([], [], color=col, lw=2, label=f"{lab}: {w:.0f} x {dp:.1f} um")
    ax.add_patch(plt.Rectangle((-123.5 / 2, -36.0), 123.5, 36.0, fill=False, ec=MUTED, ls="--", lw=1))
    ax.plot([], [], color=MUTED, ls="--", label="measured 123.5 x 36 um (NIST)")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlim(-120, 120)
    ax.set_ylim(-80, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("y [um]", fontsize=8.5)
    ax.set_ylabel("z [um] (0 = plate surface)", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    ax.set_title("(a) melted region (peak T >= solidus)", fontsize=9.5, loc="left")
    # (b) peak temperature along the track centre, at the surface
    ax = axs[1]
    for v, col, lab in (("flat", BLUE, "no Marangoni"), ("marangoni", ORANGE, "Marangoni")):
        Tp = d[v]["T_peak"]
        j = Tp.shape[1] // 2
        x = (np.arange(Tp.shape[0] - 2) + 0.5) * h
        ax.plot(x, Tp[1:-1, j, nz_metal], color=col, lw=2, label=lab)
    ax.axhline(TS, color=MUTED, lw=0.8, ls=":")
    ax.text(5, TS + 30, "solidus", fontsize=7.5, color=MUTED)
    ax.set_xlabel("x along the track [um]", fontsize=8.5)
    ax.set_ylabel("peak surface temperature [C]", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_title("(b) peak temperature on the track centreline", fontsize=9.5, loc="left")
    # (c) final surface height (Marangoni)
    ax = axs[2]
    srf = (d["marangoni"]["surface"][1:-1, 1:-1] - (nz_metal + 1) * h * 1e-6) * 1e6
    im = ax.imshow(srf.T, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4, extent=[0, srf.shape[0] * h, -150, 150],
                   aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("surface height change [um]", fontsize=8)
    ax.set_xlabel("x [um]", fontsize=8.5)
    ax.set_ylabel("y [um]", fontsize=8.5)
    ax.set_title("(c) final surface, with Marangoni flow\n(single-cell speckle: interface noise at 10 um)", fontsize=9.5,
                 loc="left")
    for a in axs:
        a.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle("AM-Bench 2018-02 case B (IN625, 179.2 W, 0.8 m/s) with the free-surface lattice Boltzmann melt pool, "
                 "10 um cells on a CPU", fontsize=10.5, x=0.02, ha="left")
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
