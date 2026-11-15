"""Figures for pf3d_in625_eds.py: the 3-D cells and their synthetic EDS maps."""
import json

import numpy as np

import pf3d_in625_eds as M

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ELEMS = ["Nb", "Mo", "Cr", "Fe", "Ni"]
MODE_TITLE = {"sem15": "SEM-EDS, 15 kV", "sem5": "SEM-EDS, 5 kV", "stem": "STEM-EDS, 100 nm foil, 200 kV"}
LINE = {"Ka": "Kα", "La": "Lα"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)


def _img(ax, a, extent, cmap=None, vmin=None, vmax=None, rgb=False):
    if rgb:
        im = ax.imshow(a, origin="lower", extent=extent, interpolation="nearest")
    else:
        im = ax.imshow(a, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbbbbb")
    return im


def _rgb(a, colour, lo=1.0, hi=99.0):
    p0, p1 = np.percentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-12), 0, 1)[..., None] * np.asarray(colour)[None, None, :]


def figure():
    import matplotlib
    matplotlib.use("Agg")
    store = json.load(open(M.STORE))
    d = np.load(M.FIELDS.format(job="3d_W10"))
    fig_eds(d, store)
    fig_3d(d, store)


# ----------------------------------------------------------------------------------------
def fig_eds(d, store):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator
    from scipy.ndimage import uniform_filter

    from tpmsfem.post.eds import COLOURS

    r3, r2 = store["3d_W10"], store.get("2d_W10", {})
    dx = float(d["dx"]) * 1e6
    fig = plt.figure(figsize=(15, 15.6))
    gs = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1, 1.25], hspace=0.32, wspace=0.12, left=0.03, right=0.985,
                          top=0.915, bottom=0.04)
    overlay = ("Nb", "Cr", "Ni")
    ov_col = {"Nb": (1.0, 0.2, 0.15), "Cr": (0.2, 0.45, 1.0), "Ni": (0.2, 0.85, 0.3)}
    # row 0: the simulated composition of the section
    true = {e: d[f"xy_true_{e}"].astype(float) for e in ELEMS}
    L = true["Nb"].shape[1] * dx
    for c, e in enumerate(ELEMS):
        ax = fig.add_subplot(gs[0, c])
        a = true[e]
        cm = LinearSegmentedColormap.from_list(e, [(0, 0, 0), COLOURS[e]])
        im = _img(ax, a, [0, L, 0, L], cmap=cm, vmin=np.percentile(a, 1), vmax=np.percentile(a, 99))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6.5)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()
        ax.set_title(f"{e} [wt%]", fontsize=9, loc="left")
        if c == 0:
            ax.set_ylabel("simulated composition", fontsize=9)
    ax = fig.add_subplot(gs[0, 5])
    _img(ax, sum(_rgb(true[e], ov_col[e]) for e in overlay).clip(0, 1), [0, L, 0, L], rgb=True)
    ax.set_title("Nb (red), Cr (blue), Ni (green)", fontsize=8.5, loc="left")
    ax.plot([0.1 * L, 0.1 * L + 0.5], [0.06 * L] * 2, color="white", lw=2)
    ax.text(0.1 * L + 0.25, 0.09 * L, "0.5 um", color="white", fontsize=7.5, ha="center")
    # rows 1-3: EDS maps (counts, with counting noise)
    info = r3["eds"]
    for rr, mode in enumerate(("sem15", "sem5", "stem"), start=1):
        px = float(d[f"eds_{mode}_xy_pixel"]) * 1e6
        for c, e in enumerate(ELEMS):
            ax = fig.add_subplot(gs[rr, c])
            a = d[f"eds_{mode}_xy_{e}"].astype(float)
            Lm = a.shape[1] * px
            _img(ax, _rgb(a, COLOURS[e]), [0, Lm, 0, Lm], rgb=True)
            ln = LINE.get(info[mode]["lines"][e], info[mode]["lines"][e])
            res = info[mode]["resolution_nm"][e]
            ax.set_title(f"{e} {ln}  (90 % within {res:.0f} nm)", fontsize=8.5, loc="left")
            if c == 0:
                ax.set_ylabel(MODE_TITLE[mode], fontsize=9)
        ax = fig.add_subplot(gs[rr, 5])
        maps = {e: uniform_filter(d[f"eds_{mode}_xy_{e}"].astype(float), 3, mode="wrap") for e in overlay}
        _img(ax, sum(_rgb(maps[e], ov_col[e]) for e in overlay).clip(0, 1), [0, Lm, 0, Lm], rgb=True)
        ax.set_title(f"overlay (3x3 px smoothed); Nb contrast\nkept: {100 * r3['eds_contrast'][mode]:.0f} % (no noise)",
                     fontsize=8.5, loc="left")
    # row 4: Nb and Mo profiles across the section, longitudinal maps
    sub = gs[4, :].subgridspec(1, 5, width_ratios=[1.35, 1.35, 0.75, 0.75, 0.75], wspace=0.28)
    for c, e in enumerate(("Nb", "Mo")):
        ax = fig.add_subplot(sub[0, c])
        t = true[e]
        row = t.shape[0] // 2
        ax.plot(np.arange(t.shape[1]) * dx, t[row], color=INK, lw=1.4, label="simulated")
        for mode, col in (("stem", "#1baf7a"), ("sem5", "#2a78d6"), ("sem15", "#eb6834")):
            a = d[f"eds_{mode}_xy_clean_{e}"].astype(float)
            px = float(d[f"eds_{mode}_xy_pixel"]) * 1e6
            rr = int(round((row + 0.5) * dx / px - 0.5))
            ax.plot(np.arange(a.shape[1]) * px + 0.5 * px, a[rr], color=col, lw=1.3, label=MODE_TITLE[mode])
        ax.set_xlabel("distance along the middle of the section [um]", fontsize=8)
        ax.set_ylabel(f"{e} [wt%] (EDS without counting noise)", fontsize=8)
        ax.set_title(f"({'a' if c == 0 else 'b'}) {e} across the cells", fontsize=9, loc="left")
        ax.set_xlim(0, t.shape[1] * dx)
        _style(ax)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.45 * (hi_ - lo_))                 # room for the legend
        if c == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
    lo = int(d["xz_lo"])
    xz = d["xz_Nb"].astype(float)[lo:int(d["tip"]) + 1]
    phi = d["xz_phi"].astype(float)[lo:int(d["tip"]) + 1]
    H = xz.shape[0] * dx
    ax = fig.add_subplot(sub[0, 2])
    im = _img(ax, np.where(phi > -0.5, xz, np.nan), [0, L, 0, H], cmap="magma", vmin=2.5, vmax=8.0)
    ax.set_title("(c) Nb, x-z section [wt%]", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    for c, mode in enumerate(("sem15", "stem")):
        ax = fig.add_subplot(sub[0, 3 + c])
        a = d[f"eds_{mode}_xz_Nb"].astype(float)
        px = float(d[f"eds_{mode}_xz_pixel"]) * 1e6
        _img(ax, _rgb(a, COLOURS["Nb"]), [0, a.shape[1] * px, 0, a.shape[0] * px], rgb=True)
        ax.set_title(f"({'d' if c == 0 else 'e'}) Nb, {MODE_TITLE[mode].split(',')[0]}"
                     f"{', 15 kV' if mode == 'sem15' else ''}", fontsize=9, loc="left")
    sp = [s for s in r3.get("spacing_3d", []) if s["depth_um"] >= 0.6]
    sp_txt = ", ".join(f"{s['spacing_um']:.2f}" for s in sp)
    seg = r3["segregation_wt"]
    fig.suptitle(
        "IN625 cells in 3-D (multicomponent phase field) at the AM-Bench case-B mid-pool point (G = 6.6e6 K/m, "
        "R = 0.092 m/s) and what EDS would record\n"
        f"Transverse section {r3['section_depth_um']:.1f} um below the tips, {r3['size_um'][0]:.1f} x {r3['size_um'][1]:.1f} um; "
        f"EDS maps with counting noise, contrast stretched per map.\n3-D cell spacing "
        f"sqrt(A/N) = {sp_txt} um at 0.6-3 um below the tips (2-D, same model: {r2.get('spacing_2d_um', float('nan')):.2f} um). "
        f"Nb 5-95 %: {seg['Nb']['p05']:.1f}-{seg['Nb']['p95']:.1f} wt%, Mo {seg['Mo']['p05']:.1f}-{seg['Mo']['p95']:.1f} wt%",
        fontsize=10, x=0.02, ha="left")
    fig.savefig(M.FIG, dpi=130)
    print("wrote", M.FIG)


# ----------------------------------------------------------------------------------------
def fig_3d(d, store):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import LightSource, Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    dx = float(d["dx"]) * 1e6 * 2.0                       # the stored 3-D views are coarsened by 2
    fig = plt.figure(figsize=(16, 6.2))
    ls = LightSource(azdeg=315, altdeg=40)

    def mesh(ax, vol, level, colour_fn, alpha=1.0):
        v, f, n, _ = marching_cubes(vol, level)
        v = v[:, ::-1] * dx                                # (z, y, x) -> (x, y, z)
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        shade = 0.35 + 0.65 * np.abs(nrm @ ls.direction)
        cols = colour_fn(tri.mean(axis=1))
        cols[:, :3] *= shade[:, None]
        pc = Poly3DCollection(tri, facecolors=cols, edgecolors="none", alpha=alpha)
        ax.add_collection3d(pc)
        return v

    # (a) the solidification front (phi = 0) in the final window, coloured by height
    phi = d["window_phi"].astype(float)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    zz = np.flatnonzero((phi > 0).reshape(phi.shape[0], -1).any(axis=1))
    top = int(zz.max()) + 3
    bot = max(top - int(round(0.9 / dx)), 0)
    vol = phi[bot:top]
    norm = Normalize(0, vol.shape[0] * dx)
    mesh(ax, vol, 0.0, lambda c: cm.viridis(norm(c[:, 2])))
    ax.set_xlim(0, phi.shape[2] * dx)
    ax.set_ylim(0, phi.shape[1] * dx)
    ax.set_zlim(0, vol.shape[0] * dx)
    ax.set_box_aspect((phi.shape[2], phi.shape[1], vol.shape[0]))
    ax.view_init(elev=32, azim=-55)
    ax.set_title("(a) the cell front (phi = 0) at the end of the run,\ncoloured by height", fontsize=9.5, loc="left")
    _ax3(ax)
    # (b) cut block: Nb on three faces of the solidified block
    nb = d["block_Nb"].astype(float)
    tip_b = (int(d["tip"]) - int(d["block_lo"])) // 2
    nb = nb[:max(tip_b - int(round(0.3 / dx)), 2)]           # below the tips
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    nz, ny, nx = nb.shape
    cmap, nn = cm.magma, Normalize(2.5, 8.0)
    X, Y = np.meshgrid(np.arange(nx + 1) * dx, np.arange(ny + 1) * dx)
    ax.plot_surface(X, Y, np.full_like(X, nz * dx), facecolors=cmap(nn(nb[-1])), shade=False, rstride=1, cstride=1,
                    linewidth=0, antialiased=False)
    Xz, Zz = np.meshgrid(np.arange(nx + 1) * dx, np.arange(nz + 1) * dx)
    ax.plot_surface(Xz, np.zeros_like(Xz), Zz, facecolors=cmap(nn(nb[:, 0, :])), shade=False, rstride=1, cstride=1,
                    linewidth=0, antialiased=False)
    Yz, Zy = np.meshgrid(np.arange(ny + 1) * dx, np.arange(nz + 1) * dx)
    ax.plot_surface(np.full_like(Yz, nx * dx), Yz, Zy, facecolors=cmap(nn(nb[:, :, -1])), shade=False, rstride=1,
                    cstride=1, linewidth=0, antialiased=False)
    ax.set_box_aspect((nx, ny, nz))
    ax.view_init(elev=28, azim=-60)
    ax.set_title(f"(b) Nb on three faces of the solidified block\n({nx * dx:.1f} x {ny * dx:.1f} um; build direction up)",
                 fontsize=9.5, loc="left")
    _ax3(ax)
    sm = cm.ScalarMappable(norm=nn, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.62, 0.2, 0.008, 0.45])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Nb [wt%]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # (c) the Nb-rich intercellular walls as an iso-surface
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    sub = nb[-int(round(0.6 / dx)):]
    lvl = 0.5 * (float(np.percentile(sub, 10)) + float(np.percentile(sub, 97)))
    mesh(ax, sub, lvl, lambda c: np.tile(np.array([[0.85, 0.25, 0.2, 1.0]]), (len(c), 1)))
    ax.set_xlim(0, nx * dx)
    ax.set_ylim(0, ny * dx)
    ax.set_zlim(0, sub.shape[0] * dx)
    ax.set_box_aspect((nx, ny, sub.shape[0]))
    ax.view_init(elev=50, azim=-60)
    ax.set_title(f"(c) Nb-rich columns along the cell edges: surface Nb = {lvl:.1f} wt%\n(the top 0.6 um of the block in (b))",
                 fontsize=9.5, loc="left")
    _ax3(ax)
    fig.suptitle("3-D cells of IN625 from the multicomponent phase field (AM-Bench case B, mid-pool: G = 6.6e6 K/m, "
                 "R = 0.092 m/s, W0 = 10 nm); axes in um", fontsize=10.5, x=0.02, ha="left")
    fig.subplots_adjust(left=0.0, right=0.98, top=0.86, bottom=0.02, wspace=0.02)
    fig.savefig(M.FIG3D, dpi=130)
    print("wrote", M.FIG3D)


def _ax3(ax):
    from matplotlib.ticker import MaxNLocator
    ax.tick_params(labelsize=6.5, colors=MUTED, pad=0)
    ax.set_facecolor((1, 1, 1, 0))                     # do not hide the neighbouring panel's labels
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_major_locator(MaxNLocator(4))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor("#dddddd")
