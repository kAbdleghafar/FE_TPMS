"""Animation: one LPBF layer (right) and the cells growing at its melt-pool fusion boundary (left).

Right panel. The calibrated AM-Bench case-B laser (179.2 W, 800 mm/s, D4sigma 170 um) scans a
5-track bidirectional hatch (100 um spacing) over a 20 um IN625 powder layer
(``MeltPoolModel``, as in example 05). The panel shows, in top view:
- the surface temperature;
- the melt pool (liquidus contour) and the laser spot (1/e^2 radius);
- the region fused so far;
- the marker that locates the left panel.

Left panel. For each track, the 2-D quantitative phase field (``DirectionalPF``, IN625 as Ni-4Nb,
W0 = 10 nm) grows cells at that track's fusion-boundary solidification conditions, taken from
the same melt-pool run: the median G and R = v |n_x| at the bottom of the track. It runs on the
same physical clock as the laser:
- it starts when the solidification front passes the bottom of the pool behind the laser,
  where the marker sits;
- each simulated microsecond of laser scan is one microsecond of cell growth.
The colour is Nb; the white line is the solid-liquid interface.

Simplifications:
- each track restarts from a planar front, so epitaxy from the previous track is not modelled;
- the phase field is 2-D in the plane that contains the growth direction;
- G and R are held fixed for each track.

    python examples/07_layer_animation.py            # ~1 h on 4 CPU cores; stages are cached
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "literature"))
import in625_material as IM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, raster  # noqa: E402
from tpmsfem.process import phasefield as PF  # noqa: E402

OUT = "results/process/animation"
CAL = json.load(open("benchmarks/literature/results/ambench2018_tracks.json"))["calibration"]
P_W, V_MM_S, R_BEAM = 179.2, 800.0, 0.085
X0, X1, Y0, Y1, HATCH = 0.15, 0.75, -0.2, 0.2, 0.1
FRAME_DT = 20e-6                      # s between frames
T_EXTRA = 3e-4                        # cooling after the last track [s]
PF_W, PF_H, PF_W0 = 3.0e-6, 1.6e-6, 10e-9
SHOW_H = 2.6e-6                       # height of the microstructure column shown [m]


def stage_meltpool():
    f = os.path.join(OUT, "meltpool.npz")
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))
    mat = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
    md = MeltPoolModel(mat, CAL["h"], length=0.9, width=0.7, depth=0.12, powder=0.02, symmetric_y=False, T0=25.0,
                       liquid_k_factor=(1.0, 1.0, CAL["f_z"]), emissivity=0.4, evaporation=Evaporation())
    path = raster(X0, X1, Y0, Y1, hatch=HATCH, speed=V_MM_S, power=P_W, along="x")
    t_scan = sum(np.hypot(v.x1 - v.x0, v.y1 - v.y0) / v.speed for v in path)
    times = np.arange(FRAME_DT, t_scan + T_EXTRA, FRAME_DT)
    t0 = time.perf_counter()
    res = md.run(Laser(P_W, CAL["A"], R_BEAM, depth=CAL.get("delta", 0.0)), path, t_extra=T_EXTRA,
                 snapshot_times=list(times), snapshot_reduce=lambda T: (1.5 * T[:, :, -1] - 0.5 * T[:, :, -2]).astype(np.float32),
                 verbose=True)
    print(f"melt pool: {res.steps} steps, {time.perf_counter() - t0:.0f} s", flush=True)
    # fusion-boundary conditions of each track (cells that last solidified during that track)
    tracks = []
    tcur = 0.0
    for v in path:
        L = float(np.hypot(v.x1 - v.x0, v.y1 - v.y0))
        tracks.append(dict(y=v.y0, x0=v.x0, x1=v.x1, t0=tcur, t1=tcur + L / v.speed))
        tcur += L / v.speed
    X, Y = np.meshgrid(res.x, res.y, indexing="ij")
    for k, tr in enumerate(tracks):
        lo, hi = min(tr["x0"], tr["x1"]) + 0.15, max(tr["x0"], tr["x1"]) - 0.15
        cols = (np.abs(Y - tr["y"]) <= 0.011) & (X >= lo) & (X <= hi)
        G, R, lag = [], [], []
        for i, j in zip(*np.nonzero(cols)):
            ts = res.t_solid[i, j]
            mz = np.flatnonzero(np.isfinite(ts) & (ts >= tr["t0"]) & (ts <= tr["t1"] + 1e-3))
            if not len(mz):
                continue
            kz = mz.min()                                      # deepest cell solidified by this track
            G.append(res.G[i, j, kz] * 1e3)
            R.append(V_MM_S * 1e-3 * abs(res.grad_dir[i, j, kz, 0]))
            s = (res.x[i] - tr["x0"]) / (tr["x1"] - tr["x0"])
            lag.append(ts[kz] - (tr["t0"] + s * (tr["t1"] - tr["t0"])))
        tr.update(G=float(np.median(G)), R=float(np.median(R)), lag=float(np.median(lag)), n=len(G))
        print(f"track {k + 1}: G = {tr['G']:.2e} K/m, R = {tr['R']:.3f} m/s, solidification lag "
              f"{tr['lag'] * 1e6:.0f} us ({tr['n']} columns)", flush=True)
    data = dict(t=np.array([s[0] for s in res.snapshots]), T=np.stack([s[1] for s in res.snapshots]),
                x=res.x, y=res.y, Tl=res.T_liquidus, Ts=res.T_solidus, tracks=np.array(json.dumps(tracks)))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f, **data)
    return data


def stage_phasefield(mp):
    f = os.path.join(OUT, "phasefield.npz")
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))
    tracks = json.loads(str(mp["tracks"]))
    times = mp["t"]
    starts = [tr["t0"] + tr["lag"] for tr in tracks]
    frames_c, frames_phi, info = [], [], []
    pf, k_active, t_pf = None, -1, 0.0
    for n, t in enumerate(times):
        k = max([i for i, s in enumerate(starts) if t >= s], default=-1)
        if k != k_active and k >= 0:
            tr = tracks[k]
            pf = PF.DirectionalPF(PF.IN625_NB, tr["G"], tr["R"], PF_W, PF_H, W0_m=PF_W0, seed=k)
            k_active, t_pf = k, 0.0
        if pf is None:
            frames_c.append(None)
            frames_phi.append(None)
            info.append(dict(track=-1))
            continue
        target = t - starts[k_active]
        if target > t_pf:
            pf.run(target - t_pf, record_every=10 ** 9)
            t_pf = pf.t * pf.tau0
        c, phi = asnum_column(pf)
        frames_c.append(c.astype(np.float16))
        frames_phi.append(phi.astype(np.float16))
        r = PF.PFResult(pf.phi, pf.composition(), np.zeros((0, pf.nx)), pf.dx * pf.W0, t_pf, 0, 0.0,
                        params=dict(W0_m=pf.W0, lT_m=pf.lT * pf.W0))
        lam, ncell = r.spacing()
        info.append(dict(track=k_active, t_pf=t_pf, grown=tracks[k_active]["R"] * t_pf, cells=ncell,
                         spacing=lam if np.isfinite(lam) else None))
        if n % 10 == 0:
            print(f"  frame {n}/{len(times)}: track {k_active + 1}, PF t = {t_pf * 1e6:.0f} us, "
                  f"{ncell:.1f} cells", flush=True)
    shape = next(c.shape for c in frames_c if c is not None)
    blank = np.full(shape, np.nan, np.float16)
    data = dict(c=np.stack([c if c is not None else blank for c in frames_c]),
                phi=np.stack([p if p is not None else blank for p in frames_phi]),
                info=np.array(json.dumps(info)), dx=np.array(PF_W0 * 0.8))
    np.savez_compressed(f, **data)
    return data


def asnum_column(pf):
    """The top SHOW_H of the grown microstructure: rows that left the window plus the window."""
    c = pf.composition()
    phi = pf.phi
    if pf.solid_rows:
        hist = np.concatenate(pf.solid_rows, axis=0)
        c = np.concatenate([hist, c], axis=0)
        phi = np.concatenate([np.ones_like(hist), phi], axis=0)
    rows = np.flatnonzero((phi > 0).any(axis=1))
    top = int(rows.max()) if len(rows) else 0
    n_show = int(round(SHOW_H / (pf.dx * pf.W0)))
    hi = min(top + int(0.35 * n_show), c.shape[0])
    lo = hi - n_show
    if lo < 0:
        pad = -lo
        c = np.concatenate([np.full((pad, c.shape[1]), np.nan), c], axis=0)
        phi = np.concatenate([np.full((pad, c.shape[1]), np.nan), phi], axis=0)
        lo, hi = 0, n_show
    return c[lo:hi] * PF.IN625_NB.c_inf, phi[lo:hi]


def render(mp, pfd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    tracks = json.loads(str(mp["tracks"]))
    info = json.loads(str(pfd["info"]))
    x, y, Tl = mp["x"], mp["y"], float(mp["Tl"])
    dx_um = float(pfd["dx"]) * 1e6
    nshow, ncol = pfd["c"].shape[1:]
    fused = np.zeros(mp["T"].shape[1:], bool)
    fig = plt.figure(figsize=(12.8, 5.6), dpi=100, facecolor="#fcfcfb")
    axL = fig.add_axes([0.05, 0.11, 0.26, 0.74])
    axR = fig.add_axes([0.40, 0.11, 0.52, 0.74])
    caxL = fig.add_axes([0.315, 0.11, 0.01, 0.74])
    caxR = fig.add_axes([0.93, 0.11, 0.01, 0.74])
    fig.text(0.05, 0.95, "One LPBF layer of IN625 and the cells growing at its melt-pool fusion boundary",
             fontsize=13, color="#0b0b0b", weight="bold")
    fig.text(0.05, 0.915, "tpmsfem: melt-pool model (right) and 2-D quantitative phase field (left), on one "
             "physical clock; AM-Bench case-B laser, 179.2 W, 800 mm/s, 5 tracks, 100 um hatch, 20 um powder",
             fontsize=8.5, color="#52514e")
    ext = [0, ncol * dx_um, 0, nshow * dx_um]
    imL = axL.imshow(np.full((nshow, ncol), np.nan), origin="lower", extent=ext, cmap="magma", vmin=2.0, vmax=8.0,
                     interpolation="bilinear")
    cb = fig.colorbar(imL, cax=caxL)
    cb.set_label("Nb [wt%]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    axL.set_xlabel("x [um]", fontsize=9)
    axL.set_ylabel("growth direction [um]", fontsize=9)
    axL.tick_params(labelsize=8, colors="#52514e")
    titleL = axL.set_title("", fontsize=9.5, loc="left")
    noteL = axL.text(0.5, 0.5, "", transform=axL.transAxes, ha="center", va="center", color="#f0efec", fontsize=10)
    imR = axR.imshow(mp["T"][0].T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno", vmin=25,
                     vmax=2600, interpolation="bilinear")
    cb = fig.colorbar(imR, cax=caxR)
    cb.set_label("surface temperature [°C]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    axR.set_xlim(0.05, 0.85)
    axR.set_ylim(-0.29, 0.29)
    axR.set_xlabel("x [mm] (scan direction alternates)", fontsize=9)
    axR.set_ylabel("y [mm]", fontsize=9)
    axR.tick_params(labelsize=8, colors="#52514e")
    titleR = axR.set_title("", fontsize=9.5, loc="left")
    spot = Circle((0, 0), R_BEAM, fill=False, ec="#e4e3df", lw=1.0, ls="--")
    axR.add_patch(spot)
    marker, = axR.plot([], [], "o", ms=7, mfc="#1baf7a", mec="white", mew=1.2)
    dyn = []

    def laser_at(t):
        for tr in tracks:
            if tr["t0"] <= t < tr["t1"]:
                s = (t - tr["t0"]) / (tr["t1"] - tr["t0"])
                return tr["x0"] + s * (tr["x1"] - tr["x0"]), tr["y"], tr
        return None, None, None

    def update(n):
        for a in dyn:
            a.remove()
        dyn.clear()
        t = float(mp["t"][n])
        T = mp["T"][n]
        fused[:] |= T >= Tl
        imR.set_data(T.T)
        dyn.append(axR.contour(x, y, T.T, levels=[Tl], colors=["#ffffff"], linewidths=1.0))
        dyn.append(axR.contour(x, y, fused.T.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=0.8,
                               linestyles=":"))
        xl, yl, tr = laser_at(t)
        spot.set_visible(xl is not None)
        if xl is not None:
            spot.center = (xl, yl)
        it = info[n]
        if it["track"] >= 0:
            trk = tracks[it["track"]]
            d = np.sign(trk["x1"] - trk["x0"])
            xm = (xl if (xl is not None and tr is trk) else trk["x1"]) - d * V_MM_S * trk["lag"]
            xm = np.clip(xm, min(trk["x0"], trk["x1"]), max(trk["x0"], trk["x1"]))
            marker.set_data([xm], [trk["y"]])
            imL.set_data(pfd["c"][n].astype(float))
            ph = pfd["phi"][n].astype(float)
            if np.isfinite(ph).any():
                yy = (np.arange(nshow) + 0.5) * dx_um
                xx = (np.arange(ncol) + 0.5) * dx_um
                dyn.append(axL.contour(xx, yy, np.nan_to_num(ph, nan=1.0), levels=[0.0], colors=["#ffffff"],
                                       linewidths=0.6))
            sp = f", spacing {it['spacing'] * 1e6:.2f} um" if it.get("spacing") else ""
            titleL.set_text(f"cells, track {it['track'] + 1}: G = {trk['G']:.1e} K/m, R = {trk['R']:.3f} m/s\n"
                            f"t = {it['t_pf'] * 1e6:.0f} us after the front passed, grown "
                            f"{it['grown'] * 1e6:.1f} um{sp}")
            noteL.set_text("")
        else:
            marker.set_data([], [])
            titleL.set_text("cells (phase field)")
            noteL.set_text("waiting for the first\nsolidification front")
        k_now = tracks.index(tr) + 1 if tr is not None else None
        titleR.set_text(f"layer, top view: t = {t * 1e3:.2f} ms" + (f", track {k_now} of {len(tracks)}" if k_now
                        else ", cooling") + "\nwhite: melt pool (liquidus)   dashed: laser spot   dotted green: "
                        "fused so far   green dot: phase-field window")
        return []

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "layer_cells.mp4")
    writer = FFMpegWriter(fps=15, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, out, dpi=100):
        for n in range(len(mp["t"])):
            update(n)
            writer.grab_frame(facecolor=fig.get_facecolor())
    print("wrote", out, flush=True)
    update(int(0.55 * len(mp["t"])))
    fig.savefig(os.path.join(OUT, "layer_cells_frame.png"), dpi=130, facecolor=fig.get_facecolor())
    return out


if __name__ == "__main__":
    mp = stage_meltpool()
    pfd = stage_phasefield(mp)
    render(mp, pfd)
