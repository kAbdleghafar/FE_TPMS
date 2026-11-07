"""AM-Bench 2018-02 case B: grains in the melt pool, predicted from the *measured* substrate.

Data: NIST example EBSD map of a case-B track (nominal 195 W / 800 mm/s, AMMT true power
179.2 W), transverse cross-section, Oxford .ctf, 1218 x 890 points at 0.14 um
(https://s3.amazonaws.com/nist-midas/1889/AMB2018-02_CaseB_EBSDdata.zip).

Test. The melt pool grows epitaxially from the plate's grains, so the measured substrate
around the fusion line fixes the initial condition. The measured orientations inside the
fusion zone are hidden, and the grain model of tpmsfem.process.solidification regrows them,
pixel by pixel in the order of the simulated solidification times. Each pixel takes the
grain of the already-solid neighbour whose nearest <100> is best aligned with the simulated
growth direction (grad T at solidification), so this is competitive growth. The result is
compared with the measured orientations (point-wise disorientation) and with two baselines:

* "normal epitaxy": each pixel copies the substrate pixel nearest to it on the fusion line
  (growth along the fusion-line normal, no competition);
* "vertical epitaxy": each pixel copies the substrate pixel straight below it.

Geometry. The fusion line is the calibrated model's (width 123.3 / depth 36.0 um, equal to
NIST's case-B mean by calibration), measured from the local surface of the map; its centre is
placed where the long axes of the elongated fusion-zone grains change their lean (the fan
centre), found from the measured grain shapes. Thermal fields are the steady-state transverse
section of the case-B simulation (x = 0.7 mm). The section normal's sign relative to the scan
direction is not recorded, so the in-plane (projected) growth direction is used; the 3-D
direction with either sign is reported as a sensitivity.

    python benchmarks/literature/nist_ambench2018_tracks_ebsd.py      # ~5 min
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import in625_material as IM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process import ebsd as E  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

CTF = "results/literature/ambench2018_tracks/micro/ebsd/High_resolution_Map_Data_3.ctf"
URL = "https://s3.amazonaws.com/nist-midas/1889/AMB2018-02_CaseB_EBSDdata.zip"
OUT = "benchmarks/literature/results/ambench2018_tracks_ebsd.json"
BIN = 2                                   # 2 x 2 binning: 0.28 um pixels
MAT = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())


def fetch():
    if os.path.exists(CTF):
        return
    import urllib.request
    import zipfile
    d = os.path.dirname(CTF)
    os.makedirs(d, exist_ok=True)
    zp = os.path.join(os.path.dirname(d), "AMB2018-02_CaseB_EBSDdata.zip")
    req = urllib.request.Request(URL, headers={"User-Agent": "tpmsfem-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(zp, "wb") as fh:
        fh.write(r.read())
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith(".ctf"):
                open(CTF, "wb").write(z.read(n))


def simulate_section():
    cal = json.load(open("benchmarks/literature/results/ambench2018_tracks.json"))["calibration"]
    md = MeltPoolModel(MAT, cal["h"], length=1.2, width=0.2, depth=0.12, T0=25.0,
                       liquid_k_factor=(1.0, 1.0, cal["f_z"]), emissivity=0.4, evaporation=Evaporation())
    res = md.run(Laser(179.2, cal["A"], 0.085, depth=cal.get("delta", 0.0)), [ScanVector(0.15, 0.0, 0.95, 0.0, 800.0)],
                 t_extra=3e-4)
    i = int(np.argmin(np.abs(res.x - 0.7)))
    return res, i


def binned(m):
    ny = 400 // BIN * BIN
    g = m.orientations()[:ny:BIN, ::BIN]
    valid = (m.phase[:ny] > 0).reshape(ny // BIN, BIN, -1, BIN).all(axis=(1, 3))[:, : g.shape[1]]
    return g, valid, m.step[0] * BIN


def fan_centre(lab, step, top):
    """x [um] where elongated grains in the upper region change their lean (least squares of
    the long-axis angle against x)."""
    sh = E.grain_shapes(lab, (step, step))
    depth = sh["cy"] - np.interp(sh["cx"], np.arange(len(top)) * step, top * step)
    sel = (sh["length"] > 3 * sh["width"]) & (sh["length"] > 12) & (depth < 30) & (sh["width"] > 1.5)
    x, a = sh["cx"][sel], sh["angle"][sel]
    w = sh["area"][sel]
    best = None
    for xc in np.arange(x.min() + 10, x.max() - 10, 0.5):           # lean changes sign at x_c
        s = np.sign(x - xc)
        # image frame (y down): a grain left of the centre grows up-right, i.e. a negative angle,
        # one right of it a positive angle
        score = np.sum(w * (s * a > 0))
        if best is None or score > best[0]:
            best = (score, xc)
    return best[1], int(sel.sum())


def main():
    fetch()
    m = E.read_ctf(CTF)
    g, valid, step = binned(m)
    ny, nx = valid.shape
    lab = E.segment_grains(g, valid, threshold=10.0, min_size=10)
    top = np.array([np.argmax(valid[:, j]) if valid[:, j].any() else 0 for j in range(nx)], float)
    from scipy.ndimage import median_filter
    top = median_filter(top, size=21, mode="nearest")
    xc, n_elong = fan_centre(lab, step, top)
    print(f"EBSD: {nx} x {ny} px at {step:.2f} um, {lab.max() + 1} grains (10 deg), fan centre x = {xc:.1f} um "
          f"({n_elong} elongated grains)", flush=True)

    res, i = simulate_section()
    yy = np.r_[-res.y[::-1], res.y]                                  # mm, full width
    zz = res.z
    Tp = np.vstack([res.T_peak[i][::-1], res.T_peak[i]])            # (2ny, nz)
    ts = np.vstack([res.t_solid[i][::-1], res.t_solid[i]])
    gd = np.concatenate([res.grad_dir[i][::-1] * np.array([1, -1, 1]), res.grad_dir[i]], axis=0)
    # pixel coordinates -> model (y, z) [mm]
    X = (np.arange(nx) + 0.5) * step
    Ypx = (np.arange(ny) + 0.5) * step
    depth = Ypx[:, None] - top[None, :] * step                       # um below the local surface
    ym = (X[None, :] - xc) * 1e-3 * np.ones((ny, 1))
    zm = -depth * 1e-3
    from scipy.interpolate import RegularGridInterpolator as RGI
    pts = np.column_stack([np.clip(ym.ravel(), yy[0], yy[-1]), np.clip(zm.ravel(), zz[0], zz[-1])])
    Tpi = RGI((yy, zz), Tp, bounds_error=False, fill_value=25.0)(pts).reshape(ny, nx)
    # linear interpolation of the solidification time (the unmelted substrate counts as solid from
    # t = 0) and of the growth direction: nearest-neighbour values would give every 18 x 18-pixel
    # block one time and let the regrowth sweep each block in raster order
    melted = np.isfinite(ts)
    tsi = RGI((yy, zz), np.where(melted, ts, 0.0), bounds_error=False, fill_value=0.0)(pts).reshape(ny, nx)
    gdi = np.stack([RGI((yy, zz), np.nan_to_num(gd[..., k]), bounds_error=False, fill_value=0.0)(pts)
                    for k in range(3)], -1).reshape(ny, nx, 3)
    fz_geom = (Tpi >= res.T_solidus) & (depth >= 0)                   # smooth fusion line of the model
    fz = fz_geom & valid
    sub = valid & ~fz & (depth >= 0)
    print(f"fusion zone: {fz.sum()} px ({fz.sum() * step * step:.0f} um^2); substrate seeds {sub.sum()} px", flush=True)

    # model frame (x scan, y transverse, z up) -> EBSD sample frame (X = y, Y = -z, Z = +-x)
    def to_ebsd(d, sz):
        return np.stack([d[..., 1], -d[..., 2], sz * d[..., 0]], -1)

    preds = {}
    for name, d_e in (("in-plane", to_ebsd(gdi * np.array([0.0, 1.0, 1.0]), 1.0)),
                      ("3D, Z = +x", to_ebsd(gdi, 1.0)), ("3D, Z = -x", to_ebsd(gdi, -1.0))):
        n = np.linalg.norm(d_e, axis=-1, keepdims=True)
        preds[name] = regrow(g, sub, fz, tsi, d_e / np.maximum(n, 1e-12))
        if name == "in-plane":
            dn = d_e / np.maximum(n, 1e-12)
            preds["dendrite capture"] = regrow(g, sub, fz, tsi, dn, rule="dendrite")
            preds["upstream"] = regrow(g, sub, fz, tsi, dn, rule="upstream")
            preds["gradient trace"] = trace_gradient(g, sub, fz, dn, step)
    preds["normal epitaxy"] = nearest_seed(g, sub, fz, step, mode="nearest")
    preds["vertical epitaxy"] = nearest_seed(g, sub, fz, step, mode="vertical")

    results = {}
    gm = g[fz]
    rng = np.random.default_rng(0)
    rand = S.random_rotations(fz.sum(), rng)
    mis_rand = E.misorientation(gm, rand)
    results["random orientations"] = dict(frac_lt10=float((mis_rand < 10).mean()), median=float(np.median(mis_rand)))
    for name, gp in preds.items():
        mis = E.misorientation(gm, gp[fz])
        results[name] = dict(frac_lt10=float((mis < 10).mean()), frac_lt5=float((mis < 5).mean()),
                             median=float(np.median(mis)))
        print(f"{name:18s}: {100 * results[name]['frac_lt10']:.1f} % of fusion-zone pixels within 10 deg of the "
              f"measured orientation (median {results[name]['median']:.1f} deg)", flush=True)
    print(f"random            : {100 * results['random orientations']['frac_lt10']:.1f} %", flush=True)

    # grain statistics in the fusion zone, measured vs predicted (in-plane model)
    stats = {}
    for name, gg in (("measured", g), ("tpmsfem max-alignment", preds["in-plane"]),
                     ("tpmsfem dendrite capture", preds["dendrite capture"]), ("gradient trace", preds["gradient trace"]),
                     ("tpmsfem upstream", preds["upstream"]),
                     ("normal epitaxy", preds["normal epitaxy"])):
        gl = gg.copy()
        lab2 = E.segment_grains(gl, fz, threshold=10.0, min_size=10)
        sh = E.grain_shapes(lab2, (step, step))
        a = sh["area"]
        build = np.array([0.0, -1.0, 0.0])
        al = S.alignment(gg[fz], build)
        big = a >= 1.0
        stats[name] = dict(grains=int(lab2.max() + 1), mean_area_um2=float(a.mean()),
                           area_weighted_d_eq_um=float((sh["d_eq"] * a).sum() / a.sum()),
                           area_weighted_aspect=float(((sh["length"][big] / np.maximum(sh["width"][big], 0.28))
                                                       * a[big]).sum() / a[big].sum()),
                           frac_100_build_15deg=float((al >= np.cos(np.radians(15))).mean()))
    print(json.dumps(stats, indent=1), flush=True)
    sub_lab = E.segment_grains(g, sub, threshold=10.0, min_size=10)
    sub_sh = E.grain_shapes(sub_lab, (step, step))
    sub_d = float(np.median(sub_sh["d_eq"][sub_sh["area"] > 4]))
    out = dict(source=URL, step_um=step, fan_centre_um=xc, fusion_zone_px=int(fz.sum()),
               substrate_median_grain_d_eq_um=sub_d, agreement=results, grain_stats=stats)
    json.dump(out, open(OUT, "w"), indent=1)
    figure(g, valid, fz_geom, preds, step, xc, top, out)


def regrow(g, sub, fz, tsol, d, rule="max-alignment"):
    """Competitive epitaxial growth on the pixel grid (8 neighbours).

    ``max-alignment`` (the rule of solidification.grow_grains): the pixel joins the solid
    neighbour whose nearest <100> is best aligned with the growth direction, weighted by how
    far upstream the neighbour lies. ``dendrite``: a neighbour grain captures the pixel along
    its own dendrite arm -- the <100> axis closest to the growth direction -- so the score is
    (step from neighbour to pixel . arm) * (arm . growth direction), as in decentred-
    octahedron CA models."""
    ny, nx = sub.shape
    out = np.where(sub[..., None, None], g, np.nan)
    solid = sub.copy()
    order = np.argsort(tsol[fz], kind="stable")
    pix = np.argwhere(fz)[order]
    offs = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    # neighbour offsets in EBSD frame (X right, Y down) as unit vectors
    on = np.column_stack([offs[:, 1], offs[:, 0], np.zeros(8)]).astype(float)
    on /= np.linalg.norm(on, axis=1, keepdims=True)
    for (r, c) in pix:
        nb = np.array([r, c]) + offs
        ok = (nb[:, 0] >= 0) & (nb[:, 0] < ny) & (nb[:, 1] >= 0) & (nb[:, 1] < nx)
        nb, o = nb[ok], on[ok]
        s = solid[nb[:, 0], nb[:, 1]]
        if not s.any():
            continue
        nb, o = nb[s], o[s]
        dd = d[r, c]
        rot = out[nb[:, 0], nb[:, 1]]
        if rule == "upstream":
            score = np.clip(-o @ dd, 0.0, None) + 1e-6           # pure epitaxy along the gradient
        elif rule == "dendrite":
            proj = np.einsum("nij,j->ni", rot, dd)                # cos of each <100> axis with d
            ax = np.argmax(np.abs(proj), axis=1)
            sgn = np.sign(proj[np.arange(len(ax)), ax])
            arm = rot[np.arange(len(ax)), ax] * sgn[:, None]       # dendrite arm closest to d
            step_dir = -o                                          # from neighbour to pixel
            score = np.clip(np.einsum("ni,ni->n", step_dir, arm), 0.0, None) * np.abs(proj[np.arange(len(ax)), ax])
        else:
            up = np.clip(-o @ dd, 0.0, None)                    # neighbour lies upstream of the growth
            score = S.alignment(rot, dd) * (0.25 + up)
        k = int(np.argmax(score))
        out[r, c] = rot[k]
        solid[r, c] = True
    # pixels never reached (isolated) keep nan -> fill with nearest solid neighbour orientation
    miss = fz & ~solid
    if miss.any():
        out[miss] = nearest_seed(np.nan_to_num(out), solid, miss, 1.0, "nearest")[miss]
    return out


def trace_gradient(g, sub, fz, d, step, max_steps=400):
    """Copy the substrate orientation reached by walking from each pixel against the growth
    direction (down the thermal gradient) until the substrate: epitaxy along the simulated
    gradient, no competition."""
    ny, nx = sub.shape
    out = g.copy()
    pf = np.argwhere(fz).astype(float)
    pos = pf.copy()
    done = np.zeros(len(pf), bool)
    src = np.zeros((len(pf), 2), int)
    for _ in range(max_steps):
        r = np.clip(np.rint(pos[:, 0]).astype(int), 0, ny - 1)
        c = np.clip(np.rint(pos[:, 1]).astype(int), 0, nx - 1)
        hit = sub[r, c] & ~done
        src[hit] = np.column_stack([r[hit], c[hit]])
        done |= hit
        if done.all():
            break
        dd = d[r, c]
        # one pixel against the in-plane growth direction (EBSD: X right = col, Y down = row)
        dv = np.column_stack([dd[:, 1], dd[:, 0]])
        nrm = np.linalg.norm(dv, axis=1, keepdims=True)
        dv = np.where(nrm > 1e-9, dv / np.maximum(nrm, 1e-12), np.array([[1.0, 0.0]]))   # fallback: straight down
        pos[~done] -= 0.5 * dv[~done]
    rest = ~done
    if rest.any():
        fb = nearest_seed(g, sub, fz, step, "nearest")
        out[pf[rest, 0].astype(int), pf[rest, 1].astype(int)] = fb[pf[rest, 0].astype(int), pf[rest, 1].astype(int)]
    ok = done
    out[pf[ok, 0].astype(int), pf[ok, 1].astype(int)] = g[src[ok, 0], src[ok, 1]]
    return out


def nearest_seed(g, sub, fz, step, mode="nearest"):
    from scipy.spatial import cKDTree
    out = g.copy()
    ps = np.argwhere(sub)
    pf = np.argwhere(fz)
    if mode == "vertical":
        for (r, c) in pf:
            col = np.flatnonzero(sub[r:, c])
            out[r, c] = g[r + col[0], c] if len(col) else g[r, c]
        return out
    # nearest substrate pixel on the fusion line (substrate pixels adjacent to the fusion zone)
    from scipy.ndimage import binary_dilation
    edge = sub & binary_dilation(fz)
    pe = np.argwhere(edge)
    _, j = cKDTree(pe).query(pf)
    out[pf[:, 0], pf[:, 1]] = g[pe[j, 0], pe[j, 1]]
    return out


def figure(g, valid, fz, preds, step, xc, top, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK2 = "#fcfcfb", "#52514e"
    plt.rcParams.update({"font.size": 8.5, "figure.facecolor": SURF, "axes.facecolor": SURF, "xtick.color": INK2,
                         "ytick.color": INK2})
    build = (0.0, -1.0, 0.0)
    ny, nx = valid.shape
    ext = [0, nx * step, ny * step, 0]

    def ipf(gg, mask):
        rgb = S.ipf_color(np.nan_to_num(gg), build)
        rgb[~mask] = 1.0
        return rgb
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.6), constrained_layout=True)
    a = out["agreement"]
    panels = ((g, valid, "Measured (NIST EBSD, case B), IPF colour of the build direction"),
              (preds["gradient trace"], valid,
               f"tpmsfem, epitaxy along the simulated thermal gradient: {100 * a['gradient trace']['frac_lt10']:.0f} % "
               f"of melt-pool pixels within 10° of measured"),
              (preds["in-plane"], valid,
               f"tpmsfem, competitive growth: {100 * a['in-plane']['frac_lt10']:.0f} %"),
              (preds["normal epitaxy"], valid,
               f"Baseline, copy the nearest fusion-line grain: {100 * a['normal epitaxy']['frac_lt10']:.0f} %   "
               f"(random orientations: {100 * a['random orientations']['frac_lt10']:.1f} %)"))
    for ax, (gg, mask, title) in zip(axes, panels):
        ax.imshow(ipf(gg, mask), extent=ext, interpolation="nearest")
        ax.contour((np.arange(nx) + 0.5) * step, (np.arange(ny) + 0.5) * step, fz.astype(float), levels=[0.5],
                   colors=["#0b0b0b"], linewidths=0.8)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("depth [µm]")
        ax.set_xlim(xc - 85, xc + 85)
    fig.suptitle("AM-Bench 2018-02 case B: melt-pool grains from the measured substrate (black: model fusion line)",
                 fontsize=10, x=0.02, ha="left")
    fig.savefig("docs/img/ebsd_caseB_regrowth.png", dpi=200)


if __name__ == "__main__":
    main()
