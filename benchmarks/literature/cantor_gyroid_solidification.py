"""Solidification of an LPBF gyroid-sheet lattice in the CoCrFeMnNi (Cantor) alloy.

Question. How do the solidification conditions, cells, Mn/Ni segregation and grain texture
change between thin and thick lattice walls, and on down-facing surfaces built over powder?

Approach (tpmsfem.process):
1. Validation. Piglione et al. (2018) built a single LPBF layer of CoCrFeMnNi with a Renishaw
   AM250 (200 W, 80 us exposure, 60 um point distance) and measured cell spacings of
   0.52-0.64 um. Here:
   - a single track at those settings gives the fusion-boundary G and R;
   - the 2-D phase field runs there in the CrFeCo-(MnNi) pseudo-binary, for freezing ranges
     of 40, 60 and 90 K;
   - the resulting spacing is compared with theirs.
2. Gyroid. A gyroid sheet with a 4 mm cell, its wall thickness graded linearly from 0.2 mm
   (x = 0) to 0.6 mm (x = 4 mm), is built at the same settings: 30 um layers, 85 um hatch,
   one contour, no rotation. Two 1.2 mm windows of the layer at 1.3 mm build height are
   simulated, one thin-walled and one thick-walled:
   - the substrate is the lattice built so far, with powder elsewhere
     (MeltPoolModel.set_part_geometry);
   - the scan path is the clipped contour plus hatch (process.scanpath);
   - fusion-boundary G and R are classified as thin wall, thick wall, or down-facing
     (melted into the powder of the layers below);
   - the phase field runs at each class's median;
   - competitive grain growth from a random 20 um substrate gives the <100> texture.

Assumptions:
- Cantor thermal data are the Li et al. (2021) simulation inputs (cantor_material.py);
- the absorptivity is 0.35 and the absorption depth 15 um (not calibrated for this alloy);
- the pulsed AM250 exposure is treated as a continuous scan at 0.75 m/s;
- the 316L plate of the validation build is replaced by Cantor alloy;
- the 30 um layer of the validation build is assumed; the paper does not give it;
- evaporation is that of Mn at its 20 at% (Raoult) on top of the melt;
- the build-plate temperature is 80 C;
- part-scale heat accumulation over many layers is not included.

    python benchmarks/literature/cantor_gyroid_solidification.py            # ~3 h on 4 cores
    python benchmarks/literature/cantor_gyroid_solidification.py --part validation
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_material as CM  # noqa: E402
from common import ResultStore  # noqa: E402

from tpmsfem.geometry.fields import Field  # noqa: E402
from tpmsfem.geometry.tpms import sheet_with_thickness  # noqa: E402
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402
from tpmsfem.process.scanpath import slice_scan_path  # noqa: E402

STORE = "benchmarks/literature/results/cantor_gyroid.json"
FIELDS = "benchmarks/literature/results/cantor_gyroid_fields.npz"
FIG = "docs/img/cantor_gyroid.png"
MAT = CM.material()
P_W, V_MM_S, R_SPOT, ABS, DELTA = 200.0, 750.0, 0.035, 0.35, 0.015
LAYER, HATCH, CONTOUR_OFFSET, T_PLATE = 0.03, 0.085, 0.035, 80.0
H = 0.005
EVAP = Evaporation(T_boil=2061.0, L_v=4.02e6, M=0.05494, p0=0.2 * 101325.0)     # Mn, x_Mn = 0.2
L_CELL, Z_LAYER = 4.0, 1.3
WINDOWS = {"thin": (0.2, 1.4, 1.0, 2.2), "thick": (2.6, 3.8, 0.55, 1.75)}          # x0, x1, y0, y1 [mm]
DT0 = (40.0, 60.0, 90.0)
PF_W0, PF_W, PF_H, PF_GROW = 10e-9, 6.0e-6, 2.0e-6, 10e-6     # 6 um wide: spacing steps of ~10 %


def gyroid():
    return sheet_with_thickness("gyroid", L_CELL, lambda x, y, z: 0.2 + 0.4 * np.clip(x / L_CELL, 0.0, 1.0))


def laser():
    return Laser(P_W, ABS, R_SPOT, depth=DELTA)


def fusion_boundary(res, melted_ok):
    """Deepest melted cell of each column (where the cells start): G [K/m], R [m/s]
    (cooling rate / G), cooling rate, and indices."""
    mel = (res.n_melt > 0) & np.isfinite(res.t_solid) & (res.G > 0) & melted_ok
    has = mel.any(axis=2)
    kz = np.argmax(mel, axis=2)
    ii, jj = np.nonzero(has)
    kk = kz[ii, jj]
    return dict(G=res.G[ii, jj, kk] * 1e3, R=res.R[ii, jj, kk] * 1e-3, cr=res.cooling_rate[ii, jj, kk], i=ii, j=jj, k=kk)


def summary(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return dict(n=int(a.size), p10=float(np.percentile(a, 10)), p50=float(np.median(a)), p90=float(np.percentile(a, 90)))


# ----------------------------------------------------------------------------------- PF
def run_pf(G, R, dT0, seed=0):
    alloy = CM.pseudo_binary(dT0)
    m = PF.DirectionalPF(alloy, G, R, PF_W, PF_H, W0_m=PF_W0, seed=seed)
    hist = []
    n_chunks = 25
    for _ in range(n_chunks):
        r = m.run(PF_GROW / R / n_chunks, record_every=500)
        tips = r.tips()
        hist.append(dict(grown_um=R * r.t_s * 1e6, n=r.boundaries(),
                         cs=float(np.nanmedian([t["cs"] for t in tips])) if tips else float("nan"),
                         cl=float(np.nanmedian([t["cl"] for t in tips])) if tips else float("nan")))
    late = hist[int(0.6 * n_chunks):]
    Lx = m.nx * m.dx * m.W0
    n = float(np.mean([max(h["n"], 1) for h in late]))
    liq = r.c[r.phi < 0.0]
    return dict(G=G, R=R, dT0=dT0, spacing_um=Lx / n * 1e6,
                spacing_um_range=[Lx / max(h["n"], 1) * 1e6 for h in late],
                cs_rel=float(np.nanmean([h["cs"] for h in late])), cl_rel=float(np.nanmean([h["cl"] for h in late])),
                intercell_max_rel=float(liq.max()) if liq.size else float("nan"),
                D_over_V_in_W0=alloy.D / R / PF_W0, hunt_um=float(S.primary_spacing(G, R, sc(dT0), "hunt")),
                kurz_fisher_um=float(S.primary_spacing(G, R, sc(dT0), "kurz-fisher")), history=hist), r


def sc(dT0):
    return S.SolidificationConstants(freezing_range=dT0, D_liquid=3e-9, gibbs_thomson=3.65e-7, partition=0.74)


def save_field(tag, r):
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
    lo = max(top - int(1.5e-6 / r.dx_m), 0)
    fields[tag + "_c"] = r.c[lo:top + 10].astype(np.float16)
    fields[tag + "_dx"] = np.array(r.dx_m)
    np.savez_compressed(FIELDS, **fields)


# --------------------------------------------------------------------------- validation
def part_validation(store):
    if store.get("validation_meltpool") is None:
        md = MeltPoolModel(MAT, H, length=1.2, width=0.2, depth=0.15, powder=LAYER, T0=T_PLATE, emissivity=0.4,
                           evaporation=EVAP)
        t0 = time.perf_counter()
        res = md.run(laser(), [ScanVector(0.15, 0.0, 1.05, 0.0, V_MM_S)], t_extra=2e-4)
        mid = (res.x[:, None, None] > 0.4) & (res.x[:, None, None] < 0.8) & np.ones(res.T.shape, bool)
        fb = fusion_boundary(res, mid)
        jc = fb["j"] <= 2                                         # centreline columns: pool bottom
        from tpmsfem.process.meltpool import fusion_zone
        w, d = fusion_zone(res, 0.6)
        out = dict(G=summary(fb["G"]), R=summary(fb["R"]), cr=summary(fb["cr"]),
                   G_bottom=float(np.median(fb["G"][jc])), R_bottom=float(np.median(fb["R"][jc])),
                   width_um=1e3 * w, depth_um=1e3 * d, T_peak_max=float(res.T_peak.max()),
                   seconds=time.perf_counter() - t0)
        store.put("validation_meltpool", out)
    v = store.get("validation_meltpool")
    print(f"validation melt pool: width {v['width_um']:.0f} um, depth {v['depth_um']:.0f} um; fusion boundary "
          f"G p50 {v['G']['p50']:.2e} K/m, R p50 {v['R']['p50']:.3f} m/s; pool bottom G {v['G_bottom']:.2e}, "
          f"R {v['R_bottom']:.3f}", flush=True)
    for dT0 in DT0:
        key = f"validation_pf|dT0={dT0:.0f}"
        if store.get(key) is None:
            out, r = run_pf(v["G"]["p50"], v["R"]["p50"], dT0)
            store.put(key, out)
            if dT0 == 60.0:
                save_field("validation", r)
        o = store.get(key)
        print(f"  dT0 = {dT0:.0f} K: spacing {o['spacing_um']:.2f} um (measured 0.52-0.64, mean 0.59); Hunt "
              f"{o['hunt_um']:.2f}, KF {o['kurz_fisher_um']:.2f}", flush=True)


# ------------------------------------------------------------------------------ gyroid
def part_window(store, name):
    key = f"window|{name}"
    if store.get(key) is not None:
        return store.get(key)
    x0, x1, y0, y1 = WINDOWS[name]
    margin = 0.1
    gy = gyroid()
    zb = Z_LAYER - LAYER                                          # top of the part built so far
    xc, yc = x0 - margin, 0.5 * (y0 + y1)
    Lx, Ly = (x1 - x0) + 2 * margin, (y1 - y0) + 2 * margin
    local = Field(lambda x, y, z: gy(x + xc, y + yc, z + zb), "gyroid (window)")
    md = MeltPoolModel(MAT, H, length=Lx, width=Ly, depth=0.15, powder=LAYER, symmetric_y=False, T0=T_PLATE,
                       emissivity=0.4, evaporation=EVAP)
    solid_frac = md.set_part_geometry(local)
    powder0 = np.asarray(md.powder_mask).copy()
    path, st = slice_scan_path(gy, Z_LAYER - 0.5 * LAYER, (x0, x1, y0, y1), hatch=HATCH, speed=V_MM_S,
                               contour_offset=CONTOUR_OFFSET)
    path = [ScanVector(v.x0 - xc, v.y0 - yc, v.x1 - xc, v.y1 - yc, v.speed, v.power) for v in path]
    print(f"window {name}: substrate solid fraction {solid_frac:.2f}, {st['contours']} contours "
          f"({st['contour_length_mm']:.1f} mm), {st['hatch_vectors']} hatch vectors ({st['hatch_length_mm']:.1f} mm), "
          f"scan time {st['scan_time_s'] * 1e3:.1f} ms", flush=True)
    t0 = time.perf_counter()
    res = md.run(laser(), path, t_extra=3e-4, verbose=True)
    print(f"  melt pool: {res.steps} steps, {time.perf_counter() - t0:.0f} s, energy balance {res.energy_balance:.1e}",
          flush=True)
    nzs = md.nz_sub
    fb = fusion_boundary(res, np.ones(res.T.shape, bool))
    # down-facing: the column had powder in the top substrate cell (the layer below) at the start
    down = powder0[fb["i"], fb["j"], nzs - 1]
    # wall thickness at the column (graded design value)
    xg = res.x[fb["i"]] + xc
    t_wall = 0.2 + 0.4 * np.clip(xg / L_CELL, 0, 1)
    classes = {"supported": ~down, "down-facing": down}
    out = dict(window=WINDOWS[name], solid_fraction=solid_frac, scan=st, steps=res.steps,
               energy_balance=float(res.energy_balance), wall_thickness_mm=summary(t_wall))
    for cname, sel in classes.items():
        out[cname] = dict(G=summary(fb["G"][sel]), R=summary(fb["R"][sel]), cr=summary(fb["cr"][sel]),
                          n_columns=int(sel.sum()))
    # grains: competitive epitaxial growth from a random 20 um substrate
    gr = S.grow_grains(res, substrate_grain_size=0.02, seed=1)
    gs = S.grain_stats(gr, res)
    out["grains"] = {k: (float(v) if np.isscalar(v) else v) for k, v in gs.items() if np.isscalar(v)}
    # maps for the figure: top view of the new layer (mid-layer z) and the fusion-boundary G
    kz = nzs + LAYER / 2 / H
    kz = int(min(kz, res.T.shape[2] - 1))
    maps = dict(x=res.x + xc, y=res.y + yc, n_melt=res.n_melt[:, :, kz].astype(np.int8),
                G=np.where(res.n_melt[:, :, kz] > 0, res.G[:, :, kz] * 1e3, np.nan).astype(np.float32),
                R=np.where(res.n_melt[:, :, kz] > 0, res.R[:, :, kz] * 1e-3, np.nan).astype(np.float32),
                powder_below=powder0[:, :, nzs - 1])
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    for k, v in maps.items():
        fields[f"{name}_{k}"] = v
    np.savez_compressed(FIELDS, **fields)
    store.put(key, out)
    return out


def part_gyroid(store):
    for name in WINDOWS:
        w = part_window(store, name)
        for c in ("supported", "down-facing"):
            d = w[c]
            if d["n_columns"]:
                print(f"  {name:5s} {c:11s}: {d['n_columns']:5d} columns, G p50 {d['G']['p50']:.2e} K/m, "
                      f"R p50 {d['R']['p50']:.3f} m/s, cooling {d['cr']['p50']:.1e} K/s", flush=True)
    # phase field at the medians of each class (baseline dT0 = 60 K)
    for name in WINDOWS:
        w = store.get(f"window|{name}")
        for c in ("supported", "down-facing"):
            d = w[c]
            if d["n_columns"] < 20:
                continue
            key = f"pf|{name}|{c}"
            if store.get(key) is None:
                out, r = run_pf(d["G"]["p50"], d["R"]["p50"], 60.0)
                store.put(key, out)
                save_field(f"{name}_{c}", r)
            o = store.get(key)
            print(f"  PF {name:5s} {c:11s}: spacing {o['spacing_um']:.2f} um, c_s*/c_inf {o['cs_rel']:.3f}, "
                  f"c_l*/c_inf {o['cl_rel']:.3f}, intercellular max {o['intercell_max_rel']:.2f} x nominal", flush=True)


def report(store):
    lines = ["## Validation: single LPBF layer (Piglione et al. 2018)"]
    v = store.get("validation_meltpool")
    if v:
        lines.append(f"Melt pool {v['width_um']:.0f} x {v['depth_um']:.0f} um; fusion boundary G = {v['G']['p50']:.2e} "
                     f"K/m, R = {v['R']['p50']:.3f} m/s (medians).")
        lines += ["", "| freezing range | phase-field spacing | Hunt | Kurz-Fisher | measured |", "|---|---|---|---|---|"]
        for dT0 in DT0:
            o = store.get(f"validation_pf|dT0={dT0:.0f}")
            if o:
                lines.append(f"| {dT0:.0f} K | {o['spacing_um']:.2f} um | {o['hunt_um']:.2f} um | {o['kurz_fisher_um']:.2f} um "
                             "| 0.52-0.64 um (mean 0.59) |")
    lines += ["", "## Gyroid sheet, layer at 1.3 mm", "",
              "| window | class | columns | G p50 [K/m] | R p50 [m/s] | cooling p50 [K/s] | PF spacing [um] "
              "| c_s*/c_inf | c_l*/c_inf | intercellular max / c_inf |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in WINDOWS:
        w = store.get(f"window|{name}")
        if not w:
            continue
        for c in ("supported", "down-facing"):
            d = w[c]
            o = store.get(f"pf|{name}|{c}") or {}
            if not d["n_columns"]:
                continue
            lines.append(f"| {name} | {c} | {d['n_columns']} | {d['G']['p50']:.2e} | {d['R']['p50']:.3f} | {d['cr']['p50']:.1e} | "
                         f"{o.get('spacing_um', float('nan')):.2f} | {o.get('cs_rel', float('nan')):.3f} | "
                         f"{o.get('cl_rel', float('nan')):.3f} | {o.get('intercell_max_rel', float('nan')):.2f} |")
        g = w.get("grains", {})
        lines.append(f"| {name} | grains | " + ", ".join(f"{k} = {v:.3g}" for k, v in g.items()) + " | | | | | | | |")
    text = "\n".join(lines)
    open(STORE.replace(".json", ".md"), "w").write(text + "\n")
    print(text)


BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def figure(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    F = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    fig = plt.figure(figsize=(13, 8.4))
    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.42, left=0.05, right=0.97, top=0.93, bottom=0.07)
    # (a) the layer, windows, scan path of the thin window
    ax = fig.add_subplot(gs[0, 0])
    g = np.linspace(0, L_CELL, 400)
    from tpmsfem.process.scanpath import slice_mask
    m = slice_mask(gyroid(), Z_LAYER - 0.5 * LAYER, g, g)
    ax.imshow(m.T, origin="lower", extent=[0, L_CELL, 0, L_CELL], cmap="Greys", vmin=0, vmax=2.2)
    for name, (x0, x1, y0, y1) in WINDOWS.items():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=BLUE if name == "thin" else ORANGE, lw=1.5))
        ax.text(x0, y1 + 0.05, f"{name} walls", color=INK, fontsize=8)
    ax.set_xlabel("x [mm] (wall 0.2 mm at x = 0 -> 0.6 mm at x = 4)", fontsize=8.5)
    ax.set_ylabel("y [mm]", fontsize=8.5)
    ax.set_title("(a) gyroid sheet layer at z = 1.3 mm", fontsize=9, loc="left")
    ax.tick_params(labelsize=8, colors=MUTED)
    # (b, c) G maps of the new layer with the down-facing outline
    for i, name in enumerate(WINDOWS):
        if f"{name}_G" not in F:
            continue
        ax = fig.add_subplot(gs[0, 1 + i])
        x, y = F[f"{name}_x"], F[f"{name}_y"]
        from matplotlib.colors import LogNorm

        im = ax.imshow(F[f"{name}_G"].T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis",
                       norm=LogNorm(2e5, 3e7))
        ax.contour(x, y, (F[f"{name}_powder_below"] & (F[f"{name}_n_melt"] > 0)).T.astype(float), [0.5],
                   colors=["#ffffff"], linewidths=0.8)
        ax.set_title(f"({'bc'[i]}) {name} walls: G at solidification, mid-layer\n"
                     "white outline: built over powder (down-facing)", fontsize=9, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8.5)
        ax.tick_params(labelsize=8, colors=MUTED)
        if i == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("G [K/m]", fontsize=8.5)
            cb.ax.tick_params(labelsize=8)
    # (d) spacing by class vs the validation
    ax = fig.add_subplot(gs[0, 3])
    ax.axhspan(*CM.PIGLIONE2018["cell_spacing_um"], color="#d6d5d0", lw=0)
    ax.text(0.02, CM.PIGLIONE2018["cell_spacing_um"][1] + 0.02, "measured, single layer\n(Piglione et al. 2018)",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="bottom")
    rows = [("validation", store.get("validation_pf|dT0=60"), INK)]
    for name, col in (("thin", BLUE), ("thick", ORANGE)):
        for c in ("supported", "down-facing"):
            lab = "over solid" if c == "supported" else "over powder"
            rows.append((f"{name}\n{lab}", store.get(f"pf|{name}|{c}"), col))
    for k, (lab, o, col) in enumerate(rows):
        if not o:
            continue
        lo, hi = min(o["spacing_um_range"]), max(o["spacing_um_range"])
        ax.errorbar(k, o["spacing_um"], yerr=[[o["spacing_um"] - lo], [hi - o["spacing_um"]]], fmt="o", color=col,
                    ms=7, capsize=2, mfc="white" if "powder" in lab else col, mew=1.5)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("primary cell spacing [um]", fontsize=8.5)
    ax.set_title("(d) cell spacing (2-D phase field, dT0 = 60 K)", fontsize=9, loc="left")
    _style(ax)
    # (e-h) Mn+Ni maps
    tags = [("validation", "validation (single layer)"), ("thin_supported", "thin wall, supported"),
            ("thin_down-facing", "thin wall, down-facing"), ("thick_down-facing", "thick wall, down-facing")]
    for i, (tag, title) in enumerate(tags):
        if f"{tag}_c" not in F:
            continue
        ax = fig.add_subplot(gs[1, i])
        c = F[f"{tag}_c"].astype(float) * 40.0
        dx = float(F[f"{tag}_dx"]) * 1e6
        im = ax.imshow(c, origin="lower", cmap="magma", vmin=34, vmax=56, extent=[0, c.shape[1] * dx, 0, c.shape[0] * dx])
        ax.set_title(f"({'efgh'[i]}) Mn+Ni, {title}", fontsize=9, loc="left")
        ax.set_xlabel("x [um]", fontsize=8.5)
        ax.tick_params(labelsize=8, colors=MUTED)
        if i == 3:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("Mn + Ni [wt%] (pseudo-binary)", fontsize=8.5)
            cb.ax.tick_params(labelsize=8)
    fig.suptitle("Solidification of an LPBF CoCrFeMnNi gyroid sheet (200 W, 0.75 m/s, 30 um layers): melt-pool model + "
                 "2-D phase field", fontsize=11, x=0.05, ha="left")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "validation", "gyroid", "report", "figure"])
    a = ap.parse_args()
    store = ResultStore(STORE)
    if a.part in ("all", "validation"):
        part_validation(store)
    if a.part in ("all", "gyroid"):
        part_gyroid(store)
    report(store)
    if a.part in ("all", "figure"):
        figure(store)
    json.dump(store.data, open(STORE, "w"), indent=1)


if __name__ == "__main__":
    main()
