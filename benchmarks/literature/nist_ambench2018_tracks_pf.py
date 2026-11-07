"""AM-Bench 2018-02 case B: phase-field cells at the melt-pool G and R vs measured IN625 cells.

Chain:
1. The calibrated case-B melt pool (179.2 W, 800 mm/s; nist_ambench2018_tracks.py) gives,
   in the transverse section at x = 0.7 mm, the thermal gradient G and the growth rate
   R = v |n_x| at solidification. The steady-pool relation R = v cos(alpha) is smoother on
   the 5 um voxels than cooling rate / G. These are sampled up the pool centreline, from the
   fusion boundary (where the cells start) towards the surface.
2. At each (G, R), the 2-D quantitative phase field (tpmsfem.process.phasefield) grows cells
   in IN625, treated as quasi-binary Ni-4 wt% Nb. The model and alloy follow Ghosh et al.
   (2018); W0 = 10 nm.
3. The primary spacing and the Nb partitioning are compared with:
   - Ghosh, Ma, Levine, Ricker, Stoudt, Heigel & Guyer, JOM 70 (2018) 1011 (arXiv:1802.05827).
     Single tracks on an IN625 plate at 195 W / 800 mm/s, the nominal LPBF setting, close to
     case B. The SEM average primary spacing is about 0.2 to 1.0 um, coarse at the pool bottom
     and fine at its rear. Their phase field gives 0.20 to 0.43 um for G = 0.6 to 1.5e7 K/m and
     R = 0.05 to 0.2 m/s, with cell cores of 3.7 wt% Nb and a liquid spike of 5.1 wt%.
   - The Hunt and Kurz-Fisher spacing models with the same alloy constants.

Validity: the Mullins-Sekerka check in ghosh_2017_phasefield.py shows that W0 = 10 nm is
quantitative only for D/V >~ 10 W0, i.e. R <~ 0.03 m/s. Points at larger R are reported with
the convergence error measured there.

    python benchmarks/literature/nist_ambench2018_tracks_pf.py     # ~20 min (melt pool ~4 min)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ResultStore  # noqa: E402

from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

STORE = "benchmarks/literature/results/ambench2018_tracks_pf.json"
FIELDS = "benchmarks/literature/results/ambench2018_tracks_pf_fields.npz"
FIG = "docs/img/pf_ambench_caseB.png"
ALLOY = PF.IN625_NB
W0 = 10e-9
V_SCAN = 0.8                                     # m/s
MEASURED = dict(spacing_um=(0.2, 1.0), source="Ghosh et al., JOM 70 (2018) 1011: SEM, 195 W / 800 mm/s single track",
                pf_spacing_um=(0.20, 0.43), pf_G=(0.6e7, 1.5e7), pf_R=(0.05, 0.2), pf_cs_wt=3.7, pf_cl_wt=5.1)
SC = S.SolidificationConstants(freezing_range=ALLOY.freezing_range, D_liquid=ALLOY.D,
                               gibbs_thomson=ALLOY.gibbs_thomson, partition=ALLOY.k)


def section_path(store):
    """(z, G, R) up the pool centreline of the case-B section, from the fusion boundary."""
    if store.get("path") is None:
        import nist_ambench2018_tracks_ebsd as EB

        res, i = EB.simulate_section()
        mel = (res.n_melt[i] > 0) & np.isfinite(res.t_solid[i])
        cols = slice(0, 3)                       # |y| < 15 um (half-width mirrored model)
        pts = []
        for k, z in enumerate(res.z):
            sel = mel[cols, k]
            if sel.sum() >= 2:
                pts.append(dict(z_um=float(z * 1e3), G=float(np.median(res.G[i][cols, k][sel]) * 1e3),
                                R=float(np.median(V_SCAN * np.abs(res.grad_dir[i][cols, k, 0][sel]))),
                                R_cr_over_G=float(np.median(res.R[i][cols, k][sel]) * 1e-3)))
        store.put("path", dict(x_mm=float(res.x[i]), points=pts))
    return store.get("path")


def run_point(G, R, width=4.0e-6, height=2.0e-6, grow=12e-6, seed=0, fields=None, tag=None):
    m = PF.DirectionalPF(ALLOY, G, R, width, height, W0_m=W0, seed=seed)
    hist = []
    n_chunks = 30
    for _ in range(n_chunks):
        r = m.run(grow / R / n_chunks, record_every=500)
        tips = r.tips()
        nb = r.boundaries()
        theta = (r.front_history[-1][1] - R * m.t * m.tau0) / (m.lT * m.W0)
        hist.append(dict(grown_m=R * m.t * m.tau0, n_cells=nb, n_tips=len(tips), theta_tip=float(theta),
                         cs=float(np.nanmedian([t["cs"] for t in tips])) if tips else float("nan"),
                         cl=float(np.nanmedian([t["cl"] for t in tips])) if tips else float("nan")))
        print(f"    G = {G:.2e} K/m, R = {R:.3f} m/s: grown {hist[-1]['grown_m'] * 1e6:5.1f} um, {nb:4.1f} cells ({len(tips)} tips), "
              f"theta_tip {theta:+.3f}, c_s* {hist[-1]['cs'] * ALLOY.c_inf:.2f} wt% [{r.seconds:.0f} s]", flush=True)
    late = hist[int(0.6 * n_chunks):]
    Lx = m.nx * m.dx * m.W0
    n = float(np.mean([max(h["n_cells"], 1) for h in late]))
    win = r.c[r.phi < 0.0]
    n_tips = float(np.mean([max(h["n_tips"], 1) for h in late]))
    out = dict(G=G, R=R, cooling_rate=G * R, width_m=Lx, spacing_um=Lx / n * 1e6, spacing_tips_um=Lx / n_tips * 1e6,
               spacing_um_range=[Lx / max(h["n_cells"], 1) * 1e6 for h in late],
               cs_tip_wt=float(np.nanmean([h["cs"] for h in late])) * ALLOY.c_inf,
               cl_tip_wt=float(np.nanmean([h["cl"] for h in late])) * ALLOY.c_inf,
               intercellular_liquid_max_wt=float(win.max()) * ALLOY.c_inf if win.size else float("nan"),
               tip_undercooling_K=ALLOY.freezing_range * (1.0 - float(np.mean([h["theta_tip"] for h in late]))),
               D_over_V_in_W0=ALLOY.D / R / W0, hunt_um=float(S.primary_spacing(G, R, SC, "hunt")),
               kurz_fisher_um=float(S.primary_spacing(G, R, SC, "kurz-fisher")),
               kurz_fisher_default_um=float(S.primary_spacing(G, R)), steps=int(round(m.t / m.dt)), history=hist)
    if fields is not None and tag:
        top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
        lo = max(top - int(1.5e-6 / r.dx_m), 0)
        fields[tag + "_c"] = r.c[lo:top + 8].astype(np.float16)
        fields[tag + "_dx"] = np.array(r.dx_m)
    return out


def report(store):
    path = store.get("path")
    lines = ["| z [um] | G [K/m] | R [m/s] | G R [K/s] | D/V [W0] | spacing, PF [um] | Hunt [um] | Kurz-Fisher [um] "
             "| c_s* [wt% Nb] | c_l* [wt% Nb] | intercellular max [wt% Nb] | tip undercooling [K] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in path["points"]:
        r = store.get(f"pf|z={p['z_um']:.1f}")
        if r is None:
            continue
        lines.append(f"| {p['z_um']:.1f} | {r['G']:.2e} | {r['R']:.3f} | {r['cooling_rate']:.1e} | "
                     f"{r['D_over_V_in_W0']:.1f} | {r['spacing_um']:.2f} ({min(r['spacing_um_range']):.2f}-"
                     f"{max(r['spacing_um_range']):.2f}) | {r['hunt_um']:.2f} | {r['kurz_fisher_um']:.2f} | "
                     f"{r['cs_tip_wt']:.2f} | {r['cl_tip_wt']:.2f} | {r['intercellular_liquid_max_wt']:.1f} | "
                     f"{r['tip_undercooling_K']:.1f} |")
    m = MEASURED
    lines += ["", f"Measured ({m['source']}): average primary spacing {m['spacing_um'][0]}-{m['spacing_um'][1]} um. "
              f"Their PF: {m['pf_spacing_um'][0]}-{m['pf_spacing_um'][1]} um, c_s = {m['pf_cs_wt']} wt%, "
              f"liquid spike {m['pf_cl_wt']} wt%."]
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

    path = store.get("path")
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    pts = [(p, store.get(f"pf|z={p['z_um']:.1f}")) for p in path["points"]]
    pts = [(p, r) for p, r in pts if r is not None]
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], width_ratios=[1, 1, 0.15, 2.2], hspace=0.32, wspace=0.35,
                          left=0.06, right=0.98, top=0.93, bottom=0.08)
    zs = [p["z_um"] for p in path["points"]]
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx([p["G"] for p in path["points"]], zs, "o-", color=BLUE, lw=1.5, ms=5)
    ax.set_xlabel("G [K/m]", fontsize=9)
    ax.set_ylabel("height in the section [um] (surface = 0)", fontsize=9)
    ax.set_title("(a) thermal gradient", fontsize=9, loc="left")
    _style(ax)
    ax = fig.add_subplot(gs[0, 1], sharey=ax)
    ax.plot([p["R"] for p in path["points"]], zs, "o-", color=BLUE, lw=1.5, ms=5, label="v |n_x| (used)")
    ax.plot([p["R_cr_over_G"] for p in path["points"]], zs, "x", color=MUTED, ms=6, label="cooling rate / G")
    ax.set_xlabel("R [m/s]", fontsize=9)
    ax.set_title("growth rate", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    plt.setp(ax.get_yticklabels(), visible=False)
    _style(ax)
    ax = fig.add_subplot(gs[0, 3])
    from matplotlib.patches import Patch

    ax.axhspan(*MEASURED["spacing_um"], color="#d6d5d0", lw=0)
    ax.axhspan(*MEASURED["pf_spacing_um"], facecolor="none", edgecolor="#8f8e89", hatch="///", lw=0)
    ax.legend(handles=[Patch(color="#d6d5d0", label="measured, Ghosh et al. 2018 (SEM, 195 W / 800 mm/s)"),
                       Patch(facecolor="none", edgecolor="#8f8e89", hatch="///", label="Ghosh et al. 2018 phase field")],
              fontsize=7.5, frameon=False, loc="upper left")
    z = np.array([p["z_um"] for p, _ in pts])
    ax.plot(z, [r["hunt_um"] for _, r in pts], "--", color=ORANGE, lw=1.5)
    ax.plot(z, [r["kurz_fisher_um"] for _, r in pts], "--", color=AQUA, lw=1.5)
    ax.errorbar(z, [r["spacing_um"] for _, r in pts],
                yerr=[[r["spacing_um"] - min(r["spacing_um_range"]) for _, r in pts],
                      [max(r["spacing_um_range"]) - r["spacing_um"] for _, r in pts]],
                fmt="o-", color=BLUE, lw=1.5, ms=6, capsize=2)
    x_end = z.max() + 0.3
    ax.text(x_end, pts[-1][1]["hunt_um"], "Hunt", color=INK, fontsize=8, va="center")
    ax.text(x_end, pts[-1][1]["kurz_fisher_um"], "Kurz-Fisher", color=INK, fontsize=8, va="center")
    ax.text(z.mean(), 0.12, "tpmsfem, 2-D phase field", color=INK, fontsize=8, ha="center")
    ax.set_xlim(z.min() - 0.8, z.max() + 3.2)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("height in the section [um] (fusion boundary at left)", fontsize=9)
    ax.set_ylabel("primary cell spacing [um]", fontsize=9)
    ax.set_title("(b) cell spacing up the case-B centreline", fontsize=9, loc="left")
    _style(ax)
    keys = sorted((k for k in fields if k.endswith("_c")), key=lambda k: float(k[1:-2]))
    keys = [keys[0], keys[-1]] if len(keys) > 1 else keys
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[1, 0:2] if i == 0 else gs[1, 3])
        c = fields[k].astype(float) * ALLOY.c_inf
        dx = float(fields[k.replace("_c", "_dx")]) * 1e6
        im = ax.imshow(c, origin="lower", cmap="magma", vmin=2.0, vmax=8.0, extent=[0, c.shape[1] * dx, 0, c.shape[0] * dx])
        p = next(p for p in path["points"] if f"z{p['z_um']:.1f}_c" == k)
        ax.set_title(f"({'c' if i == 0 else 'd'}) Nb, height {p['z_um']:.1f} um: G = {p['G']:.1e} K/m, "
                     f"R = {p['R']:.3f} m/s", fontsize=9, loc="left")
        ax.set_xlabel("x [um]", fontsize=9)
        ax.set_ylabel("z [um]", fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
    if keys:
        cb = fig.colorbar(im, ax=fig.axes[-len(keys):], fraction=0.025, pad=0.01)
        cb.set_label("Nb [wt%]", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)


def main():
    store = ResultStore(STORE)
    path = section_path(store)
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    for p in path["points"]:
        if p["R"] > 0.15 or p["z_um"] > -15:
            continue                            # far outside the converged range of W0 = 10 nm
        key = f"pf|z={p['z_um']:.1f}"
        if store.get(key) is None:
            store.put(key, run_point(p["G"], p["R"], fields=fields, tag=f"z{p['z_um']:.1f}"))
            np.savez_compressed(FIELDS, **fields)
    report(store)
    figure(store)
    json.dump(store.data, open(STORE, "w"), indent=1)


if __name__ == "__main__":
    main()
