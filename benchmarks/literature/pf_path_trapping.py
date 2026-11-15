"""AM-Bench 2018-02 case B: phase-field cells up the melt-pool centreline, with solute trapping.

Extends nist_ambench2018_tracks_pf.py (W0 = 10 nm, lower half of the pool only) to the whole
path from the fusion boundary to the surface:

1. The same centreline (z, G, R) of the calibrated case-B melt pool (179.2 W, 800 mm/s, section
   at x = 0.7 mm), read from ambench2018_tracks_pf.json. R rises from 0.027 m/s at the fusion
   boundary to 0.29 m/s near the surface, so D/V falls from 110 to 10 nm.
2. Solute trapping by the continuous growth model (Aziz & Kaplan), as velocity-corrected alloy
   parameters (PF.DiluteAlloy.at_velocity): k(V), m(V) and the freezing range at the local R,
   with the diffusive speed V_D = 1 m/s (assumed; values of 0.5-2 m/s are used for Ni-Nb).
3. The interface width follows the diffusion length: W0 = min(10 nm, D/(5 R)), i.e.
   D/V >= 5 W0 at every point. The full criterion D/V >= 10 W0 would cost ~300 CPU-hours here
   (the cost scales as W0^-5) and is left for the GPU; the error at 5 W0 is measured with a
   convergence series at z = -22.5 um (W0 = 10, 6.5 and 4.1 nm: D/V = 3.3, 5 and 8 W0).
4. One run without trapping at z = -12.5 um (same W0) isolates the trapping effect.

Outputs per point: primary spacing (cell boundaries and active tips), tip solid and liquid
compositions c_s*, c_l*, the effective partition c_s*/c_l*, and the tip undercooling.
Compared with Ghosh et al., JOM 70 (2018) 1011 (SEM 0.2-1.0 um, 195 W / 800 mm/s; their phase
field 0.20-0.43 um, cell cores 3.7 wt% Nb) and the Hunt and Kurz-Fisher spacing models.

Windows: 4 x 2 um and 12 um of growth at W0 = 10 nm (as before); 2.4 x 1.2 um and 6 um of
growth below that; 1.6 x 0.8 um for the two fastest points. The cell count settles within
~3 um of growth (previous runs); the last 40 % of the growth is averaged. Spacings are
quantised by the periodic width (L/n).

    python benchmarks/literature/pf_path_trapping.py jobs              # list jobs and cost
    python benchmarks/literature/pf_path_trapping.py run TAG [TAG ...]  # NUMBA_NUM_THREADS sets threads
    python benchmarks/literature/pf_path_trapping.py report
    python benchmarks/literature/pf_path_trapping.py figure
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ResultStore  # noqa: E402

from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

PATH_STORE = "benchmarks/literature/results/ambench2018_tracks_pf.json"
STORE = "benchmarks/literature/results/pf_path_trapping.json"
FIELDS = "benchmarks/literature/results/pf_path_trapping_fields.npz"
FIG = "docs/img/pf_path_trapping.png"
ALLOY = PF.IN625_NB
V_D = 1.0                                        # m/s, diffusive speed for trapping
DV_W0 = 5.0                                      # D/V in units of W0
MEASURED = dict(spacing_um=(0.2, 1.0), pf_spacing_um=(0.20, 0.43), pf_cs_wt=3.7,
                source="Ghosh et al., JOM 70 (2018) 1011")


def path_points():
    return json.load(open(PATH_STORE))["path"]["points"]


def jobs():
    """tag -> (z, G, R, W0, trapping, width, height, grow)."""
    out = {}
    pts = path_points()
    for i, p in enumerate(pts):
        z, G, R = p["z_um"], p["G"], p["R"]
        W0 = min(10e-9, ALLOY.D / R / DV_W0)
        if W0 >= 9.99e-9:
            win = (4.0e-6, 2.0e-6, 12e-6)
        elif i >= len(pts) - 2:
            win = (1.6e-6, 0.8e-6, 6e-6)
        else:
            win = (2.4e-6, 1.2e-6, 6e-6)
        out[f"z={z:.1f}|trap"] = (z, G, R, W0, True) + win
        if abs(z + 22.5) < 0.1:                  # convergence series in W0
            for dv in (None, 8.0):
                W = 10e-9 if dv is None else ALLOY.D / R / dv
                out[f"z={z:.1f}|trap|W0={W * 1e9:.1f}"] = (z, G, R, W, True, 2.4e-6, 1.2e-6, 6e-6)
        if abs(z + 12.5) < 0.1:                  # trapping off, same W0 and window
            out[f"z={z:.1f}|eq"] = (z, G, R, W0, False) + win
    return out


def cost(job):
    """Rough cost in cell-steps."""
    z, G, R, W0, trap, w, h, grow = job
    a = ALLOY.at_velocity(R, V_D) if trap else ALLOY
    m = PF.DirectionalPF(a, G, R, 20 * W0, 20 * W0, W0_m=W0)
    m.nz = int(round(h / (m.dx * W0)))
    m.adapt_dt(tip_row=int(0.5 * m.nz))
    return (w / (m.dx * W0)) * (h / (m.dx * W0)) * grow / R / (m.dt * m.tau0)


def run_job(tag):
    z, G, R, W0, trap, width, height, grow = jobs()[tag]
    alloy = ALLOY.at_velocity(R, V_D) if trap else ALLOY
    t_start = time.perf_counter()
    m = PF.DirectionalPF(alloy, G, R, width, height, W0_m=W0, seed=0)
    hist = []
    n_chunks = 30
    steps = 0
    for _ in range(n_chunks):
        r = m.run(grow / R / n_chunks, record_every=500, adapt_dt=True)
        steps += r.steps
        tips = r.tips()
        nb = r.boundaries()
        theta = (r.front_history[-1][1] - R * m.t * m.tau0) / (m.lT * m.W0)
        hist.append(dict(grown_m=R * m.t * m.tau0, n_cells=nb, n_tips=len(tips), theta_tip=float(theta),
                         cs=float(np.nanmedian([t["cs"] for t in tips])) if tips else float("nan"),
                         cl=float(np.nanmedian([t["cl"] for t in tips])) if tips else float("nan"),
                         dt=m.dt))
        print(f"  [{tag}] grown {hist[-1]['grown_m'] * 1e6:5.2f} um, {nb:4.1f} cells ({len(tips)} tips), "
              f"theta_tip {theta:+.3f}, c_s* {hist[-1]['cs'] * alloy.c_inf:.2f} wt%, dt {m.dt:.4f} "
              f"[{time.perf_counter() - t_start:.0f} s]", flush=True)
    late = hist[int(0.6 * n_chunks):]
    Lx = m.nx * m.dx * m.W0
    n = float(np.mean([max(h["n_cells"], 1) for h in late]))
    n_tips = float(np.mean([max(h["n_tips"], 1) for h in late]))
    seg = r.segregation()
    sc = S.SolidificationConstants(freezing_range=alloy.freezing_range, D_liquid=alloy.D,
                                   gibbs_thomson=alloy.gibbs_thomson, partition=alloy.k)
    out = dict(z_um=z, G=G, R=R, W0_nm=W0 * 1e9, D_over_V_in_W0=alloy.D / R / W0, trapping=trap, V_D=V_D if trap else None,
               k=alloy.k, freezing_range_K=alloy.freezing_range, width_um=Lx * 1e6, height_um=m.nz * m.dx * m.W0 * 1e6,
               grown_um=R * m.t * m.tau0 * 1e6,
               spacing_um=Lx / n * 1e6, spacing_tips_um=Lx / n_tips * 1e6,
               spacing_um_range=[Lx / max(h["n_cells"], 1) * 1e6 for h in late],
               cs_tip_wt=float(np.nanmean([h["cs"] for h in late])) * alloy.c_inf,
               cl_tip_wt=float(np.nanmean([h["cl"] for h in late])) * alloy.c_inf,
               tip_undercooling_K=alloy.freezing_range * (1.0 - float(np.mean([h["theta_tip"] for h in late]))),
               solid_min_wt=seg["min"] * alloy.c_inf, solid_p50_wt=seg["p50"] * alloy.c_inf,
               solid_p95_wt=seg["p95"] * alloy.c_inf,
               hunt_um=float(S.primary_spacing(G, R, sc, "hunt")),
               kurz_fisher_um=float(S.primary_spacing(G, R, sc, "kurz-fisher")),
               steps=steps, cpu_s=time.perf_counter() - t_start,
               history=hist)
    ResultStore(STORE).put(tag, out)
    # composition near the tips for the figure (float16, 1 um below the leading tip)
    top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
    lo = max(top - int(1.0e-6 / r.dx_m), 0)
    np.savez_compressed(FIELDS.replace(".npz", f"_{_safe(tag)}.npz"), c=(r.c[lo:top + 8] * alloy.c_inf).astype(np.float16),
                        dx=np.array(r.dx_m))
    print(f"[{tag}] done: spacing {out['spacing_um']:.3f} um, c_s* {out['cs_tip_wt']:.2f}, c_l* {out['cl_tip_wt']:.2f} wt%, "
          f"{out['cpu_s'] / 3600:.2f} h", flush=True)
    return out


def _safe(tag):
    return tag.replace("|", "_").replace("=", "").replace(".", "p")


def report():
    store = ResultStore(STORE)
    old = ResultStore(PATH_STORE)
    rows = ["| height above the fusion boundary [um] | G [K/m] | R [m/s] | k(V) | W0 [nm] | D/V [W0] | spacing [um] "
            "| spacing, tips [um] | Hunt [um] | c_s* [wt% Nb] | c_l* [wt% Nb] | c_s*/c_l* | tip undercooling [K] "
            "| W0 = 10 nm, no trapping: spacing [um], c_s* [wt%] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    z0 = min(p["z_um"] for p in path_points())
    for tag, j in sorted(jobs().items(), key=lambda kv: (kv[1][0], -kv[1][3])):
        r = store.get(tag)
        if r is None:
            continue
        o = old.get(f"pf|z={r['z_um']:.1f}")
        prev = f"{o['spacing_um']:.2f}, {o['cs_tip_wt']:.2f}" if o and tag.endswith("|trap") else ""
        label = f"{r['z_um'] - z0:.0f}" + ("" if tag.endswith("|trap") else (" (no trapping)" if tag.endswith("|eq") else ""))
        rows.append(f"| {label} | {r['G']:.2e} | {r['R']:.3f} | {r['k']:.3f} | {r['W0_nm']:.1f} | {r['D_over_V_in_W0']:.1f} | "
                    f"{r['spacing_um']:.3f} ({min(r['spacing_um_range']):.2f}-{max(r['spacing_um_range']):.2f}) | "
                    f"{r['spacing_tips_um']:.3f} | {r['hunt_um']:.2f} | {r['cs_tip_wt']:.2f} | {r['cl_tip_wt']:.2f} | "
                    f"{r['cs_tip_wt'] / r['cl_tip_wt']:.2f} | {r['tip_undercooling_K']:.1f} | {prev} |")
    text = "\n".join(rows)
    text += (f"\n\nV_D = {V_D} m/s. Measured ({MEASURED['source']}): SEM spacing {MEASURED['spacing_um'][0]}-"
             f"{MEASURED['spacing_um'][1]} um; their phase field {MEASURED['pf_spacing_um'][0]}-{MEASURED['pf_spacing_um'][1]} um "
             f"with cell cores at {MEASURED['pf_cs_wt']} wt% Nb.\n")
    open(STORE.replace(".json", ".md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "jobs":
        tot = 0.0
        for tag, j in jobs().items():
            c = cost(j)
            tot += c
            print(f"{tag:28s} W0 {j[3] * 1e9:4.1f} nm window {j[5] * 1e6:.1f} x {j[6] * 1e6:.1f} um, grow {j[7] * 1e6:.0f} um: "
                  f"{c:.2e} cell-steps (~{c / 4.2e7 / 3600:.1f} h on one thread)")
        print(f"total {tot:.2e} cell-steps")
    elif cmd == "run":
        for tag in sys.argv[2:]:
            if ResultStore(STORE).get(tag) is None:
                run_job(tag)
    elif cmd == "report":
        report()
    elif cmd == "figure":
        import pf_path_trapping_figure as F
        F.figure()
