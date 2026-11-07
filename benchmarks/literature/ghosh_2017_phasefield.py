"""Phase-field cells in laser-deposited Ni-Nb: tpmsfem vs Ghosh, Ma, Ofori-Opoku & Guyer (2017).

Literature: S. Ghosh, L. Ma, N. Ofori-Opoku, J.E. Guyer, "On the primary spacing and
microsegregation of cellular dendrites in laser deposited Ni-Nb alloys", Modelling Simul.
Mater. Sci. Eng. 25 (2017) 065002 (arXiv:1705.06669).

Their model is the same class as ``tpmsfem.process.phasefield``:
- a 2-D quantitative phase field (Echebarria et al. anti-trapping, frozen temperature);
- Ni-5 wt% Nb (k = 0.48, m = -10.5 K/wt%, D = 3e-9 m^2/s, Gamma = 3.65e-7 K m, eps4 = 0.03);
- W0 = 10 nm, dx = 8 nm;
- G = 1e7 K/m, with V varied.

Published values compared here:
- primary spacing 0.71 um at a cooling rate of 3e5 K/s (V = 0.03 m/s);
- primary spacing 0.23 um at 1e6 K/s (V = 0.1 m/s);
- the fit PDAS = 800 um * (G V)^-0.57 over 5e4 to 3e6 K/s;
- cell-tip solid composition c_s* = 3.65 wt% Nb at V = 0.03 m/s;
- the Hunt and Kurz-Fisher spacing models.

The script does three things:
1. Verification. Growth rates of small planar-front perturbations are checked against the
   Mullins-Sekerka dispersion relation. The alloy is the same, isotropic (eps4 = 0), at
   G = 1e7 K/m, V = 0.03 and 0.1 m/s, and W0 = 10 and 5 nm. This tests whether the
   thin-interface model is still quantitative at AM speeds: D/V is only 3-10 W0 there.
2. Code-to-code. Steady primary spacing and tip composition for V = 0.03 to 0.2 m/s, at
   Ghosh's W0 = 10 nm.
3. Convergence. The V = 0.1 m/s spacing is repeated at W0 = 5 nm.

    python benchmarks/literature/ghosh_2017_phasefield.py            # all parts, ~1.5 h on 4 cores
    python benchmarks/literature/ghosh_2017_phasefield.py --part ms  # verification only, ~4 min
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ResultStore  # noqa: E402

from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

STORE = "benchmarks/literature/results/ghosh2017_pf.json"
FIELDS = "benchmarks/literature/results/ghosh2017_pf_fields.npz"
FIG = "docs/img/pf_ghosh2017.png"
ALLOY = PF.NI5NB_GHOSH2017
G = 1.0e7
GHOSH = dict(points=[(3.0e5, 0.71), (1.0e6, 0.23)], fit_A_um=800.0, fit_n=-0.57, fit_range=(5e4, 3e6),
             cs_tip_wt=3.65, cl_max_wt=6.87, tip_V=0.03, W0_m=10e-9)
SC = S.SolidificationConstants(freezing_range=ALLOY.freezing_range, D_liquid=ALLOY.D,
                               gibbs_thomson=ALLOY.gibbs_thomson, partition=ALLOY.k)


def ghosh_fit(cr):
    return GHOSH["fit_A_um"] * np.asarray(cr, float) ** GHOSH["fit_n"]


# ----------------------------------------------------------------------------- verification
def ms_growth(V, W0, lam_m, iso=True):
    """Fitted growth rate of a cos(2 pi x / lam) perturbation vs the Mullins-Sekerka rate."""
    a = PF.DiluteAlloy(**{**ALLOY.__dict__, "eps4": 0.0}) if iso else ALLOY
    nx = max(int(round(lam_m / (0.8 * W0))), 8)
    Lx = nx * 0.8 * W0
    q = 2 * np.pi / Lx
    w_th = float(PF.mullins_sekerka_rate(a, G, V, q))
    m = PF.DirectionalPF(a, G, V, Lx, max(12 * a.D / V, 40 * W0) / 0.6, W0_m=W0, noise=0.0)
    x = (np.arange(m.nx) + 0.5) * m.dx
    z = (np.arange(m.nz) + 0.5) * m.dx
    A0 = 0.004 * Lx / W0
    m.phi = -np.tanh((z[:, None] - (m.z0 + A0 * np.cos(q * x))[None, :]) / np.sqrt(2.0))
    te = 1.0 / w_th
    ts, amps = [], []
    for _ in range(40):
        m.run(4.0 * te / 40, record_every=10 ** 9)
        h = PF.front_profile(m.phi, m.dx)
        ts.append(m.t * m.tau0)
        amps.append(2 * np.abs(np.fft.rfft(h)[1]) / m.nx)
    ts, amps = np.array(ts), np.array(amps)
    sel = (ts > te) & (amps < 0.03 * Lx / W0)                  # after the start-up, still linear
    w_pf = float(np.polyfit(ts[sel], np.log(amps[sel]), 1)[0])
    return dict(V=V, W0_m=W0, lam_m=Lx, q=q, omega_pf=w_pf, omega_ms=w_th, ratio=w_pf / w_th,
                D_over_V_in_W0=a.D / V / W0, n_fit=int(sel.sum()))


def part_ms(store):
    for V in (0.03, 0.1):
        qs = np.linspace(1e5, 1e8, 4000)
        w = PF.mullins_sekerka_rate(PF.DiluteAlloy(**{**ALLOY.__dict__, "eps4": 0.0}), G, V, qs)
        lam_max = 2 * np.pi / qs[np.nanargmax(w)]
        for W0 in (10e-9, 5e-9):
            for f in (0.75, 1.0, 1.5, 2.5):
                key = f"ms|V={V}|W0={W0:.0e}|f={f}"
                if store.get(key) is None:
                    r = ms_growth(V, W0, f * lam_max)
                    store.put(key, r)
                r = store.get(key)
                print(f"  MS  V = {V:.2f} m/s, W0 = {W0 * 1e9:.0f} nm, lambda = {r['lam_m'] * 1e9:4.0f} nm: "
                      f"omega_PF / omega_MS = {r['ratio']:.3f}   (D/V = {r['D_over_V_in_W0']:.0f} W0)", flush=True)


# ----------------------------------------------------------------------------- cells
def run_cells(V, W0, seed=0, fields=None):
    cr = G * V
    lam_exp = float(ghosh_fit(cr)) * 1e-6
    width = max(2.0e-6, 8 * lam_exp)
    grow = max(10e-6, 25 * lam_exp)
    m = PF.DirectionalPF(ALLOY, G, V, width, 3.0e-6, W0_m=W0, seed=seed)
    hist = []
    n_chunks = 40
    for i in range(n_chunks):
        r = m.run(grow / V / n_chunks, record_every=500)
        tips = r.tips()                       # active tips: within 2 % of dT0 of the leading one
        n = r.boundaries()                    # Nb-rich cell boundaries below the tips (as on a micrograph)
        lam = m.nx * m.dx * m.W0 / n if n else float("inf")
        h_tip = r.front_history[-1][1]
        theta = (h_tip - V * m.t * m.tau0) / (m.lT * m.W0)
        hist.append(dict(t_s=m.t * m.tau0, grown_m=V * m.t * m.tau0, n_cells=n, n_tips=len(tips), spacing_m=lam,
                         theta_tip=float(theta),
                         cs_tip=float(np.nanmedian([t["cs"] for t in tips])) if tips else float("nan"),
                         cl_tip=float(np.nanmedian([t["cl"] for t in tips])) if tips else float("nan")))
        print(f"    V = {V} m/s, W0 = {W0 * 1e9:.0f} nm: grown {hist[-1]['grown_m'] * 1e6:5.1f} um, "
              f"{n:4.1f} cells ({len(tips)} tips), theta_tip = {theta:+.3f}, c_s*/c_inf = {hist[-1]['cs_tip']:.3f} "
              f"[{r.seconds:.0f} s]", flush=True)
    late = hist[int(0.6 * n_chunks):]
    n_mean = float(np.mean([max(h["n_cells"], 1) for h in late]))
    n_tips = float(np.mean([max(h["n_tips"], 1) for h in late]))
    theta = float(np.mean([h["theta_tip"] for h in late]))
    cs = float(np.nanmean([h["cs_tip"] for h in late]))
    cl = float(np.nanmean([h["cl_tip"] for h in late]))
    seg = r.segregation()
    out = dict(V=V, W0_m=W0, G=G, cooling_rate=cr, width_m=m.nx * m.dx * m.W0, grown_m=hist[-1]["grown_m"],
               n_cells_mean=n_mean, spacing_um=m.nx * m.dx * m.W0 / n_mean * 1e6,
               spacing_tips_um=m.nx * m.dx * m.W0 / n_tips * 1e6,
               spacing_um_range=[m.nx * m.dx * m.W0 / max(h["n_cells"], 1) * 1e6 for h in late],
               tip_undercooling_K=ALLOY.freezing_range * (1.0 - theta), theta_tip=theta,
               cs_tip_wt=cs * ALLOY.c_inf, cl_tip_wt=cl * ALLOY.c_inf, k_eff=cs / cl,
               c_max_wt=seg["max"] * ALLOY.c_inf, c_p99_wt=seg["p99"] * ALLOY.c_inf,
               ghosh_fit_um=float(ghosh_fit(cr)), hunt_um=float(S.primary_spacing(G, V, SC, "hunt")),
               kurz_fisher_um=float(S.primary_spacing(G, V, SC, "kurz-fisher")),
               nx=m.nx, nz=m.nz, steps=int(round(m.t / m.dt)), history=hist)
    if fields is not None:
        top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
        lo = max(top - int(1.5e-6 / r.dx_m), 0)
        fields[f"V{V}_W{W0 * 1e9:.0f}_c"] = r.c[lo:top + 8].astype(np.float16)
        fields[f"V{V}_W{W0 * 1e9:.0f}_dx"] = np.array(r.dx_m)
    return out


def part_cells(store, Vs=(0.2, 0.1, 0.05, 0.03), W0s=(10e-9,)):
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    for W0 in W0s:
        for V in Vs:
            key = f"cells|V={V}|W0={W0:.0e}"
            if store.get(key) is None:
                store.put(key, run_cells(V, W0, fields=fields))
                np.savez_compressed(FIELDS, **fields)
            r = store.get(key)
            print(f"  cells  V = {V:.2f} m/s (G V = {r['cooling_rate']:.1e} K/s), W0 = {W0 * 1e9:.0f} nm: "
                  f"spacing {r['spacing_um']:.3f} um (Ghosh fit {r['ghosh_fit_um']:.3f}, Hunt {r['hunt_um']:.3f}, "
                  f"KF {r['kurz_fisher_um']:.3f}), c_s* = {r['cs_tip_wt']:.2f} wt%, "
                  f"tip undercooling {r['tip_undercooling_K']:.1f} K", flush=True)


# ----------------------------------------------------------------------------- report
def report(store):
    rows = ["| V [m/s] | G V [K/s] | W0 [nm] | spacing, tpmsfem [um] | (active tips) | Ghosh 2017 fit [um] | ratio | Hunt [um] "
            "| Kurz-Fisher [um] | c_s* [wt% Nb] | c_l* [wt% Nb] | c_s*/c_l* | tip undercooling [K] |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(k for k in store.data if k.startswith("cells|")):
        r = store.get(key)
        rows.append(f"| {r['V']:.2f} | {r['cooling_rate']:.1e} | {r['W0_m'] * 1e9:.0f} | {r['spacing_um']:.3f} "
                    f"({min(r['spacing_um_range']):.2f}-{max(r['spacing_um_range']):.2f}) | {r['spacing_tips_um']:.3f} | "
                    f"{r['ghosh_fit_um']:.3f} | "
                    f"{r['spacing_um'] / r['ghosh_fit_um']:.2f} | {r['hunt_um']:.3f} | {r['kurz_fisher_um']:.3f} | "
                    f"{r['cs_tip_wt']:.2f} | {r['cl_tip_wt']:.2f} | {r['k_eff']:.2f} | {r['tip_undercooling_K']:.1f} |")
    ms = ["| V [m/s] | W0 [nm] | D/V [W0] | wavelength [nm] | omega_PF / omega_MS |", "|---|---|---|---|---|"]
    for key in sorted(k for k in store.data if k.startswith("ms|")):
        r = store.get(key)
        ms.append(f"| {r['V']:.2f} | {r['W0_m'] * 1e9:.0f} | {r['D_over_V_in_W0']:.0f} | {r['lam_m'] * 1e9:.0f} | "
                  f"{r['ratio']:.3f} |")
    text = "\n".join(["## Mullins-Sekerka verification", *ms, "", "## Cells vs Ghosh et al. (2017)", *rows, "",
                      f"Ghosh et al. (2017), V = {GHOSH['tip_V']} m/s: c_s* = {GHOSH['cs_tip_wt']} wt%, "
                      f"c_l,max = {GHOSH['cl_max_wt']} wt% (c_s*/c_l = 0.53); spacing 0.71 um at 3e5 K/s, "
                      "0.23 um at 1e6 K/s."])
    open(STORE.replace(".json", ".md"), "w").write(text + "\n")
    print(text)


def figure(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0.05, 0.12, 0.27, 0.76])
    a_iso = PF.DiluteAlloy(**{**ALLOY.__dict__, "eps4": 0.0})
    for V, col in ((0.03, "C0"), (0.1, "C3")):
        lam = np.linspace(0.12e-6, 1.0e-6, 300)
        ax.plot(lam * 1e9, PF.mullins_sekerka_rate(a_iso, G, V, 2 * np.pi / lam) / 1e6, col, lw=1.5,
                label=f"Mullins-Sekerka, V = {V} m/s")
        for W0, mk in ((10e-9, "o"), (5e-9, "s")):
            pts = [store.get(k) for k in store.data if k.startswith(f"ms|V={V}|W0={W0:.0e}")]
            if pts:
                ax.plot([p["lam_m"] * 1e9 for p in pts], [p["omega_pf"] / 1e6 for p in pts], mk, color=col,
                        mfc="none" if W0 > 6e-9 else col, label=f"phase field, W0 = {W0 * 1e9:.0f} nm")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("perturbation wavelength [nm]")
    ax.set_ylabel("growth rate [1/us]")
    ax.set_title("(a) planar-front instability, Ni-5Nb, G = 1e7 K/m", fontsize=9)
    ax.legend(fontsize=6.5)
    ax = fig.add_axes([0.38, 0.12, 0.27, 0.76])
    cr = np.logspace(np.log10(5e4), np.log10(3e6), 50)
    ax.loglog(cr, ghosh_fit(cr), "k-", lw=1.5, label="Ghosh 2017 PF fit, 800 (GV)^-0.57")
    ax.loglog(*zip(*GHOSH["points"]), "k*", ms=10, label="Ghosh 2017 PF, quoted")
    ax.loglog(cr, S.primary_spacing(G, cr / G, SC, "hunt"), "C2--", lw=1, label="Hunt")
    ax.loglog(cr, S.primary_spacing(G, cr / G, SC, "kurz-fisher"), "C1--", lw=1, label="Kurz-Fisher")
    for W0, mk in ((10e-9, "o"), (5e-9, "s")):
        pts = [store.get(k) for k in store.data if k.startswith("cells|") and abs(store.get(k)["W0_m"] - W0) < 1e-10]
        if pts:
            ax.errorbar([p["cooling_rate"] for p in pts], [p["spacing_um"] for p in pts],
                        yerr=[[p["spacing_um"] - min(p["spacing_um_range"]) for p in pts],
                              [max(p["spacing_um_range"]) - p["spacing_um"] for p in pts]],
                        fmt=mk, color="C0", mfc="none" if W0 > 6e-9 else "C0", capsize=2,
                        label=f"tpmsfem, W0 = {W0 * 1e9:.0f} nm")
    ax.set_xlabel("cooling rate G V [K/s]")
    ax.set_ylabel("primary spacing [um]")
    ax.set_title("(b) primary spacing, G = 1e7 K/m", fontsize=9)
    ax.legend(fontsize=6.5)
    keys = [k for k in ("V0.03_W10_c", "V0.1_W10_c") if k in fields]
    for i, k in enumerate(keys):
        ax = fig.add_axes([0.70 + 0.15 * i, 0.12, 0.13, 0.76])
        c = fields[k].astype(float) * ALLOY.c_inf
        dx = float(fields[k.replace("_c", "_dx")]) * 1e6
        im = ax.imshow(c, origin="lower", cmap="magma", vmin=2.0, vmax=9.0,
                       extent=[0, c.shape[1] * dx, 0, c.shape[0] * dx])
        ax.set_title(f"(c) V = {k[1:].split('_')[0]} m/s", fontsize=9)
        ax.set_xlabel("x [um]")
        if i == 0:
            ax.set_ylabel("z [um]")
    if keys:
        cb = fig.colorbar(im, ax=fig.axes[-len(keys):], fraction=0.05, pad=0.02)
        cb.set_label("Nb [wt%]")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "ms", "cells", "conv", "report", "figure"])
    args = ap.parse_args()
    store = ResultStore(STORE)
    if args.part in ("all", "ms"):
        part_ms(store)
    if args.part in ("all", "cells"):
        part_cells(store)
    if args.part in ("all", "conv"):
        part_cells(store, Vs=(0.1,), W0s=(5e-9,))       # the hardest case for W0 = 10 nm (D/V = 3 W0)
    if args.part in ("all", "report", "ms", "cells", "conv"):
        report(store)
    if args.part in ("all", "figure"):
        figure(store)


if __name__ == "__main__":
    main()
