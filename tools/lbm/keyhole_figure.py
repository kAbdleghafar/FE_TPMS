"""Keyhole stage 2 results against the X-ray data: reads the runs of keyhole_ti64.py in a folder
and draws the figure (docs/img/keyhole_ti64.png) and a summary (keyhole_summary.json).

Per run, the quasi-steady window is the second half of the track (the beam between 50 and 100 %
of the track length): mean, minimum and maximum keyhole depth (Gan et al.'s error bars are the
same max/min over the middle of their tracks), mean absorbed fraction, mean front-wall tangent.

    python tools/lbm/keyhole_figure.py [folder]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks", "literature"))


def summarise(path):
    d = json.load(open(path))
    log = [r for r in d["log"] if r.get("x_l") is not None and np.isfinite(r.get("depth", np.nan))]
    if not log:
        return None
    x = np.array([r["x_l"] for r in log])
    x0, x1 = x.min(), x.max()
    w = x >= x0 + 0.5 * (x1 - x0)
    dep = np.array([r["depth"] for r in log])
    ab = np.array([r.get("ray_absorbed", np.nan) for r in log])
    ft = np.array([r.get("front_tan", np.nan) for r in log])
    last = d["log"][-1]
    pores = last.get("bubbles", [])
    from tpmsfem.process import defects as DF
    import ti64_material as TM
    m = TM.GAN2021
    al = 33.4 / (m["rho_liquid"] * 830.0)
    eta = float(DF.keyhole_absorptivity(d["P"], d["v"], d["r0"], m["eta_min"], m["rho_liquid"], 830.0, 1933.0, 303.0))
    Ke = float(DF.keyhole_number(d["P"], d["v"], eta, d["r0"], m["rho_liquid"], 830.0, 1933.0, 303.0, al))
    return dict(case=d["case"], dx_um=d["dx"] * 1e6, P=d["P"], v=d["v"], finished=d.get("finished", False),
                Ke=Ke, eta_eq6=eta, measured_depth_um=None if d["measured_depth"] is None else 1e6 * d["measured_depth"],
                measured_eta=d["measured_eta"], gan_depth_um=1e6 * float(DF.keyhole_depth(Ke, d["r0"])),
                depth_um=1e6 * float(dep[w].mean()), depth_min_um=1e6 * float(dep[w].min()),
                depth_max_um=1e6 * float(dep[w].max()), absorbed=float(np.nanmean(ab[w])),
                front_tan=float(np.nanmean(ft[w])), aspect=float(dep[w].mean() / d["r0"]),
                pores=len(pores), pore_d_um=[(6.0 * p["volume_um3"] / np.pi) ** (1 / 3) for p in pores],
                formed=last.get("bubbles_formed", 0), released=last.get("bubbles_released", 0),
                limited=last.get("limited", 0), recoil_clipped=last.get("recoil_clipped", 0),
                mass_error=d.get("mass_error"), energy_balance=d.get("energy_balance"),
                energy_residual=d.get("energy_residual"), mlups=d.get("mlups"),
                t=[r["t"] for r in log], depth_series=[r["depth"] for r in log])


def figure(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    runs = [s for s in (summarise(p) for p in sorted(glob.glob(os.path.join(folder, "*_dx*um.json")))) if s]
    json.dump([{k: v for k, v in r.items() if k not in ("t", "depth_series")} for r in runs],
              open(os.path.join(folder, "keyhole_summary.json"), "w"), indent=1)

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    fig.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.13, wspace=0.3)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(runs), 1)))
    ax = axs[0]
    for r, c in zip(runs, cols):
        ax.plot(1e6 * np.array(r["t"]), 1e6 * np.array(r["depth_series"]), color=c, lw=1.1,
                label=f"{r['case']} {r['P']:.0f} W {r['v']:.1f} m/s, {r['dx_um']:g} µm")
        if r["measured_depth_um"]:
            ax.axhline(r["measured_depth_um"], color=c, lw=0.8, ls="--")
    ax.set_xlabel("time [µs]", fontsize=8.5)
    ax.set_ylabel("keyhole depth [µm]", fontsize=8.5)
    ax.set_title("(a) keyhole depth (dashed: X-ray)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.5, frameon=False)
    style(ax)
    ax = axs[1]
    hi = max([r["depth_max_um"] for r in runs] + [r["measured_depth_um"] or 0 for r in runs] + [100]) * 1.1
    ax.plot([0, hi], [0, hi], color=INK, lw=0.9)
    for r, c in zip(runs, cols):
        if r["measured_depth_um"]:
            ax.errorbar([r["measured_depth_um"]], [r["depth_um"]],
                        yerr=[[r["depth_um"] - r["depth_min_um"]], [r["depth_max_um"] - r["depth_um"]]],
                        fmt="o", color=c, ms=5, capsize=3)
            ax.plot([r["measured_depth_um"]], [r["gan_depth_um"]], "x", color=MUTED, ms=5)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("measured depth [µm] (X-ray)", fontsize=8.5)
    ax.set_ylabel("simulated depth [µm] (second half of the track)", fontsize=8.5)
    ax.set_title("(b) depth: simulation (o, max/min) and Gan's law (x)", fontsize=9.5, loc="left")
    style(ax)
    ax = axs[2]
    kl = np.linspace(0, 6, 100)
    ax.plot(kl, 0.7 * (1 - np.exp(-0.6 * kl)), color=MUTED, lw=1.0, label="Gan et al. Eq. 6")
    from tpmsfem.process import defects as DF
    for r, c in zip(runs, cols):
        KeL = DF.keyhole_porosity_number(r["P"], r["v"], 47.5e-6, 0.26, 3920.0, 830.0, 1933.0, 303.0)
        ax.plot([KeL], [r["absorbed"]], "o", color=c, ms=5)
    ax.set_xlabel("Ke_m L_d*", fontsize=8.5)
    ax.set_ylabel("absorbed fraction (ray tracing)", fontsize=8.5)
    ax.set_title("(c) absorptivity", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    ax = axs[3]
    es = np.linspace(0, 9, 50)
    ax.plot(es, 0.77 * es + 0.49, color=MUTED, lw=1.0, label="X-ray fit (Gan data): 0.77 e* + 0.49")
    for r, c in zip(runs, cols):
        ax.plot([r["aspect"]], [r["front_tan"]], "o", color=c, ms=5)
    ax.set_xlabel("keyhole aspect ratio e* (simulated)", fontsize=8.5)
    ax.set_ylabel("tan(front-wall angle)", fontsize=8.5)
    ax.set_title("(d) front wall", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.suptitle("Keyhole stage 2: Ti-6Al-4V single tracks (free-surface LBM, ray-traced Fresnel laser, bubble "
                 "pressures) against operando X-ray keyholes (Cunningham et al. 2019, via Gan et al. 2021)",
                 fontsize=10.5, x=0.02, ha="left")
    out = os.path.join(HERE, "..", "..", "docs", "img", "keyhole_ti64.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
    for r in runs:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("t", "depth_series")})


if __name__ == "__main__":
    figure(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "keyhole_runs"))
