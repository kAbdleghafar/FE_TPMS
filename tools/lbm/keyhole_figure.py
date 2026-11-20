"""Keyhole stage 2 results against the X-ray data: reads the runs of keyhole_ti64.py in a folder
and draws the figure (docs/img/keyhole_ti64.png) and a summary (keyhole_summary.json).

Per run, the quasi-steady window is the second half of the track (the beam between 50 and 100 %
of the track length): mean, minimum and maximum keyhole depth (Gan et al.'s error bars are the
same max/min over the middle of their tracks), mean absorbed fraction, mean front-wall tangent.

    python tools/lbm/keyhole_figure.py [folder]
    python tools/lbm/keyhole_figure.py check <folder>     # one run: sections, depth, absorption
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


def centreline_depth(fill_section, z_surface, dx):
    """Depth of the open keyhole on a centreline section (x, z): the fill = 1/2 level of the
    section smoothed once in x and z, deepest point below the original surface [m]."""
    e = np.asarray(fill_section, float)
    for ax in (0, 1):
        n = e.shape[ax]
        lo = np.take(e, np.r_[0, np.arange(n - 1)], axis=ax)
        hi = np.take(e, np.r_[np.arange(1, n), n - 1], axis=ax)
        e = 0.25 * (lo + 2.0 * e + hi)
    has = e >= 0.5
    k = e.shape[1] - 1 - np.argmax(has[:, ::-1], axis=1)
    dep = np.where(has.any(axis=1), z_surface - (k + 1.0) * dx, 0.0)
    return float(dep[1:-1].max())


def check_figure(folder, tag="T3_dx5um", out=None):
    """The 5 um CPU check of one case: centreline sections and the depth and absorbed fraction
    over time (docs/img/keyhole_cpu_check.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    d = json.load(open(os.path.join(folder, tag + ".json")))
    z = np.load(os.path.join(folder, tag + "_sections.npz"))
    dx, zs = d["dx"], d["z_surface"]
    t = z["t"]
    dep = np.array([centreline_depth(z["fill"][i], zs, dx) for i in range(len(t))])
    log = d["log"]
    ab = np.array([r.get("ray_absorbed", np.nan) for r in log])
    bn = np.array([r.get("ray_mean_bounces", np.nan) for r in log])
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 0.05, 0.12, 1.15], wspace=0.3, left=0.04, right=0.955,
                          top=0.8, bottom=0.13)
    for k, tt in enumerate((40e-6, 80e-6, 120e-6)):
        i = int(np.argmin(np.abs(t - tt)))
        ax = fig.add_subplot(gs[k])
        f = z["fill"][i].astype(float).T
        T = z["T"][i].astype(float).T
        ext = [0, f.shape[1] * dx * 1e6, 0, f.shape[0] * dx * 1e6]
        im = ax.imshow(np.where(f > 0.01, T, np.nan), origin="lower", cmap="inferno", vmin=25, vmax=4500, extent=ext,
                       interpolation="nearest")
        ax.contour((np.arange(f.shape[1]) + 0.5) * dx * 1e6, (np.arange(f.shape[0]) + 0.5) * dx * 1e6, f, [0.5],
                   colors=["#4fd1e8"], linewidths=0.9)
        ax.axvline(log[i]["x_l"] * 1e6, color="white", lw=0.7, ls="--")
        ax.set_ylim(zs * 1e6 - 160, zs * 1e6 + 40)
        ax.set_title(f"({'abc'[k]}) t = {t[i] * 1e6:.0f} µs: {dep[i] * 1e6:.0f} µm deep, absorbs {ab[i]:.2f}",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("x [µm] (scan →, dashed: beam axis)", fontsize=8)
        ax.set_ylabel("z [µm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[3]))
    cb.set_label("T [°C]", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[5])
    ax.plot(t * 1e6, dep * 1e6, color=BLUE, lw=1.6, label="keyhole depth (centreline)")
    ax.set_xlabel("time [µs]", fontsize=8.5)
    ax.set_ylabel("depth [µm]", fontsize=8.5, color=BLUE)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax2 = ax.twinx()
    ax2.plot([r["t"] * 1e6 for r in log], ab, color=ORANGE, lw=1.4, label="absorbed fraction (rays)")
    if d.get("measured_eta"):
        ax2.axhline(d["measured_eta"], color=ORANGE, lw=0.8, ls=":")
        ax2.text(2, d["measured_eta"] + 0.01, f"Gan et al. (steady): {d['measured_eta']:.2f}", fontsize=7, color=ORANGE)
    ax2.set_ylim(0, 0.8)
    ax2.set_ylabel("absorbed fraction", fontsize=8.5, color=ORANGE)
    ax2.tick_params(labelsize=7.5, colors=ORANGE)
    ax.set_title(f"(d) depth and absorption ({np.nanmax(bn):.1f} bounces at the end)", fontsize=9.5, loc="left")
    for k_ in ("top",):
        ax.spines[k_].set_visible(False)
    fig.suptitle(f"Keyhole stage 2, CPU check: Ti-6Al-4V, {d['P']:.0f} W, {d['v']} m/s, r0 47.5 µm, {dx * 1e6:g} µm cells, "
                 f"{d['dt'] * 1e9:.0f} ns steps (ray-traced laser, bubbles, sub-grid viscosity). "
                 f"Mass error {d['mass_error']:.0e}, booked energy residual {d['energy_residual']:.1%}\n"
                 f"A short track (120 µs): the keyhole is still deepening; X-ray steady depth {d['measured_depth'] * 1e6:.0f} µm "
                 f"needs the 500 µm GPU tracks", fontsize=10, x=0.02, ha="left")
    out = out or os.path.join(HERE, "..", "..", "docs", "img", "keyhole_cpu_check.png")
    fig.savefig(out, dpi=140)
    print("wrote", out, "depth", [round(v * 1e6) for v in dep[9::10]])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "check":
        check_figure(sys.argv[2])
    else:
        figure(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "keyhole_runs"))
