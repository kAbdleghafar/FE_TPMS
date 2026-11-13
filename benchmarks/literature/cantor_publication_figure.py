"""Figure for cantor_publication_checks.py."""
import json
import os

import numpy as np

import cantor_material as CM

FIG = "docs/img/cantor_publication_checks.png"
BLUE, ORANGE, AQUA, INK, MUTED, GRID, BAND = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#d6d5d0"
STUDY = "benchmarks/literature/results/cantor_gyroid.json"


def _style(ax, title):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title(title, fontsize=9, loc="left", color=INK)


def _measured(ax, horizontal=False):
    lo, hi = CM.PIGLIONE2018["cell_spacing_um"]
    (ax.axhspan if horizontal else ax.axvspan)(lo, hi, color=BAND, lw=0, zorder=0)


def _dotrows(ax, rows, xlabel):
    """rows: (label, value, hunt or None, colour)."""
    rows = [r for r in rows if r[1] is not None]
    for i, (lab, v, hunt, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(v, y, "o", color=col, ms=7, zorder=3)
        if hunt is not None:
            ax.plot(hunt, y, "D", mfc="white", mec=MUTED, ms=5, mew=1.2, zorder=2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)


def figure(store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    g = store.get
    study = json.load(open(STUDY)) if os.path.exists(STUDY) else {}
    sp = lambda k: (g(k) or {}).get("spacing_um")             # noqa: E731
    hu = lambda k: (g(k) or {}).get("hunt_um")                # noqa: E731
    fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.4))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.08, hspace=0.42, wspace=0.62)

    # (a) validation: scan mode and absorptivity
    ax = axs[0, 0]
    _measured(ax)
    base = study.get("validation_pf|dT0=60", {})
    rows = [("continuous 0.75 m/s, A 0.35\n(the study)", base.get("spacing_um"), base.get("hunt_um"), INK)]
    for dl, v in ((1, 0.74), (20, 0.60), (50, 0.46)):
        rows.append((f"points, {dl} us delay\n(mean {v:.2f} m/s)", sp(f"pf|pulsed|delay={dl}"), hu(f"pf|pulsed|delay={dl}"), BLUE))
    for v in (600, 462):
        rows.append((f"continuous {v / 1e3:.2f} m/s", sp(f"pf|continuous|v={v}"), hu(f"pf|continuous|v={v}"), AQUA))
    for a in (0.30, 0.45, 0.60):
        rows.append((f"continuous 0.75 m/s, A {a:.2f}", sp(f"pf|absorb|A={a:.2f}"), hu(f"pf|absorb|A={a:.2f}"), ORANGE))
    _dotrows(ax, rows, "primary cell spacing [um]")
    ax.set_xlim(0.3, 1.0)
    _style(ax, "(a) validation track: scan mode and absorptivity")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="phase field"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=MUTED, label="Hunt"),
                       plt.Rectangle((0, 0), 1, 1, color=BAND, label="measured 0.52-0.64 um")],
              fontsize=7, loc="lower right", frameon=False)

    # (b) stability band
    ax = axs[0, 1]
    _measured(ax, horizontal=True)
    ax.plot([0, 2.2], [0, 2.2], color=MUTED, lw=0.8, ls="--")
    ax.text(1.75, 1.88, "kept", fontsize=7.5, color=MUTED, rotation=35)
    for name, col, lab in (("supported", BLUE, "over solid (G 3.0e7 K/m, R 0.027 m/s)"),
                           ("down", ORANGE, "over powder (G 2.5e7 K/m, R 0.010 m/s)")):
        pts = sorted((o["lam0_um"], o["spacing_um"]) for k, o in store.data.items() if k.startswith(f"pf|band|{name}|"))
        if pts:
            x, y = np.array(pts).T
            ax.plot(x, y, "o-", color=col, lw=2, ms=7, label=lab)
        o = g(f"pf|num|{name}|base")
        if o:
            ax.plot(0.1, o["spacing_um"], ">", color=col, ms=8)
    ax.text(0.14, 0.12, "from noise", fontsize=7, color=MUTED, rotation=90, va="bottom")
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("seeded spacing [um]", fontsize=8.5)
    ax.set_ylabel("spacing after 10 um of growth [um]", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _style(ax, "(b) which spacings the phase field sustains")

    # (c) numerics
    ax = axs[0, 2]
    labels = [("base", "base: 6 um window,\nW0 10 nm, 10 um growth"), ("seed=1", "other random seed"),
              ("wide", "12 um window,\n15-20 um growth"), ("W0=7", "W0 7 nm"), ("tall", "3 um tall window")]
    for j, (name, col) in enumerate((("supported", BLUE), ("down", ORANGE))):
        for i, (k, _) in enumerate(labels):
            o = g(f"pf|num|{name}|{k}")
            if o:
                ax.plot(o["spacing_um"], len(labels) - 1 - i + (0.12 if j == 0 else -0.12), "o", color=col, ms=7)
    ax.set_yticks(range(len(labels)), [lab for _, lab in labels][::-1], fontsize=7.5)
    _measured(ax)
    ax.set_xlim(0.3, 1.2)
    ax.set_xlabel("primary cell spacing [um]", fontsize=8.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="over solid"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, label="over powder")],
              fontsize=7, loc="lower right", frameon=False)
    _style(ax, "(c) phase-field numerics")

    # (d) down-facing contrast vs absorptivity (thin-wall window)
    ax = axs[1, 0]
    A = [0.30, 0.35, 0.50]
    for c, col, lab in (("supported", BLUE, "over solid"), ("down-facing", ORANGE, "over powder")):
        ys = [sp(f"pf|window|A=0.30|{c}"), study.get(f"pf|thin|{c}", {}).get("spacing_um"), sp(f"pf|window|A=0.50|{c}")]
        xa = [a for a, y in zip(A, ys) if y is not None]
        ya = [y for y in ys if y is not None]
        ax.plot(xa, ya, "o-", color=col, lw=2, ms=7, label=lab)
    for a, key in ((0.30, "pf|window|A=0.30|"), (0.35, None), (0.50, "pf|window|A=0.50|")):
        s = study.get("pf|thin|supported", {}).get("spacing_um") if key is None else sp(key + "supported")
        d = study.get("pf|thin|down-facing", {}).get("spacing_um") if key is None else sp(key + "down-facing")
        if s and d:
            ax.text(a, d + 0.06, f"{d / s:.2f}x", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(A, ["0.30", "0.35\n(study)", "0.50"])
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("absorptivity", fontsize=8.5)
    ax.set_ylabel("primary cell spacing [um]", fontsize=8.5)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    _style(ax, "(d) thin-wall window: down-facing contrast")

    # (e) pseudo-binary parameters
    ax = axs[1, 1]
    _measured(ax)
    rows = [("base: D 3e-9 m2/s, k 0.74,\neps4 0.02", sp("pf|num|supported|base"), hu("pf|num|supported|base"), INK),
            ("D 6e-9 m2/s", sp("pf|mat|D=6e-9"), hu("pf|mat|D=6e-9"), BLUE),
            ("k 0.81 (Mn)", sp("pf|mat|k=0.81"), hu("pf|mat|k=0.81"), BLUE),
            ("k 0.88 (walls counted at\n0.05 c_inf contrast)", sp("pf|mat|k=0.88|prom=0.05"), hu("pf|mat|k=0.88|prom=0.05"), BLUE),
            ("eps4 0.01", sp("pf|mat|eps4=0.01"), hu("pf|mat|eps4=0.01"), BLUE),
            ("eps4 0.04", sp("pf|mat|eps4=0.04"), hu("pf|mat|eps4=0.04"), BLUE)]
    _dotrows(ax, rows, "primary cell spacing [um]")
    ax.set_xlim(0.3, 1.0)
    _style(ax, "(e) pseudo-binary parameters (over solid)")

    # (f) residual strain: voxel and lump size, indexed to the study's resolution
    ax = axs[1, 2]
    cases = [("vox0.1_lump0.4", INK, "0.1 mm voxels, 0.4 mm lumps (study)"),
             ("vox0.1_lump0.2", BLUE, "0.1 mm voxels, 0.2 mm lumps"),
             ("vox0.05_lump0.4", ORANGE, "0.05 mm voxels, 0.4 mm lumps")]
    ref = g("residual|vox0.1_lump0.4")
    metrics = [("mean |eps_zz|", lambda o: o["removed"]["abs_ezz_mean"]),
               ("eps_zz 95th pct.", lambda o: o["removed"]["ezz_p5_p95"][1]),
               ("eps_zz 5th pct.", lambda o: o["removed"]["ezz_p5_p95"][0]),
               ("von Mises median", lambda o: o["removed"]["vm_p50"]),
               ("plastic strain median", lambda o: o["peeq_percent_p50_p95"][0])]
    if ref:
        for j, (c, col, lab) in enumerate(cases):
            o = g(f"residual|{c}")
            if not o:
                continue
            for i, (_, f) in enumerate(metrics):
                ax.plot(f(o) / f(ref), len(metrics) - 1 - i + 0.15 * (1 - j), "o", color=col, ms=7,
                        label=lab if i == 0 else None)
        ax.axvline(1.0, color=MUTED, lw=0.8)
        ax.legend(fontsize=7, loc="lower left", frameon=False, bbox_to_anchor=(0, -0.42))
    ax.set_yticks(range(len(metrics)), [m[0] for m in metrics][::-1], fontsize=7.5)
    ax.set_xlabel("relative to the study's resolution (2 mm column, plate removed)", fontsize=8)
    _style(ax, "(f) residual strain: voxel and lump size")

    fig.suptitle("CoCrFeMnNi gyroid: checks before publication (cell spacing at dT0 = 60 K; grey band: measured, "
                 "Piglione et al. 2018)", fontsize=11, x=0.02, ha="left")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)
