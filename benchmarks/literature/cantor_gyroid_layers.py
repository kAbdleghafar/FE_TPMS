"""Cell structure and Mn+Ni segregation in the first, middle and final layers of the LPBF
CoCrFeMnNi gyroid column (phase field at each layer's fusion-boundary conditions).

- First layer: on the plate at 80 C. Conditions of the single track on a solid plate
  (cantor_gyroid_solidification.py, validation run). There is no powder below it.
- Middle layer (4.4 mm) and final layer (8.8 mm): the part-scale build gives the temperature
  the layer meets (cantor_gyroid_multilayer.py). G and R of the thin-wall window are
  interpolated in that temperature between the melt-pool runs at 80 C and 140 C, for wall
  regions over solid and over powder (down-facing).

    python benchmarks/literature/cantor_gyroid_layers.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
from common import ResultStore  # noqa: E402

STORE = "benchmarks/literature/results/cantor_gyroid_layers.json"
FIELDS = "benchmarks/literature/results/cantor_gyroid_layers_fields.npz"
Z_MID, Z_END = 4.4, 8.8


def conditions():
    base = json.load(open(CG.STORE))
    pre = json.load(open(CG.STORE.replace(".json", "_preheat.json")))
    ml = json.load(open("benchmarks/literature/results/cantor_gyroid_multilayer.json"))
    recs = (ml.get("build|layer_time=10") or ml["build|layer_time=10|partial"])["records"]
    z = np.array([r["z_top"] for r in recs])
    T = np.array([r["T_top_mean"] for r in recs])
    v = base["validation_meltpool"]
    out = {"first|solid": dict(z=0.03, T0=CG.T_PLATE, G=v["G"]["p50"], R=v["R"]["p50"])}
    w80, w140 = base["window|thin"], pre["window|thin|z9.3"]
    for name, zl in (("middle", Z_MID), ("final", Z_END)):
        T0 = float(np.interp(zl, z, T))
        f = (T0 - CG.T_PLATE) / (w140["T0"] - CG.T_PLATE)
        for c, lab in (("supported", "solid"), ("down-facing", "powder")):
            out[f"{name}|{lab}"] = dict(z=zl, T0=T0, G=(1 - f) * w80[c]["G"]["p50"] + f * w140[c]["G"]["p50"],
                                        R=(1 - f) * w80[c]["R"]["p50"] + f * w140[c]["R"]["p50"])
    return out


def main():
    store = ResultStore(STORE)
    for key, c in conditions().items():
        if store.get(key) is None:
            out, r = CG.run_pf(c["G"], c["R"], 60.0)
            out.update(z=c["z"], T0=c["T0"])
            fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
            top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
            lo = max(top - int(1.4e-6 / r.dx_m), 0)
            fields[key + "_c"] = r.c[lo:top + 10].astype(np.float16)
            fields[key + "_phi"] = r.phi[lo:top + 10].astype(np.float16)
            fields[key + "_dx"] = np.array(r.dx_m)
            np.savez_compressed(FIELDS, **fields)
            store.put(key, out)
        o = store.get(key)
        print(f"{key:14s} z = {o['z']:.2f} mm, T0 = {o['T0']:.0f} C, G = {o['G']:.2e} K/m, R = {o['R']:.3f} m/s: "
              f"spacing {o['spacing_um']:.2f} um, cores {o['cs_rel']:.3f}, max between cells {o['intercell_max_rel']:.2f}",
              flush=True)


FIG = "docs/img/cantor_gyroid_layers.png"


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = ResultStore(STORE)
    F = dict(np.load(FIELDS))
    cols = [("first", "first layer (on the plate)"), ("middle", f"middle layer ({Z_MID} mm)"), ("final", f"final layer ({Z_END} mm)")]
    rows = [("solid", "over solid"), ("powder", "over powder (down-facing)")]
    BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.15], hspace=0.5, wspace=0.16, left=0.06, right=0.93, top=0.9,
                          bottom=0.08)
    im = None
    n_rows = min(F[k].shape[0] for k in F if k.endswith("_c"))          # same height for every map
    for j, (lay, ltitle) in enumerate(cols):
        axp = fig.add_subplot(gs[2, j])
        for i, (sub, stitle) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            key = f"{lay}|{sub}"
            if key + "_c" not in F:
                ax.text(0.5, 0.5, "not applicable:\nthe first layer lies on the solid plate" if lay == "first"
                        else "run pending", ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
                ax.set_axis_off()
                continue
            c = F[key + "_c"].astype(float)[-n_rows:] * 40.0
            dx = float(F[key + "_dx"]) * 1e6
            o = store.get(key)
            im = ax.imshow(c, origin="lower", cmap="magma", vmin=34, vmax=56, extent=[0, c.shape[1] * dx, 0, c.shape[0] * dx])
            ax.contour((np.arange(c.shape[1]) + 0.5) * dx, (np.arange(c.shape[0]) + 0.5) * dx, F[key + "_phi"].astype(float)[-n_rows:],
                       [0.0], colors="w", linewidths=0.5)
            ax.set_title(f"{ltitle if i == 0 else ''}\n{stitle}: new layer meets {o['T0']:.0f} C, cells {o['spacing_um']:.2f} um apart",
                         fontsize=8.5, loc="left")
            ax.tick_params(labelsize=7.5, colors=MUTED)
            if j == 0:
                ax.set_ylabel("growth direction [um]", fontsize=8.5)
            # profile across the cells, 0.6 um below the top of the map
            row = c[max(c.shape[0] - int(0.75 / dx), 0)]
            axp.plot((np.arange(len(row)) + 0.5) * dx, row, color=BLUE if sub == "solid" else ORANGE, lw=1.2, label=stitle)
        axp.axhline(40.0, color=MUTED, lw=0.8, ls=":")
        axp.set_ylim(34, 58)
        axp.set_xlabel("across the cells [um]", fontsize=8.5)
        axp.set_title(f"Mn+Ni across the cells, {lay} layer (dotted: nominal 40 wt%)", fontsize=8.5, loc="left")
        axp.tick_params(labelsize=7.5, colors=MUTED)
        axp.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            axp.spines[side].set_visible(False)
        if j == 0:
            axp.set_ylabel("Mn + Ni [wt%]", fontsize=8.5)
            axp.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    cax = fig.add_axes([0.945, 0.42, 0.01, 0.46])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mn + Ni [wt%] (pseudo-binary)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Cells and Mn+Ni segregation in the first, middle and final layers of the LPBF CoCrFeMnNi gyroid "
                 "(2-D phase field; white line: solid-liquid front)", fontsize=10.5, x=0.06, ha="left")
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)


if __name__ == "__main__":
    if "--figure" in sys.argv:
        figure()
    else:
        main()
