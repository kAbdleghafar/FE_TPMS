"""Level 1a of the FCC -> HCP question: how much HCP forms in a compressed gyroid printed from a
metastable FeMnCoCr high-entropy alloy (the equiatomic CoCrFeMnNi twins instead, Level 0).

1. Lattice and loading. The graded gyroid column of cantor_gyroid_faults.py (4 x 4 x 8 mm, walls
   0.2 -> 0.6 mm, 0.1 mm voxels), compressed along the build direction between bonded platens to
   5 % nominal strain (small-strain J2; geometric nonlinearity not included).
2. Material. LPBF Fe50Mn30Co10Cr10 at room temperature (Micromachines 15 (2024) 123, Table 1):
   yield 487.6 MPa; linear hardening 1300 MPa from the tensile strength (744.9 MPa) reached near
   20 % strain (an estimate); E = 200 GPa and nu = 0.28 assumed. The measured flow curve already
   contains the TRIP hardening.
3. HCP kinetics (tpmsfem.post.hcp_kinetics). Olson-Cohen type law in the element's equivalent
   plastic strain, fitted to the same study's EBSD fractions in tension (0.94 % as built, 28.1 %
   at 20 % strain); two shapes (exponential, sigmoidal with n = 2) bound the unconstrained
   small-strain part. One-way coupling: no transformation strain, no stress-state dependence.

    python benchmarks/literature/femncocr_gyroid_hcp.py run      # ~10 min
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_faults as GF  # noqa: E402
import cantor_gyroid_solidification as CG  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.materials import Material  # noqa: E402
from tpmsfem.post import hcp_kinetics as HK  # noqa: E402

RES = GF.RES
STORE = os.path.join(RES, "femncocr_gyroid_hcp.json")
FIELDS = os.path.join(RES, "femncocr_gyroid_hcp.npz")
FIG = GF.FIG.replace("cantor_gyroid_faults.png", "femncocr_gyroid_hcp.png")
STRAINS = [0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.02, 0.03, 0.04, 0.05]
MAPS = (0.03, 0.05)
E_MPA, NU, HARD = 200e3, 0.28, 1300.0


def material():
    d = HK.FEMNCOCR_LPBF
    return Material("Fe50Mn30Co10Cr10_LPBF", E=E_MPA, nu=NU, yield_stress=d["yield_MPa"], hardening=HARD)


def run():
    t0 = time.time()
    mesh = GF.build_mesh()
    md = tf.SolidModel(mesh, material())
    z = mesh.points[:, 2]
    bot, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > z.max() - 1e-9)
    md.fix(bot)
    md.fix(top, (0, 1), 0.0)
    laws = {s: HK.femncocr_lpbf(s, E_MPA) for s in ("exponential", "sigmoidal")}
    rows, maps = [], {}
    for eps in STRAINS:
        md.fix(top, (2,), -eps * GF.HEIGHT)
        md.solve(n_inc=2, tol=1e-6)
        peeq = md.plastic_strain()[1].mean(axis=1)
        F = abs(md.reactions()[top, 2].sum())
        row = dict(strain=eps, nominal_stress_MPa=float(F / CG.L_CELL ** 2), yielded=float((peeq > 1e-6).mean()),
                   peeq_mean=float(peeq.mean()), peeq_p95=float(np.percentile(peeq, 95)), peeq_max=float(peeq.max()))
        for s, law in laws.items():
            f = law.fraction(peeq)
            row[f"hcp_mean_{s}"] = float(f.mean())
            row[f"hcp_p95_{s}"] = float(np.percentile(f, 95))
            row[f"hcp_max_{s}"] = float(f.max())
            row[f"hcp_above10_{s}"] = float((f >= 0.10).mean())
        rows.append(row)
        print(f"eps {eps * 100:4.1f} %: nominal {row['nominal_stress_MPa']:6.1f} MPa, yielded {100 * row['yielded']:5.1f} %, "
              f"PEEQ p95 {100 * row['peeq_p95']:5.2f} %, HCP mean {100 * row['hcp_mean_exponential']:5.2f} / "
              f"{100 * row['hcp_mean_sigmoidal']:5.2f} %, p95 {100 * row['hcp_p95_exponential']:5.2f} / "
              f"{100 * row['hcp_p95_sigmoidal']:5.2f} % (exp / sigm), {time.time() - t0:.0f} s", flush=True)
        if any(abs(eps - m) < 1e-9 for m in MAPS):
            k = f"{eps * 100:.0f}"
            maps[f"peeq_{k}"] = peeq.astype(np.float32)
    out = dict(material=dict(E_MPa=E_MPA, nu=NU, yield_MPa=HK.FEMNCOCR_LPBF["yield_MPa"], hardening_MPa=HARD),
               calibration=HK.FEMNCOCR_LPBF,
               laws={s: dict(f0=l.f0, beta=l.beta, alpha=l.alpha, n=l.n, shape=l.shape) for s, l in laws.items()},
               steps=rows, seconds=time.time() - t0)
    json.dump(out, open(STORE, "w"), indent=1)
    np.savez_compressed(FIELDS, ijk=mesh.cell_data["ijk"].astype(np.int16), **maps)
    print("wrote", STORE, FIELDS, flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    s = json.load(open(STORE))
    d = np.load(FIELDS)
    laws = {k: HK.StrainInducedHCP(**v) for k, v in s["laws"].items()}
    rows = s["steps"]

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.62, 0.62, 0.62], wspace=0.38, left=0.05, right=0.98, top=0.8,
                          bottom=0.12)
    ax = fig.add_subplot(gs[0])
    e = np.linspace(0, 0.25, 300)
    for k, col in (("exponential", BLUE), ("sigmoidal", ORANGE)):
        ax.plot(100 * e, 100 * laws[k].fraction(e), color=col, lw=1.6, label=k)
    c = s["calibration"]
    ax.plot([0, 100 * (c["strain"] - c["uts_MPa"] / s["material"]["E_MPa"])], [100 * c["f0"], 100 * c["f"]], "ko", ms=6,
            label="measured (EBSD, tension)")
    pmax = max(r["peeq_max"] for r in rows)
    ax.axvspan(0, 100 * pmax, color="#d6d5d0", alpha=0.5, lw=0)
    ax.text(100 * pmax / 2, 30, "plastic strains\nin the gyroid", fontsize=7, ha="center", color=MUTED)
    ax.set_xlabel("equivalent plastic strain [%]", fontsize=8.5)
    ax.set_ylabel("HCP fraction [%]", fontsize=8.5)
    ax.set_title("(a) HCP kinetics, LPBF Fe50Mn30Co10Cr10", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(ax)
    ax = fig.add_subplot(gs[1])
    eps = [100 * r["strain"] for r in rows]
    for k, col in (("exponential", BLUE), ("sigmoidal", ORANGE)):
        ax.plot(eps, [100 * r[f"hcp_mean_{k}"] for r in rows], "o-", color=col, lw=1.6, ms=4, label=f"lattice mean, {k}")
        ax.plot(eps, [100 * r[f"hcp_p95_{k}"] for r in rows], "--", color=col, lw=1.1, label=f"95th percentile, {k}")
    ax.axhline(100 * c["f0"], color=MUTED, lw=0.8, ls=":")
    ax.text(0.2, 100 * c["f0"] + 0.3, "as built", fontsize=7, color=MUTED)
    ax.set_xlabel("nominal compressive strain [%]", fontsize=8.5)
    ax.set_ylabel("HCP fraction [%]", fontsize=8.5)
    ax.set_title("(b) HCP in the compressed gyroid", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    ijk = d["ijk"].astype(int)
    jmid = (ijk[:, 1].max() + 1) // 2

    def section(v):
        sel = ijk[:, 1] == jmid
        img = np.full((ijk[:, 0].max() + 1, ijk[:, 2].max() + 1), np.nan)
        img[ijk[sel, 0], ijk[sel, 2]] = v[sel]
        return img.T

    ext = [0, CG.L_CELL, 0, GF.HEIGHT]
    pe = d["peeq_5"].astype(float)
    panels = (("(c) HCP, exponential, 5 %", 100 * laws["exponential"].fraction(pe), "magma", 0, 15),
              ("(d) HCP, sigmoidal, 5 %", 100 * laws["sigmoidal"].fraction(pe), "magma", 0, 15),
              ("(e) plastic strain, 5 %", 100 * pe, "viridis", 0, np.percentile(100 * pe[pe > 0], 99)))
    for k, (title, v, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[2 + k])
        im = ax.imshow(section(v), origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("z [mm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
        cb.set_label("[%]", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Level 1a: strain-induced HCP in a gyroid of the metastable Fe50Mn30Co10Cr10 HEA under compression "
                 "(graded gyroid column, LPBF yield 488 MPa, J2 plasticity; Olson-Cohen law fitted to LPBF EBSD fractions)\n"
                 "The equiatomic CoCrFeMnNi of the study twins instead (Level 0): its HCP fraction stays zero at room temperature",
                 fontsize=10.5, x=0.02, ha="left")
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    figure()
