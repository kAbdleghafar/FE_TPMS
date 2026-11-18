"""Level 0 of tracking a stress-induced FCC -> HCP transformation in a compressed CoCrFeMnNi
gyroid: where, and at what applied strain, can the local stress extend stacking faults (the
common precursor of deformation twins and of HCP martensite)?

1. Lattice. The graded gyroid column of the Cantor study (cantor_gyroid_solidification.py: 4 mm
   cell, sheet walls 0.2 -> 0.6 mm along x; 4 x 4 x 8 mm, two cells tall), 0.1 mm voxels, without
   the build plate and without residual stresses (an as-designed, stress-free lattice).
2. Loading. Compression along the build direction z between bonded platens: the bottom face is
   fixed, the top face moves down with no lateral motion; nominal strains up to 3 %.
3. Material. As-built LPBF CoCrFeMnNi at room temperature: E = 203 GPa (Laplanche et al. 2015),
   nu = 0.25, compressive yield 757 MPa (Kim et al. 2020), linear hardening 1500 MPa (assumed,
   as in the residual-stress study); J2 plasticity, small strain.
4. Criterion (tpmsfem.post.fcc_faults). In every element and for every grain orientation the
   largest shear stress resolved on the 12 {111}<112> twinning / transformation systems,
   tau* (the stress that drives the leading Shockley partial). A fault extends where
   tau* >= tau_c. The grain stress is the element stress (Sachs-type: grain-to-grain stress
   differences are not resolved). Nucleation also needs dislocation sources, so only elements
   that have yielded count.
5. Threshold. tau_c = 360 MPa: the onset in the best-oriented grains (Schmid factor 0.5) of a
   random polycrystal in uniaxial tension at the measured twinning stress of CoCrFeMnNi, 720 MPa at
   77 and 293 K (Laplanche et al., Acta Mater. 118 (2016) 152), with the same Sachs assumption.
   Lower thresholds stand for lower effective fault energies, in proportion (tau_c ~ gamma / b_p):
   cryogenic temperature, Mn/Fe-rich variants, or a thick HCP embryo (gamma_SF - 2 sigma).
   The strict free-partials bound (Byun 2003) with gamma = 30 mJ/m^2 is reported as well.
6. Orientations. Volume-weighted samples of the CA textures of the top layers of the thin- and
   thick-wall windows (cantor_gyroid_ca.py), used for x < 2 mm and x >= 2 mm; a random texture
   for comparison.

    python benchmarks/literature/cantor_gyroid_faults.py run
    python benchmarks/literature/cantor_gyroid_faults.py figure
    FAULTS_VOX=0.05 python benchmarks/literature/cantor_gyroid_faults.py run    # resolution check
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
import cantor_material as CM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.materials import Material  # noqa: E402
from tpmsfem.mesh.voxel import keep_connected, mask_to_mesh  # noqa: E402
from tpmsfem.post import fcc_faults as FF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
STORE = os.path.join(RES, "cantor_gyroid_faults.json")
CHECK = os.path.join(RES, "cantor_gyroid_faults_vox{vox}.json")
FIELDS = os.path.join(RES, "cantor_gyroid_faults.npz")
FIG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "img", "cantor_gyroid_faults.png"))
CA = {w: os.path.join(RES, f"cantor_gyroid_ca_{w}.npz") for w in ("thin", "thick")}

VOX = float(os.environ.get("FAULTS_VOX", "0.1"))          # 0.05: resolution check (json only)
HEIGHT = 8.0
STRAINS = [0.002, 0.004, 0.006, 0.008, 0.010, 0.0125, 0.015, 0.02, 0.025, 0.03]
MAP_STRAINS = (0.006, 0.01, 0.03)
TAU_C = (360.0, 300.0, 240.0, 180.0, 120.0)       # MPa; 360: CoCrFeMnNi twinning onset (calibrated)
N_ORI = 600
GAMMA_BYUN = 30.0                                  # mJ/m^2, measured stacking-fault energy (Okamoto 2016)
LAPLANCHE_SIGMA = 720.0                            # MPa


def material():
    return Material("CoCrFeMnNi_LPBF_RT", E=float(CM.youngs_modulus_MPa(20.0)), nu=0.25,
                    yield_stress=CM.KIM2020["yield_MPa"][0], hardening=1500.0)


def build_mesh():
    n_xy = int(round(CG.L_CELL / VOX))
    nz = int(round(HEIGHT / VOX))
    c = (np.arange(n_xy) + 0.5) * VOX
    zc = (np.arange(nz) + 0.5) * VOX
    X, Y, Z = np.meshgrid(c, c, zc, indexing="ij")
    mask = keep_connected(np.asarray(CG.gyroid()(X, Y, Z)) < 0.0)
    return mask_to_mesh(mask, (0.0, 0.0, 0.0), VOX)


def orientations(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for w, f in CA.items():
        rot = np.load(f)["top_rot"].astype(float)
        out[w] = rot[rng.choice(len(rot), N_ORI, replace=False)]
    out["random"] = FF.random_rotations(N_ORI, seed=seed + 1)
    return out


def run():
    t0 = time.time()
    mesh = build_mesh()
    md = tf.SolidModel(mesh, material())
    z = mesh.points[:, 2]
    bot = np.flatnonzero(z < 1e-9)
    top = np.flatnonzero(z > z.max() - 1e-9)
    md.fix(bot)
    md.fix(top, (0, 1), 0.0)
    cen = mesh.centroids()
    vol = np.full(len(cen), VOX ** 3)
    ori = orientations()
    thick = cen[:, 0] >= 0.5 * CG.L_CELL
    print(f"mesh: {mesh.n_cells} voxels, relative density {mesh.n_cells * VOX ** 3 / (CG.L_CELL ** 2 * HEIGHT):.3f}", flush=True)
    # orientation-only reference: uniaxial compression along z
    ref = {w: FF.uniaxial_factor(r, (0, 0, 1), -1) for w, r in ori.items()}
    rows, maps = [], {}
    for eps in STRAINS:
        md.fix(top, (2,), -eps * HEIGHT)
        info = md.solve(n_inc=2, tol=1e-6)
        sig = md.stress().mean(axis=1)                               # (ne, 6)
        _, alpha = md.plastic_strain()
        peeq = alpha.mean(axis=1)
        plastic = peeq > 1e-6
        F = -md.reactions()[top, 2].sum()
        vm = np.sqrt(0.5 * ((sig[:, 0] - sig[:, 1]) ** 2 + (sig[:, 1] - sig[:, 2]) ** 2 + (sig[:, 2] - sig[:, 0]) ** 2)
                     + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))
        row = dict(strain=eps, nominal_stress_MPa=float(F / CG.L_CELL ** 2), plastic_volume_frac=float(plastic.mean()),
                   vm_p50=float(np.median(vm)), vm_p99=float(np.percentile(vm, 99)), converged=bool(getattr(info, "converged", True)))
        # tau* for the CA textures (thin / thick by x) and the random texture
        tau_ca = np.empty((len(cen), N_ORI))
        for w, sel in (("thin", ~thick), ("thick", thick)):
            tau_ca[sel] = FF.leading_partial_stress(sig[sel], ori[w])
        tau_rd = FF.leading_partial_stress(sig, ori["random"])
        for name, tau in (("ca", tau_ca), ("random", tau_rd)):
            for tc in TAU_C:
                f_el = (tau >= tc).mean(axis=1) * plastic                  # orientation fraction per element
                row[f"frac_{name}_{tc:.0f}"] = float((f_el * vol).sum() / vol.sum())
                row[f"elems_{name}_{tc:.0f}"] = float((f_el > 0).mean())
        g_byun = np.empty((len(cen), N_ORI))
        for w, sel in (("thin", ~thick), ("thick", thick)):
            g_byun[sel] = FF.fault_energy_capacity(sig[sel], ori[w], criterion="byun")
        row["frac_byun_ca_30"] = float((((g_byun >= GAMMA_BYUN).mean(axis=1) * plastic) * vol).sum() / vol.sum())
        row["byun_gamma_max"] = float(g_byun.max())
        # thin vs thick halves
        for half, sel in (("thin", ~thick), ("thick", thick)):
            row[f"frac_ca_360_{half}"] = float(((tau_ca[sel] >= 360.0).mean(axis=1) * plastic[sel]).mean())
        rows.append(row)
        print(f"eps {eps * 100:4.2f} %: nominal {row['nominal_stress_MPa']:6.1f} MPa, plastic {row['plastic_volume_frac'] * 100:5.1f} %, "
              f"faults (CA, 360) {row['frac_ca_360'] * 100:5.2f} %, (CA, 240) {row['frac_ca_240'] * 100:5.2f} %, "
              f"Byun max {row['byun_gamma_max']:.1f} mJ/m2, {time.time() - t0:.0f} s", flush=True)
        if any(abs(eps - m) < 1e-9 for m in MAP_STRAINS):
            k = f"{eps * 100:.1f}"
            maps[f"f360_{k}"] = ((tau_ca >= 360.0).mean(axis=1) * plastic).astype(np.float16)
            maps[f"f240_{k}"] = ((tau_ca >= 240.0).mean(axis=1) * plastic).astype(np.float16)
            maps[f"peeq_{k}"] = peeq.astype(np.float32)
            maps[f"vm_{k}"] = vm.astype(np.float32)
    # orientation statistics: which grains extend faults first (compression along z)
    out = dict(material=dict(E_MPa=material().E, nu=0.25, yield_MPa=material().yield_stress, hardening_MPa=1500.0),
               tau_c_MPa=list(TAU_C), tau_c_note="360 MPa: CoCrFeMnNi twinning onset (720 MPa uniaxial, Laplanche 2016) "
               "in grains of Schmid factor 0.5", n_orientations=N_ORI, voxel_mm=VOX, n_voxels=int(mesh.n_cells),
               relative_density=float(mesh.n_cells * VOX ** 3 / (CG.L_CELL ** 2 * HEIGHT)), steps=rows,
               schmid_compression_z={w: dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)),
                                             frac_above_0_45=float((v >= 0.45).mean())) for w, v in ref.items()},
               seconds=time.time() - t0)
    if abs(VOX - 0.1) > 1e-9:                                         # resolution check: summary only
        json.dump(out, open(CHECK.format(vox=f"{VOX:g}"), "w"), indent=1)
        print("wrote", CHECK.format(vox=f"{VOX:g}"), flush=True)
        return
    json.dump(out, open(STORE, "w"), indent=1)
    np.savez_compressed(FIELDS, ijk=mesh.cell_data["ijk"].astype(np.int16), centroids=cen.astype(np.float32),
                        **{f"schmid_{w}": v.astype(np.float32) for w, v in ref.items()}, **maps)
    print("wrote", STORE, FIELDS, flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COL = {360.0: "#b42318", 300.0: "#eb6834", 240.0: "#2a78d6", 180.0: "#1baf7a", 120.0: "#7a5af8"}
    s = json.load(open(STORE))
    d = np.load(FIELDS)
    rows = s["steps"]
    eps = np.array([r["strain"] for r in rows]) * 100

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.32, left=0.05, right=0.98, top=0.88,
                          bottom=0.07)
    # (a) stress-strain with plastic fraction
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(eps, [r["nominal_stress_MPa"] for r in rows], "o-", color=INK, lw=1.6, ms=4)
    ax.set_xlabel("nominal compressive strain [%]", fontsize=8.5)
    ax.set_ylabel("nominal compressive stress [MPa]", fontsize=8.5)
    ax.set_title("(a) compression of the gyroid column", fontsize=9.5, loc="left")
    ax2 = ax.twinx()
    ax2.plot(eps, [100 * r["plastic_volume_frac"] for r in rows], "s:", color=MUTED, ms=3, lw=1)
    ax2.set_ylabel("yielded volume [%] (squares)", fontsize=8, color=MUTED)
    ax2.tick_params(labelsize=7.5, colors=MUTED)
    style(ax)
    # (b) volume fraction where faults can extend, against strain, per threshold
    ax = fig.add_subplot(gs[0, 1:3])
    for tc in s["tau_c_MPa"]:
        ax.plot(eps, [100 * r[f"frac_ca_{tc:.0f}"] for r in rows], "o-", color=COL[tc], lw=1.6, ms=4,
                label=f"tau_c = {tc:.0f} MPa" + (" (CoCrFeMnNi twinning, RT/77 K)" if tc == 360 else
                                                 f" (fault energy x {tc / 360:.2f})"))
        ax.plot(eps, [100 * r[f"frac_random_{tc:.0f}"] for r in rows], "--", color=COL[tc], lw=1.0)
    ax.plot(eps, [100 * r["frac_byun_ca_30"] for r in rows], "k:", lw=1.2, label="free-partials bound, 30 mJ/m2")
    ax.set_xlabel("nominal compressive strain [%]", fontsize=8.5)
    ax.set_ylabel("lattice volume where faults can extend [%]", fontsize=8.5)
    ax.set_title("(b) where twins / HCP could nucleate (solid: CA texture; dashed: random)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    # (c) orientation dependence: Schmid factor of the leading partial, compression along z
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0.0, 0.5, 26)
    for w, col in (("random", MUTED), ("thin", "#2a78d6"), ("thick", "#eb6834")):
        ax.hist(d[f"schmid_{w}"], bins=bins, histtype="step", color=col, lw=1.4, density=True, label=f"{w} texture" if w != "random" else "random")
    ax.axvline(360 / s["material"]["yield_MPa"], color="#b42318", lw=1, ls="--")
    ax.text(360 / s["material"]["yield_MPa"] - 0.005, ax.get_ylim()[1] * 0.92, "tau_c / yield", fontsize=7, color="#b42318",
            ha="right")
    ax.set_xlabel("leading-partial Schmid factor, compression along BD", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.set_title("(c) which grains: Schmid factor", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    # (d-g) x-z section through the column centre: fault fraction at three strains, and peeq
    ijk = d["ijk"].astype(int)
    ny = ijk[:, 1].max() + 1
    jmid = ny // 2

    def section(v):
        sel = ijk[:, 1] == jmid
        img = np.full((ijk[:, 0].max() + 1, ijk[:, 2].max() + 1), np.nan)
        img[ijk[sel, 0], ijk[sel, 2]] = v[sel]
        return img.T

    ext = [0, CG.L_CELL, 0, HEIGHT]
    panels = [(f"f360_{MAP_STRAINS[1] * 100:.1f}", f"(d) faults, tau_c = 360 MPa, {MAP_STRAINS[1] * 100:.0f} % strain", "magma", 0, 0.5),
              (f"f360_{MAP_STRAINS[2] * 100:.1f}", f"(e) faults, tau_c = 360 MPa, {MAP_STRAINS[2] * 100:.0f} % strain", "magma", 0, 0.5),
              (f"f240_{MAP_STRAINS[1] * 100:.1f}", f"(f) faults, tau_c = 240 MPa, {MAP_STRAINS[1] * 100:.0f} % strain", "magma", 0, 1.0),
              (f"peeq_{MAP_STRAINS[2] * 100:.1f}", f"(g) plastic strain, {MAP_STRAINS[2] * 100:.0f} % strain", "viridis", 0, None)]
    for k, (key, title, cmap, vmin, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        v = d[key].astype(float)
        if key.startswith("peeq"):
            v = 100 * v
            vmax = np.nanpercentile(v[v > 0], 99) if (v > 0).any() else 1.0
        im = ax.imshow(section(v), origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("x [mm] (walls thin -> thick)", fontsize=8)
        ax.set_ylabel("z [mm] (build and load direction)", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
        cb.set_label("plastic strain [%]" if key.startswith("peeq") else "fraction of grains", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Level 0: where can stacking faults (the precursor of twins and HCP) extend in a compressed CoCrFeMnNi gyroid column?\n"
        "Graded gyroid sheet (4 mm cell, walls 0.2 -> 0.6 mm along x), as-built LPBF yield 757 MPa, J2 plasticity; criterion: "
        "shear stress on the leading Shockley partial's {111}<112> system >= tau_c, in yielded elements", fontsize=10.5, x=0.02,
        ha="left")
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    if abs(VOX - 0.1) < 1e-9:
        figure()
