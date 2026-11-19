"""Level 0 from the as-built state: where can stacking faults (twins, HCP) extend when the
CoCrFeMnNi gyroid column is compressed after its simulated LPBF build, residual stresses and
build plastic strains included, compared with the same column stress-free?

1. Build. The thermo-mechanical build of cantor_gyroid_residual.py (graded gyroid column on a
   1 mm plate, 0.1 mm voxels, 0.4 mm lumps, cooling to 20 C, plate released onto a 3-2-1
   support), re-run here with its own checkpoints; the final model state is kept.
2. Compression. The plate bottom is held at its as-built position (a platen under the plate)
   and the lattice top is moved down (bonded platen) in nominal strain steps relative to the
   lattice height; the same steps as cantor_gyroid_faults.py.
3. Reference. The same mesh and supports with a fresh, stress-free model (the room-temperature
   material of the build: yield 757 MPa, hardening 1500 MPa).
4. Criterion. As in cantor_gyroid_faults.py (tpmsfem.post.fcc_faults; tau_c = 360 MPa and lower
   thresholds; CA textures). Only lattice material that yields during the compression counts
   (plastic strain increased since the start of loading): the build has already yielded much of
   the part, and nucleation needs active dislocation sources. The as-built state before loading
   is also evaluated without that condition.

    python benchmarks/literature/cantor_gyroid_faults_asbuilt.py build      # ~1.5 h, resumable
    python benchmarks/literature/cantor_gyroid_faults_asbuilt.py compress
    python benchmarks/literature/cantor_gyroid_faults_asbuilt.py figure
"""
import json
import os
import pickle
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_faults as GF  # noqa: E402
import cantor_gyroid_residual as CR  # noqa: E402
import cantor_gyroid_solidification as CG  # noqa: E402
import cantor_material as CM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.materials import Material  # noqa: E402
from tpmsfem.post import fcc_faults as FF  # noqa: E402

RES = GF.RES
STORE = os.path.join(RES, "cantor_gyroid_faults_asbuilt.json")
FIELDS = os.path.join(RES, "cantor_gyroid_faults_asbuilt.npz")
FIG = GF.FIG.replace("cantor_gyroid_faults.png", "cantor_gyroid_faults_asbuilt.png")
CKPT_DIR = os.environ.get("ASBUILT_CKPT_DIR", os.path.join(tempfile.gettempdir(), "cantor_asbuilt_ckpt"))
CKPT_BUILD = os.path.join(CKPT_DIR, "build.pkl")
CKPT_FINAL = os.path.join(CKPT_DIR, "asbuilt_final.pkl")


def build():
    os.makedirs(CKPT_DIR, exist_ok=True)
    CR.CKPT = CKPT_BUILD                                     # CR._save writes here
    t0 = time.perf_counter()
    if os.path.exists(CKPT_FINAL):
        print("as-built state already present", flush=True)
        return
    if os.path.exists(CKPT_BUILD):
        k0, tm = pickle.load(open(CKPT_BUILD, "rb"))
        print(f"resuming after lump {k0}", flush=True)
    else:
        from tpmsfem.residual.thermomech import ThermoMechanicalAM
        from tpmsfem.thermal import HeatModel

        mesh = CR.build_mesh()
        mat = CM.mechanical_material()
        mats = {1: mat, 2: mat}
        md = tf.SolidModel(mesh, mats, T_initial=CG.T_PLATE)
        hm = HeatModel(mesh, mats, T_initial=CG.T_PLATE)
        hm.set_surface_losses(h=1e-5, emissivity=0.4, T_amb=CG.T_PLATE, face_filter=CR._top_faces)
        print(mesh.summary(), flush=True)
        sub = np.flatnonzero(mesh.cell_data["material"] == 2)
        tm = ThermoMechanicalAM(md, hm, layer_height=CR.LUMP, T_deposit=CM.LI2021["liquidus_C"],
                                layer_time=CR.LUMP / CG.LAYER * CR.LAYER_TIME, n_steps=6, first_step=0.05,
                                mech_every=2, T_plate=CG.T_PLATE, substrate=sub, n_inc=CR.N_INC)
        tm.layer_of_cell = CR.connected_order(mesh, tm.layer_of_cell)
        tm.build(layers=0)
        k0 = 0
    CR._build_from(tm, k0, t0)
    tm.cool_down(T_room=20.0)
    tm.remove_from_plate()
    CR._pickle_modules()
    tm.model._cache, tm.heat._M = {}, (None, None)
    with open(CKPT_FINAL + ".tmp", "wb") as f:
        pickle.dump(tm, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(CKPT_FINAL + ".tmp", CKPT_FINAL)
    print(f"as-built state saved, {time.perf_counter() - t0:.0f} s", flush=True)


def rt_material():
    return Material("CoCrFeMnNi_LPBF_RT", E=float(CM.youngs_modulus_MPa(20.0)), nu=0.25,
                    yield_stress=CM.KIM2020["yield_MPa"][0], hardening=1500.0)


def _vm(s):
    return np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                   + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))


def _faults(sig, active, lat, thick, ori):
    """Volume fractions of the lattice where faults can extend, per threshold."""
    tau = np.zeros((lat.sum(), GF.N_ORI))
    s = sig[lat]
    th = thick[lat]
    tau[~th] = FF.leading_partial_stress(s[~th], ori["thin"])
    tau[th] = FF.leading_partial_stress(s[th], ori["thick"])
    a = active[lat].astype(float)
    out = {f"frac_{tc:.0f}": float(((tau >= tc).mean(axis=1) * a).mean()) for tc in GF.TAU_C}
    return out, tau


def compress_case(md, label, plate_nodes, top, lat, thick, ori, H):
    u0 = md.displacement().copy()
    md.release(plate_nodes)
    md.fix(plate_nodes, (0, 1, 2), u0[plate_nodes])
    md.fix(top, (0, 1), u0[top][:, :2])
    peeq0 = md.plastic_strain()[1].mean(axis=1)
    sig0 = md.stress().mean(axis=1)
    rows, maps = [], {}
    base, _ = _faults(sig0, np.ones(len(sig0), bool), lat, thick, ori)
    print(f"{label}: before loading, faults by stress alone (360 MPa) {100 * base['frac_360']:.2f} %", flush=True)
    for eps in GF.STRAINS:
        md.fix(top, (2,), u0[top, 2] - eps * H)
        md.solve(n_inc=2, tol=1e-6)
        sig = md.stress().mean(axis=1)
        dpe = md.plastic_strain()[1].mean(axis=1) - peeq0
        active = dpe > 1e-6
        F = abs(md.reactions()[top, 2].sum())
        fr, tau = _faults(sig, active, lat, thick, ori)
        row = dict(strain=eps, nominal_stress_MPa=float(F / CG.L_CELL ** 2), yielding_frac=float(active[lat].mean()),
                   vm_p50=float(np.median(_vm(sig[lat]))), **fr)
        rows.append(row)
        print(f"{label} eps {eps * 100:4.2f} %: nominal {row['nominal_stress_MPa']:6.1f} MPa, yielding "
              f"{100 * row['yielding_frac']:5.1f} %, faults 360 {100 * fr['frac_360']:5.2f} %, 240 {100 * fr['frac_240']:5.2f} %",
              flush=True)
        if abs(eps - 0.01) < 1e-9 or abs(eps - 0.03) < 1e-9:
            maps[f"f360_{eps * 100:.0f}"] = ((tau >= 360.0).mean(axis=1) * active[lat]).astype(np.float16)
    return dict(before_loading=base, steps=rows, vm_before_p50=float(np.median(_vm(sig0[lat]))),
                vm_before_p95=float(np.percentile(_vm(sig0[lat]), 95))), maps


def compress():
    t0 = time.perf_counter()
    CR._pickle_modules()
    tm = pickle.load(open(CKPT_FINAL, "rb"))
    md, mesh = tm.model, tm.model.mesh
    cen = mesh.centroids()
    lat = (np.asarray(mesh.cell_data["material"]) == 1) & (cen[:, 2] > 0.0)
    thick = cen[:, 0] >= 0.5 * CG.L_CELL
    z = mesh.points[:, 2]
    top = np.flatnonzero(z > z.max() - 1e-9)
    plate_nodes = np.asarray(tm.plate_nodes)
    H = float(z.max())                                      # lattice height above the plate (z = 0)
    ori = GF.orientations()
    out, maps = {}, {}
    out["as_built"], m1 = compress_case(md, "as-built", plate_nodes, top, lat, thick, ori, H)
    ref = tf.SolidModel(mesh, {1: rt_material(), 2: rt_material()})
    out["stress_free"], m2 = compress_case(ref, "stress-free", plate_nodes, top, lat, thick, ori, H)
    out.update(tau_c_MPa=list(GF.TAU_C), lattice_height_mm=H, n_lattice_voxels=int(lat.sum()),
               seconds=time.perf_counter() - t0)
    json.dump(out, open(STORE, "w"), indent=1)
    ijk = mesh.cell_data["ijk"][lat].astype(np.int16)
    np.savez_compressed(FIELDS, ijk=ijk, **{f"asbuilt_{k}": v for k, v in m1.items()},
                        **{f"free_{k}": v for k, v in m2.items()})
    print("wrote", STORE, FIELDS, flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COL = {360.0: "#b42318", 300.0: "#eb6834", 240.0: "#2a78d6", 180.0: "#1baf7a", 120.0: "#7a5af8"}
    s = json.load(open(STORE))
    d = np.load(FIELDS)

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.35, 0.8, 0.8], wspace=0.35, left=0.05, right=0.98, top=0.82, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    for case, ls in (("as_built", "-"), ("stress_free", "--")):
        r = s[case]["steps"]
        e = [100 * x["strain"] for x in r]
        ax.plot(e, [x["nominal_stress_MPa"] for x in r], ls, color=INK, lw=1.5, marker="o", ms=3,
                label=case.replace("_", "-"))
    ax.set_xlabel("nominal compressive strain [%]", fontsize=8.5)
    ax.set_ylabel("nominal compressive stress [MPa]", fontsize=8.5)
    ax.set_title("(a) compression with and without the build", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    style(ax)
    ax = fig.add_subplot(gs[1])
    for tc in (360.0, 240.0, 120.0):
        for case, ls in (("as_built", "-"), ("stress_free", "--")):
            r = s[case]["steps"]
            ax.plot([100 * x["strain"] for x in r], [100 * x[f"frac_{tc:.0f}"] for x in r], ls, color=COL[tc], lw=1.5,
                    marker="o" if case == "as_built" else None, ms=3,
                    label=f"tau_c = {tc:.0f} MPa, {case.replace('_', '-')}")
    ax.set_xlabel("nominal compressive strain [%]", fontsize=8.5)
    ax.set_ylabel("lattice volume where faults can extend [%]", fontsize=8.5)
    ax.set_title("(b) faults can extend (solid: as-built; dashed: stress-free)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.8, frameon=False, ncol=1, loc="upper left")
    style(ax)
    ijk = d["ijk"].astype(int)
    jmid = (ijk[:, 1].min() + ijk[:, 1].max()) // 2

    def section(v):
        sel = ijk[:, 1] == jmid
        i0, k0 = ijk[:, 0].min(), ijk[:, 2].min()
        img = np.full((ijk[:, 0].max() - i0 + 1, ijk[:, 2].max() - k0 + 1), np.nan)
        img[ijk[sel, 0] - i0, ijk[sel, 2] - k0] = v[sel]
        return img.T

    for k, (key, title) in enumerate((("asbuilt_f360_1", "(c) as-built, 1 % strain"), ("free_f360_1", "(d) stress-free, 1 % strain"))):
        ax = fig.add_subplot(gs[2 + k])
        im = ax.imshow(section(d[key].astype(float)), origin="lower", extent=[0, CG.L_CELL, 0, s["lattice_height_mm"]],
                       cmap="magma", vmin=0, vmax=0.5, interpolation="nearest")
        ax.set_title(title + "\nfaults, tau_c = 360 MPa", fontsize=9, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("z [mm]", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
        cb.set_label("fraction of grains", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Level 0 with the build: the CoCrFeMnNi gyroid column compressed from its simulated as-built state "
                 "(residual stress and build plasticity) against the same column stress-free (same mesh, plate and supports)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "figure"
    {"build": build, "compress": compress, "figure": figure}[cmd]()
