"""Residual stress and strain in an LPBF CoCrFeMnNi gyroid-sheet lattice (thermo-mechanical build).

The graded gyroid column of cantor_gyroid_solidification.py (4 mm cell, walls 0.2 -> 0.6 mm
along x) is built 8 mm tall on a 1 mm plate with the layer-lumped, sequentially coupled model
(tpmsfem.residual.ThermoMechanicalAM): each lump is born stress-free at the liquidus and cools
on the part below, with temperature-dependent elasticity, expansion and yield
(cantor_material.mechanical_material). After the last lump the part cools to 20 C on the
plate, then the plate is released (3-2-1 support).

Voxels 0.1 mm (the thinnest walls are 2 voxels wide), lumps of 0.4 mm (13 layers of 30 um),
10 s per layer, plate at 80 C. Loose powder is not modelled mechanically or thermally here.

Output: element stresses, elastic (lattice) strains and plastic strains on the plate and after
removal (results/cantor_gyroid_residual.npz), and the figure.

    python benchmarks/literature/cantor_gyroid_residual.py            # run + figure
    python benchmarks/literature/cantor_gyroid_residual.py --figure
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
import cantor_material as CM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.mesh.voxel import keep_connected, mask_to_mesh  # noqa: E402
from tpmsfem.residual.thermomech import ThermoMechanicalAM  # noqa: E402
from tpmsfem.thermal import HeatModel  # noqa: E402

OUT = "benchmarks/literature/results/cantor_gyroid_residual.npz"
FIG = "docs/img/cantor_gyroid_residual.png"
VOX, HEIGHT, PLATE, MARGIN = 0.1, 8.0, 1.0, 0.5
LUMP, LAYER_TIME = 0.4, 10.0
N_INC = 2


def build_mesh():
    n_xy = int(round((CG.L_CELL + 2 * MARGIN) / VOX))
    nzp, nzl = int(round(PLATE / VOX)), int(round(HEIGHT / VOX))
    c = (np.arange(n_xy) + 0.5) * VOX - MARGIN
    zc = -PLATE + (np.arange(nzp + nzl) + 0.5) * VOX
    X, Y, Z = np.meshgrid(c, c, zc, indexing="ij")
    inside = (X > 0) & (X < CG.L_CELL) & (Y > 0) & (Y < CG.L_CELL)
    lattice = (Z > 0) & inside & (np.asarray(CG.gyroid()(X, Y, Z)) < 0.0)
    mask = keep_connected((Z < 0) | lattice)
    mesh = mask_to_mesh(mask, (-MARGIN, -MARGIN, -PLATE), VOX, materials=np.where(Z < 0, 2, 1))
    return mesh


def connected_order(mesh, layer):
    """Postpone voxels that would be born unsupported. A gyroid slice can contain a piece that
    only joins the part a few layers higher; in the machine the powder holds it, here it would
    float. Each lump therefore activates only the cells connected (by faces) to what is already
    built; the rest wait for the lump in which they connect."""
    from scipy import ndimage

    ijk = mesh.cell_data["ijk"]
    grid = np.full(tuple(ijk.max(axis=0) + 1), -2, np.int64)             # -2: no cell
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = layer
    built = grid == -1
    out = np.where(grid == -1, -1, -2)
    structure = ndimage.generate_binary_structure(3, 1)
    n_layers = int(layer.max()) + 1
    for k in range(n_layers):
        cand = built | ((grid >= 0) & (grid <= k) & (out == -2))
        lab, _ = ndimage.label(cand, structure=structure)
        keep = np.isin(lab, np.unique(lab[built])) & cand & ~built
        out[keep] = k
        built |= keep
    left = (grid >= 0) & (out == -2)
    out[left] = n_layers - 1                                              # never connected: with the last lump
    new = out[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    print(f"build order: {int((new != layer).sum())} of {int((layer >= 0).sum())} part cells postponed, "
          f"{int(left.sum())} never connect", flush=True)
    return new


def hooke_strain(sig, E, nu):
    """Elastic strain (..., 6) from stress (..., 6) [xx, yy, zz, yz, xz, xy] (engineering shear)."""
    e = np.empty_like(sig)
    tr = sig[..., :3].sum(axis=-1, keepdims=True)
    e[..., :3] = ((1 + nu) * sig[..., :3] - nu * tr) / E
    e[..., 3:] = 2 * (1 + nu) * sig[..., 3:] / E
    return e


def _top_faces(c, n):
    return (n[:, 2] > 0.5) & (c[:, 2] > 1e-6)


CKPT = OUT.replace(".npz", "_checkpoint.pkl")


def _pickle_modules():
    """The models keep a reference to their array module (numpy / cupy): pickle it by name."""
    import copyreg
    import importlib
    import types

    copyreg.pickle(types.ModuleType, lambda m: (importlib.import_module, (m.__name__,)))


def _save(tm, k):
    """Checkpoint after every lump: long runs get interrupted on this infrastructure."""
    import pickle

    _pickle_modules()
    tmp = CKPT + ".tmp"
    cache, pre = tm.model._cache, getattr(tm.heat, "_M", None)
    tm.model._cache, tm.heat._M = {}, (None, None)     # solver caches hold closures; they are rebuilt on demand
    try:
        with open(tmp, "wb") as f:
            pickle.dump((k, tm), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CKPT)
    finally:
        tm.model._cache = cache
        if pre is not None:
            tm.heat._M = pre


def _build_from(tm, k0, t0):
    """ThermoMechanicalAM.build for lumps k0.. (the same steps, without the initial set-up)."""
    m, hm = tm.model, tm.heat
    for k in range(k0, tm.n_layers):
        t1 = time.perf_counter()
        cells = np.flatnonzero(tm.layer_of_cell == k)
        if len(cells) == 0:
            continue
        hm.activate(cells, T_new=tm.T_deposit)
        m.set_temperature(hm.T, ramp=False)
        m.activate(cells, strain_free=True)
        for i, dt in enumerate(tm.steps):
            hm.step(dt, tol=1.0)       # the heat iteration can sit in a 0.3 K limit cycle; 1 K is ample
            if (i + 1) % tm.mech_every == 0 or i == len(tm.steps) - 1:
                tm._mech(f"layer{k}.{i}")
        print(f"lump {k + 1:2d}/{tm.n_layers}: z <= {(k + 1) * LUMP:4.1f} mm, max von Mises {tm._max_vm():5.0f} MPa, "
              f"{time.perf_counter() - t1:5.0f} s (total {time.perf_counter() - t0:.0f} s)", flush=True)
        try:
            _save(tm, k + 1)
        except Exception as err:                       # keep building if the state cannot be pickled
            print(f"checkpoint failed: {err!r}", flush=True)


def run(layers=None):
    import pickle

    t0 = time.perf_counter()
    if os.path.exists(CKPT):
        k0, tm = pickle.load(open(CKPT, "rb"))
        md, mesh = tm.model, tm.model.mesh
        print(f"resuming after lump {k0}", flush=True)
    else:
        mesh = build_mesh()
        mat = CM.mechanical_material()
        mats = {1: mat, 2: mat}
        md = tf.SolidModel(mesh, mats, T_initial=CG.T_PLATE)
        hm = HeatModel(mesh, mats, T_initial=CG.T_PLATE)
        hm.set_surface_losses(h=1e-5, emissivity=0.4, T_amb=CG.T_PLATE, face_filter=_top_faces)
        print(mesh.summary(), flush=True)
        sub = np.flatnonzero(mesh.cell_data["material"] == 2)
        tm = ThermoMechanicalAM(md, hm, layer_height=LUMP, T_deposit=CM.LI2021["liquidus_C"],
                                layer_time=LUMP / CG.LAYER * LAYER_TIME, n_steps=6, first_step=0.05, mech_every=2,
                                T_plate=CG.T_PLATE, substrate=sub, n_inc=N_INC)
        tm.layer_of_cell = connected_order(mesh, tm.layer_of_cell)
        tm.build(layers=0)                                  # initial set-up only (plate, supports)
        k0 = 0
    _build_from(tm, k0, t0)
    tm.cool_down(T_room=20.0)
    E, nu = float(CM.youngs_modulus_MPa(20.0)), 0.25
    sig_plate = md.stress().mean(axis=1)
    ep_plate = md.plastic_strain()[0].mean(axis=1)
    peeq = md.plastic_strain()[1].mean(axis=1)
    u_plate = md.displacement().copy()
    tm.remove_from_plate()
    sig_free = md.stress().mean(axis=1)
    np.savez_compressed(OUT, centroids=mesh.centroids(), ijk=mesh.cell_data["ijk"], material=mesh.cell_data["material"],
                        stress_on_plate=sig_plate, stress_removed=sig_free, eps_el_on_plate=hooke_strain(sig_plate, E, nu),
                        eps_el_removed=hooke_strain(sig_free, E, nu), eps_p=ep_plate, peeq=peeq,
                        u_on_plate=u_plate, u_removed=md.displacement(), points=mesh.points, vox=VOX,
                        seconds=time.perf_counter() - t0, records=json.dumps([r.__dict__ for r in tm.records]))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(OUT)
    ijk, vox = d["ijk"], float(d["vox"])
    shape = tuple(ijk.max(axis=0) + 1)

    def grid(v):
        g = np.full(shape, np.nan)
        g[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = v
        return g

    x = (np.arange(shape[0]) + 0.5) * vox - MARGIN
    z = (np.arange(shape[2]) + 0.5) * vox - PLATE
    ezz_p, ezz_r = grid(d["eps_el_on_plate"][:, 2] * 1e6), grid(d["eps_el_removed"][:, 2] * 1e6)
    exx_r = grid(d["eps_el_removed"][:, 0] * 1e6)
    peeq = grid(d["peeq"] * 100.0)
    s = d["stress_removed"]
    vm = grid(np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                      + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)))
    jy = int(np.argmin(np.abs(x - 2.0)))
    lim = float(np.nanpercentile(np.abs(ezz_p[:, :, z > 0]), 98))
    lim = max(round(lim, -2), 500.0)
    MUTED, GRID, BLUE, ORANGE = "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.45, 1], hspace=0.3, wspace=0.42, left=0.05, right=0.97, top=0.92,
                          bottom=0.07)
    ext = [x[0] - vox / 2, x[-1] + vox / 2, z[0] - vox / 2, z[-1] + vox / 2]
    panels = [(ezz_p, "(a) elastic strain eps_zz, on the plate", "RdBu_r", -lim, lim, "microstrain"),
              (ezz_r, "(b) elastic strain eps_zz, plate removed", "RdBu_r", -lim, lim, "microstrain"),
              (exx_r, "(c) elastic strain eps_xx, plate removed", "RdBu_r", -lim, lim, "microstrain"),
              (peeq, "(d) equivalent plastic strain", "viridis", 0.0, float(np.nanpercentile(peeq, 99)), "%")]
    for k, (g, title, cmap, lo, hi, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(g[:, jy, :].T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_facecolor("#f1f0ec")
        ax.set_title(title + "\nsection y = 2 mm (thin walls left, thick right)", fontsize=8.5, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("build height z [mm]", fontsize=8.5)
        ax.tick_params(labelsize=7.5, colors=MUTED)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label(unit, fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
    # layers: first, middle, final (horizontal sections of eps_zz after removal)
    part = z > 0
    zs = z[part]
    for k, (zl, name) in enumerate(((0.25, "first layers"), (0.5 * HEIGHT, "middle"), (HEIGHT - 0.25, "final layers"))):
        ax = fig.add_subplot(gs[1, k])
        kz = int(np.argmin(np.abs(z - zl)))
        im = ax.imshow(ezz_r[:, :, kz].T, origin="lower", extent=[ext[0], ext[1], ext[0], ext[1]], cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_facecolor("#f1f0ec")
        ax.set_title(f"({'efg'[k]}) eps_zz, plate removed\n{name}, z = {z[kz]:.2f} mm", fontsize=8.5, loc="left")
        ax.set_xlabel("x [mm]", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("y [mm]", fontsize=8.5)
        ax.tick_params(labelsize=7.5, colors=MUTED)
    cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.02, pad=0.02)
    cb.set_label("microstrain", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    # profiles with height
    ax = fig.add_subplot(gs[1, 3])
    thin, thick = x < 2.0, x >= 2.0
    for sel, col, lab in ((thin, BLUE, "thin-wall half"), (thick, ORANGE, "thick-wall half")):
        ax.plot(np.nanmean(np.abs(ezz_r[sel][:, :, part]), axis=(0, 1)), zs, color=col, lw=1.5, label=lab)
    ax.set_xlabel("mean |elastic eps_zz|, plate removed [microstrain]", fontsize=8.5)
    ax.set_ylabel("build height z [mm]", fontsize=8.5)
    ax.set_title("(h) residual strain against height", fontsize=8.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="center left")
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.suptitle("Residual strain in the LPBF CoCrFeMnNi gyroid column (thermo-mechanical build, 0.1 mm voxels, room temperature)",
                 fontsize=10.5, x=0.05, ha="left")
    fig.savefig(FIG, dpi=150)
    print("wrote", FIG)
    lat = d["material"] == 1
    out = dict(vm_removed_p50=float(np.nanmedian(vm[:, :, part])), vm_removed_p95=float(np.nanpercentile(vm[:, :, part], 95)),
               ezz_plate_p5_p95=[float(v) for v in np.percentile(d["eps_el_on_plate"][lat, 2] * 1e6, [5, 95])],
               ezz_removed_p5_p95=[float(v) for v in np.percentile(d["eps_el_removed"][lat, 2] * 1e6, [5, 95])],
               exx_removed_p5_p95=[float(v) for v in np.percentile(d["eps_el_removed"][lat, 0] * 1e6, [5, 95])],
               peeq_percent_p50_p95=[float(v) for v in np.percentile(d["peeq"][lat] * 100, [50, 95])],
               seconds=float(d["seconds"]))
    json.dump(out, open(OUT.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if not a.figure:
        run(a.layers)
    if a.layers is None:
        figure()
