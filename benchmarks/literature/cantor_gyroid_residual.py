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


def run(layers=None):
    mesh = build_mesh()
    mat = CM.mechanical_material()
    mats = {1: mat, 2: mat}
    md = tf.SolidModel(mesh, mats, T_initial=CG.T_PLATE)
    hm = HeatModel(mesh, mats, T_initial=CG.T_PLATE)
    hm.set_surface_losses(h=1e-5, emissivity=0.4, T_amb=CG.T_PLATE,
                          face_filter=lambda c, n: (n[:, 2] > 0.5) & (c[:, 2] > 1e-6))
    # the heat iteration can settle into a 0.3 K limit cycle across the latent-heat range on this
    # voxel lattice; 1 K is ample for the thermal strains
    heat_step = hm.step
    hm.step = lambda dt, **kw: heat_step(dt, tol=1.0, **kw)
    print(mesh.summary(), flush=True)
    sub = np.flatnonzero(mesh.cell_data["material"] == 2)
    tm = ThermoMechanicalAM(md, hm, layer_height=LUMP, T_deposit=CM.LI2021["liquidus_C"],
                            layer_time=LUMP / CG.LAYER * LAYER_TIME, n_steps=6, first_step=0.05, mech_every=2,
                            T_plate=CG.T_PLATE, substrate=sub, n_inc=N_INC)
    tm.layer_of_cell = connected_order(mesh, tm.layer_of_cell)
    t0 = time.perf_counter()
    tm.build(layers=layers)
    if layers is not None:
        print(f"{layers} lumps in {time.perf_counter() - t0:.0f} s", flush=True)
        return
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    if not a.figure:
        run(a.layers)
