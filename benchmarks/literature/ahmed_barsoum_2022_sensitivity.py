"""Sensitivity of the 3x3x3 modulus to modelling choices the paper does not state.

Ahmed et al. (2022) do not specify (i) whether the top "remote point" was
rigid (Ansys default) or deformable, nor (ii) where the TPMS phase sits
relative to the specimen faces (MSLattice cell origin). Both change the
stiffness of a 3-cell-high specimen through end effects. This script solves
P-S and IWP-S at rho = 20 % on one moderate grid for each variant:

* ``rigid``       top face u_x = u_y = 0, u_z = delta (baseline)
* ``deformable``  top face u_z = delta only (in-plane free)
* ``shift L/4``   full (non-symmetric) model, lattice shifted by a quarter cell,
  compared with the unshifted full model on the same grid. (A half-cell shift
  maps the P and I-WP sheets onto themselves, so it is not a test.)

    python benchmarks/literature/ahmed_barsoum_2022_sensitivity.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ahmed_barsoum_2022 as B  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.geometry import TPMS, box  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402


def run_full(topo, rho, n, shift):
    surf, kind, side, _ = B.TOPOLOGIES[topo]
    o = shift * B.L
    lat = TPMS(surf, cell_size=B.L, kind=kind, density=rho, side=side, origin=(o, o, o))
    mesh = voxelize(lat & box((0, 0, 0), (30, 30, 30)), (0, 0, 0), (30, 30, 30), B.L / n)
    md = tf.SolidModel(mesh, B.IN718)
    z = mesh.points[:, 2]
    top = np.flatnonzero(z > 30 - 1e-9)
    md.fix(np.flatnonzero(z < 1e-9), (2,), 0.0)
    md.fix(top, (0, 1), 0.0)
    md.fix(top, (2,), -0.03)
    md.solve(solver="auto")
    vf = mesh.n_cells * (B.L / n) ** 3 / 27000.0
    return (md.reactions()[top, 2].sum() / 900.0) / (-0.03 / 30.0) / B.IN718.E, vf


def run(topo, rho, n, top_bc, shift):
    surf, kind, side, _ = B.TOPOLOGIES[topo]
    o = shift * B.L
    lat = TPMS(surf, cell_size=B.L, kind=kind, density=rho, side=side, origin=(o, o, o))
    h = B.L / n
    mesh = voxelize(lat & box((0, 0, 0), (30, 30, 30)), (0, 0, 0), (15, 15, 30), h)
    md = tf.SolidModel(mesh, B.IN718)
    x, y, z = mesh.points.T
    top = np.flatnonzero(z > 30 - 1e-9)
    md.fix(np.flatnonzero(z < 1e-9), (2,), 0.0)
    md.fix(np.flatnonzero(x > 15 - 1e-9), (0,), 0.0)
    md.fix(np.flatnonzero(y > 15 - 1e-9), (1,), 0.0)
    if top_bc == "rigid":
        md.fix(top, (0, 1), 0.0)
    md.fix(top, (2,), -0.03)
    md.solve(solver="auto")
    F = md.reactions()[top, 2].sum() * 4
    return (F / 900.0) / (-0.03 / 30.0) / B.IN718.E


if __name__ == "__main__":
    out = {}
    for topo in ("P-S", "IWP-S"):
        rho, n = 0.20, 42
        base = run(topo, rho, n, "rigid", 0.0)
        (f0, v0), (f1, v1) = run_full(topo, rho, 28, 0.0), run_full(topo, rho, 28, 0.25)
        out[topo] = dict(rigid=base, deformable=run(topo, rho, n, "deformable", 0.0),
                         full_n28_shift0=f0, full_n28_shift_quarter=f1, vf_shift0=v0, vf_shift_quarter=v1)
        r = out[topo]
        print(f"{topo} rho={rho} n={n}: rigid {base:.5f} | deformable {r['deformable']:.5f} "
              f"({100 * (r['deformable'] / base - 1):+.1f} %) | full model n=28: shift 0 {f0:.5f} (vf {v0:.3f}), "
              f"quarter-cell shift {f1:.5f} (vf {v1:.3f}) ({100 * (f1 / f0 - 1):+.1f} %)", flush=True)
    os.makedirs("results/literature", exist_ok=True)
    json.dump(out, open("results/literature/ahmed_barsoum_2022_sensitivity.json", "w"), indent=1)
