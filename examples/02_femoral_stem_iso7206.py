"""Femoral stem with a TPMS lattice core under ISO 7206-4-style loading.

Compares a solid stem with a lattice-core stem (stiffness = stress-shielding
proxy, peak stresses for fatigue assessment), writes VTU files for ParaView and
Abaqus/CalculiX decks of the exact same discrete problems for benchmarking.

Resolution matters: use >= 4 voxels across the thinnest lattice wall
(``--h`` <= wall/4). That is a few million elements for a real stem -> GPU.

    python examples/02_femoral_stem_iso7206.py --h 0.5 --cell 6 --rho 0.5          # CPU demo
    TPMSFEM_BACKEND=cupy python examples/02_femoral_stem_iso7206.py --h 0.15 --cell 4 --rho 0.35
"""
import argparse
import json
import os

import numpy as np

import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, FemoralStem
from tpmsfem.io.abaqus import write_inp
from tpmsfem.loadcases import head_contact_faces, iso7206_4_stem
from tpmsfem.mesh import voxelize
from tpmsfem.post import results as R

p = argparse.ArgumentParser()
p.add_argument("--h", type=float, default=0.5)
p.add_argument("--cell", type=float, default=6.0)
p.add_argument("--rho", type=float, default=0.5)
p.add_argument("--surface", default="gyroid")
p.add_argument("--designs", default="solid,lattice_core")
p.add_argument("--force", type=float, default=2300.0)
p.add_argument("--solver", default="auto")
p.add_argument("--out", default="results/02")
p.add_argument("--no-export", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
print(tf.device_info())

summary = {}
for design in a.designs.split(","):
    stem = FemoralStem(design=design, lattice=TPMS(a.surface, cell_size=a.cell, kind="sheet", density=a.rho))
    lo, hi = stem.bounds()
    mesh = voxelize(stem.field(), lo, hi, a.h)
    F, z_embed = iso7206_4_stem(stem.head_center, F=a.force)
    md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
    print(f"\n[{design}] {mesh.summary()}\n  load {np.round(F, 1)} N, potted below z={z_embed:.1f} mm")
    md.fix(mesh.nodes_where(lambda x, y, z: z <= z_embed + 1e-9))
    faces, ft = head_contact_faces(mesh, stem.head_center, stem.head_radius, F)
    md.add_surface_load(faces, ft, total_force=F)
    info = md.solve(solver=a.solver, verbose=False, name=design)
    print(f"  solve: {info}")

    r = R.save_vtu(md, os.path.join(a.out, f"stem_{design}.vtu"))
    u = r["U"]
    loaded = np.unique(faces)
    delta = float((u[loaded].mean(axis=0) @ F) / np.linalg.norm(F))
    zc = mesh.centroids()[:, 2]
    vm = r["vonMises_elem"]
    s1 = r["S_principal_elem"][:, 2]
    zone = (zc > stem.lattice_z[0]) & (zc < stem.lattice_z[1]) & (zc > z_embed)
    neck = zc > stem.neck_origin_z
    res = dict(
        n_cells=mesh.n_cells, dofs=md.ndof, seconds=info.seconds, krylov=info.krylov_iterations,
        head_displacement_mm=delta, stiffness_N_per_mm=float(np.linalg.norm(F) / delta),
        vm_p99_lattice_zone=float(np.percentile(vm[zone], 99)), vm_max_lattice_zone=float(vm[zone].max()),
        s1_max_lattice_zone=float(s1[zone].max()), vm_p99_neck=float(np.percentile(vm[neck], 99)),
        reaction_N=np.round(md.reactions().sum(axis=0), 3).tolist(),
    )
    summary[design] = res
    print(f"  head displacement along F: {delta:.4f} mm -> stiffness {res['stiffness_N_per_mm']:.0f} N/mm")
    print("  " + R.summarize(vm[zone], "von Mises, lattice zone above potting"))
    print("  " + R.summarize(vm[neck], "von Mises, neck"))
    print(f"  reaction sum {res['reaction_N']} (should balance {np.round(-F, 1)})")
    if not a.no_export:
        for dialect in ("abaqus", "calculix"):
            write_inp(md, os.path.join(a.out, f"stem_{design}_{dialect}.inp"), dialect=dialect,
                      title=f"tpmsfem femoral stem {design} ISO7206-4")

if {"solid", "lattice_core"} <= summary.keys():
    k = summary["lattice_core"]["stiffness_N_per_mm"] / summary["solid"]["stiffness_N_per_mm"]
    print(f"\nlattice-core / solid stem stiffness: {k:.3f}  (lower => less stress shielding)")
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print("Compare peak principal stresses against the fatigue strength of YOUR as-built material;"
      " voxel stair-steps inflate single-element maxima -- use percentiles or a smoothed (tet) mesh for hot spots.")
