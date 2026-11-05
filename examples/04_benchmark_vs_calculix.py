"""Code-to-code benchmark: tpmsfem vs CalculiX (and decks for Abaqus).

Every case is solved by tpmsfem, exported with *identical* nodes, elements,
constraints, consistent nodal loads and eigenstrains, solved by CalculiX, and
compared field-by-field. The same cases are written as Abaqus decks together
with tpmsfem's results (.npz) so a commercial run can be compared with:

    abaqus job=<case>_abaqus interactive
    abaqus python tools/abaqus_odb_export.py <case>_abaqus.odb <case>_abq
    python -m tpmsfem.bench.compare <case>_tpmsfem.npz --abaqus <case>_abq

Element formulation must match: hex8 here is fully integrated = CalculiX C3D8.
Abaqus' C3D8 is selectively reduced (B-bar) -> rerun with ``--bbar`` for Abaqus.

    python examples/04_benchmark_vs_calculix.py [--scale 1.0] [--bbar]
"""
import argparse
import json
import os
import time

import numpy as np

import tpmsfem as tf
from tpmsfem.bench import analytical as A
from tpmsfem.bench.compare import compare, save_ours
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, AcetabularCup, FemoralStem, lattice_cube
from tpmsfem.io.abaqus import write_inp
from tpmsfem.io.calculix import frd_field, read_dat_stress, read_frd, run_ccx
from tpmsfem.loadcases import head_contact_faces, iso7206_4_stem
from tpmsfem.mesh import voxelize
from tpmsfem.residual.am import AMProcess

p = argparse.ArgumentParser()
p.add_argument("--scale", type=float, default=1.0, help="<1 refines all meshes")
p.add_argument("--bbar", action="store_true", help="B-bar hex8 (to compare with Abaqus C3D8)")
p.add_argument("--cases", default="lattice,stem,cup_am,sphere_tet10")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--out", default="results/04")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
MAT = TI6AL4V_LPBF.elastic()
bbar = True if a.bbar else None
print(tf.device_info())


def case_lattice():
    h = 0.1 * a.scale
    lat = TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.35)
    m = voxelize(lattice_cube(lat, 4.0, plate=0.5), (0, 0, 0), (4, 4, 4), 4.0 / round(4.0 / h))
    md = tf.SolidModel(m, MAT, bbar=bbar)
    z = m.points[:, 2]
    md.fix(np.flatnonzero(z < 1e-9))
    faces, ft, _ = m.boundary_faces()
    top = faces[m.points[faces].mean(1)[:, 2] > 4 - 1e-9]
    md.add_surface_load(top, ft, total_force=(50.0, 0.0, -400.0))
    return md, None, "gyroid sheet cube, compression + shear (voxel hex8)"


def case_stem():
    stem = FemoralStem(design="lattice_core", lattice=TPMS("gyroid", cell_size=6.0, kind="sheet", density=0.5))
    lo, hi = stem.bounds()
    m = voxelize(stem.field(), lo, hi, 1.0 * a.scale)
    F, z_embed = iso7206_4_stem(stem.head_center)
    md = tf.SolidModel(m, MAT, bbar=bbar)
    md.fix(m.nodes_where(lambda x, y, z: z <= z_embed + 1e-9))
    faces, ft = head_contact_faces(m, stem.head_center, stem.head_radius, F)
    md.add_surface_load(faces, ft, total_force=F)
    return md, None, "lattice-core femoral stem, ISO 7206-4 load (voxel hex8)"


def case_cup_am():
    cup = AcetabularCup(porous_thickness=3.0, lattice=TPMS("gyroid", cell_size=3.0, kind="sheet", density=0.45))
    lo, hi = cup.bounds()
    lo[2] = 0.0
    m = voxelize(cup.field(), lo, hi, 1.0 * a.scale)
    md = tf.SolidModel(m, MAT, bbar=bbar)
    eps = (-0.004, -0.004, -0.002, 0, 0, 0)
    am = AMProcess(md, layer_height=2.0 * a.scale, inherent_strain=eps, mode="nodal",
                   birth_reference="element_mean", solver="auto")
    deck_am = AMProcess(tf.SolidModel(m, MAT, bbar=bbar), 2.0 * a.scale, eps, mode="nodal")
    return md, (am, deck_am), "porous cup, LPBF layer-by-layer + plate removal (voxel hex8)"


def case_sphere_tet10():
    m = A.gmsh_sphere_shell_octant(20.0, 27.0, size=1.5 * a.scale, order=2)
    md = tf.SolidModel(m, MAT)
    x, y, z = m.points.T
    for c, coord in enumerate((x, y, z)):
        md.fix(np.flatnonzero(np.abs(coord) < 1e-9), (c,))
    faces, ft, _ = m.boundary_faces()
    fr = np.linalg.norm(m.points[faces[:, :3]].mean(1), axis=1)
    md.add_surface_load(faces[fr < 23.5], ft, pressure=50.0)
    return md, None, "thick sphere (cup-like) internal pressure, Gmsh tet10"


CASES = dict(lattice=case_lattice, stem=case_stem, cup_am=case_cup_am, sphere_tet10=case_sphere_tet10)
report = {}
for name in a.cases.split(","):
    md, am, desc = CASES[name]()
    print(f"\n=== {name}: {desc}\n    {md.summary()}")
    t0 = time.perf_counter()
    if am is None:
        md.solve(solver="auto")
    else:
        am[0].build(verbose=False)
        am[0].remove_from_plate(verbose=False)
    t_ours = time.perf_counter() - t0
    save_ours(md, os.path.join(a.out, f"{name}_tpmsfem.npz"))
    deck_kw = dict(am=am[1]) if am else {}
    write_inp(md if am is None else tf.SolidModel(md.mesh, MAT, bbar=bbar),
              os.path.join(a.out, f"{name}_abaqus.inp"), dialect="abaqus", title=desc, **deck_kw)
    deck = write_inp(md if am is None else tf.SolidModel(md.mesh, MAT, bbar=bbar),
                     os.path.join(a.out, f"{name}_ccx.inp"), dialect="calculix", title=desc, **deck_kw)
    r = run_ccx(deck, threads=a.threads)
    U_ref = frd_field(read_frd(r["frd"]), "DISP", md.n_nodes)
    S_ref = read_dat_stress(r["dat"], md.ne, md.nq).mean(axis=1)
    res = compare(md.displacement(), md.stress().mean(axis=1), U_ref, S_ref, active=md.active_host)
    res.update(case=desc, elements=md.ne, dofs=md.ndof, element=md.mesh.cell_type + ("-bbar" if md.bbar else ""),
               t_tpmsfem_s=t_ours, t_calculix_s=r["seconds"])
    report[name] = res
    print(f"    U rel-L2 {res['U_rel_l2']:.2e} | S rel-L2 {res['S_rel_l2']:.2e} | vM p99 diff "
          f"{100 * res['vonMises_percentiles']['p99']:+.4f} % | tpmsfem {t_ours:.1f}s, CalculiX {r['seconds']:.1f}s")

json.dump(report, open(os.path.join(a.out, "benchmark.json"), "w"), indent=1)
with open(os.path.join(a.out, "benchmark.md"), "w") as f:
    f.write(f"# tpmsfem vs CalculiX ({tf.device_info()})\n\n")
    f.write("| case | element | dofs | U rel-L2 | S rel-L2 | vM p99 diff | vM max diff | tpmsfem [s] | CalculiX [s] |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for k, v in report.items():
        f.write(f"| {k} | {v['element']} | {v['dofs']:,} | {v['U_rel_l2']:.1e} | {v['S_rel_l2']:.1e} | "
                f"{100 * v['vonMises_percentiles']['p99']:+.3f} % | {100 * v['vonMises_percentiles']['p100']:+.3f} % | "
                f"{v['t_tpmsfem_s']:.1f} | {v['t_calculix_s']:.1f} |\n")
    f.write("\nU: nodal displacement; S: element-mean integration-point stress. CalculiX writes 6 significant"
            " digits to .frd, so ~1e-6 is the agreement floor for U.\n")
print(f"\nreport: {os.path.join(a.out, 'benchmark.md')}")
