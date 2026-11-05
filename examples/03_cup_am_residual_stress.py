"""LPBF residual stress & distortion of a porous (TPMS) acetabular cup.

Inherent-strain, layer-by-layer simulation: the cup is built rim-down on the
plate, each super-layer is born strain-free and shrinks by eps*, then the cup
is cut from the plate (3-2-1 support) and springs back.

Outputs: per-layer history, residual von Mises / principal stress, rim
distortion (out-of-roundness and flatness, which matter for liner seating),
VTU files and an Abaqus deck of the identical process for benchmarking.

    python examples/03_cup_am_residual_stress.py --h 0.6 --layer 1.2           # CPU demo
    python examples/03_cup_am_residual_stress.py --plastic --n-inc 2            # elasto-plastic
"""
import argparse
import json
import os

import numpy as np

import tpmsfem as tf
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, AcetabularCup
from tpmsfem.io.abaqus import write_inp
from tpmsfem.mesh import voxelize
from tpmsfem.post import results as R
from tpmsfem.residual.am import AMProcess

p = argparse.ArgumentParser()
p.add_argument("--h", type=float, default=0.5, help="voxel size [mm]")
p.add_argument("--layer", type=float, default=1.0, help="super-layer height [mm] (multiple of h)")
p.add_argument("--cell", type=float, default=3.0)
p.add_argument("--rho", type=float, default=0.4)
p.add_argument("--eps", default="-0.004,-0.004,-0.002", help="inherent strain xx,yy,zz (CALIBRATE!)")
p.add_argument("--plastic", action="store_true")
p.add_argument("--n-inc", type=int, default=1)
p.add_argument("--mode", default="element", choices=["element", "nodal"])
p.add_argument("--solver", default="auto")
p.add_argument("--out", default="results/03")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
print(tf.device_info())

cup = AcetabularCup(porous_thickness=3.0, lattice=TPMS("gyroid", cell_size=a.cell, kind="sheet", density=a.rho))
lo, hi = cup.bounds()
lo[2] = 0.0                                   # rim exactly on the plate
mesh = voxelize(cup.field(), lo, hi, a.h)
mat = TI6AL4V_LPBF if a.plastic else TI6AL4V_LPBF.elastic()
md = tf.SolidModel(mesh, mat)
print(mesh.summary())
print(md.summary())

eps = tuple(float(v) for v in a.eps.split(",")) + (0.0, 0.0, 0.0)
am = AMProcess(md, layer_height=a.layer, inherent_strain=eps, mode=a.mode, n_inc=a.n_inc, solver=a.solver)
print(f"{am.n_layers} super-layers of {a.layer} mm, {len(am.plate_nodes)} nodes welded to the plate")
if not a.plastic:   # export the identical process for Abaqus (nodal mode semantics, see docs)
    write_inp(tf.SolidModel(mesh, mat), os.path.join(a.out, "cup_am_abaqus.inp"), dialect="abaqus",
              am=AMProcess(tf.SolidModel(mesh, mat), a.layer, eps, mode="nodal"))

am.build(verbose=True)
r_plate = R.save_vtu(md, os.path.join(a.out, "cup_on_plate.vtu"))
u_on_plate = md.displacement().copy()
am.remove_from_plate(verbose=True)
r = R.save_vtu(md, os.path.join(a.out, "cup_removed.vtu"),
               extra_point=dict(U_on_plate=u_on_plate),
               extra_cell=dict(layer=am.layer_of_cell.astype(np.int32)))


def rim_metrics(u):
    """Out-of-roundness and flatness of the inner rim edge after distortion."""
    pts = mesh.points
    rr = np.hypot(pts[:, 0], pts[:, 1])
    rim = np.flatnonzero((pts[:, 2] < 1e-9) & (np.abs(rr - cup.inner_radius) < 1.5 * a.h))
    q = pts[rim] + u[rim]
    dr = np.hypot(q[:, 0], q[:, 1]) - rr[rim]          # radial displacement of each rim node
    return dict(n_nodes=len(rim), out_of_roundness=float(np.ptp(dr)), flatness=float(np.ptp(q[:, 2])),
                mean_radius_change=float(dr.mean()))


rim = rim_metrics(md.displacement())
vm = r["vonMises_elem"]
print("\n" + R.summarize(r_plate["vonMises_elem"][md.active_host], "residual von Mises ON plate"))
print(R.summarize(vm, "residual von Mises AFTER removal"))
print(f"inner rim after removal: out-of-roundness {rim['out_of_roundness'] * 1000:.1f} um, "
      f"flatness {rim['flatness'] * 1000:.1f} um, mean radius change {rim['mean_radius_change'] * 1000:.1f} um")
if a.plastic:
    print(R.summarize(r["PEEQ_elem"], "equivalent plastic strain", unit=""))
json.dump(dict(rim=rim, vm_p99=float(np.percentile(vm, 99)), vm_max=float(vm.max()),
               layers=[rec.__dict__ for rec in am.records]),
          open(os.path.join(a.out, "summary.json"), "w"), indent=1)
