"""Residual-stress effect on the apparent modulus (trend check vs Ahmed et al. 2022).

Ahmed, Barsoum & Abu Al-Rub (Metals 2022) found, with a thermo-mechanical
Ansys model, that SLM residual stresses lower the apparent elastic modulus of
IN718 TPMS lattices in the build direction -- strongly for P-S (and IWP-L),
least for IWP-S -- because pre-stressed regions yield as soon as load is
applied. tpmsfem has no thermal solver yet, so this is NOT a reproduction of
their numbers: it checks whether the inherent-strain + J2 route reproduces the
mechanism and the topology ranking with one common, illustrative inherent strain.

Steps per topology (rho = 30 %, quarter model of the 3x3x3 specimen):
1. layer-by-layer build welded to a rigid plate (inherent strain, J2 plasticity),
2. release the in-plane weld (bottom keeps u_z = 0, the test's symmetry BC),
3. attach a rigid top face and load in z (tension and compression) in small
   increments; apparent modulus = slope of the first 0.05 % strain,
4. the same test on a residual-stress-free lattice; report the reduction.

    python benchmarks/literature/ahmed_barsoum_2022_rs.py [--n 24] [--layer-voxels 2]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ahmed_barsoum_2022 as B  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.geometry import box  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402
from tpmsfem.residual.am import AMProcess  # noqa: E402

# bilinear IN718 (E, nu, yield from the paper's Table 1; tangent modulus ~2 GPa assumed)
IN718_EP = tf.Material("IN718", 165_000.0, 0.3, yield_stress=648.0, hardening=2_000.0)


def test_curve(md, top, sign, n_steps=5, strain_max=5e-4):
    """Rigid top face moved in z from the current state; returns strains, stresses."""
    u0 = md.displacement()[top].copy()
    md.fix(top, (0,), u0[:, 0])
    md.fix(top, (1,), u0[:, 1])
    out = [(0.0, 0.0)]
    F0 = md.reactions()[top, 2].sum()
    for k in range(1, n_steps + 1):
        eps = sign * strain_max * k / n_steps
        md.fix(top, (2,), u0[:, 2] + eps * 30.0)
        md.solve(n_inc=1, solver="auto", name=f"test{k}")
        F = (md.reactions()[top, 2].sum() - F0) * 4
        out.append((eps, F / 900.0))
    e, s = np.array(out).T
    return e, s


def run(topo, rho, n, layer_voxels, eps_star, verbose):
    lat = B.lattice(topo, rho)
    h = B.L / n
    mesh = voxelize(lat & box((0, 0, 0), (30, 30, 30)), (0, 0, 0), (15, 15, 30), h)
    x, y, z = mesh.points.T
    top = np.flatnonzero(z > 30 - 1e-9)
    bot = np.flatnonzero(z < 1e-9)
    symx, symy = np.flatnonzero(x > 15 - 1e-9), np.flatnonzero(y > 15 - 1e-9)
    res = dict(topology=topo, rho=rho, n=n, elements=mesh.n_cells, inherent_strain=eps_star)

    def fresh():
        md = tf.SolidModel(mesh, IN718_EP)
        md.fix(symx, (0,), 0.0)
        md.fix(symy, (1,), 0.0)
        return md

    # residual-stress-free reference
    for sign, label in ((+1, "tension"), (-1, "compression")):
        t0 = time.perf_counter()
        md = fresh()
        md.fix(bot, (2,), 0.0)
        e, s = test_curve(md, top, sign)
        res[f"E_{label}"] = float(np.polyfit(e, s, 1)[0]) / IN718_EP.E
        res[f"curve_{label}"] = [e.tolist(), s.tolist()]
        print(f"{topo} rho={rho} {label:16s}: E/E0 = {res[f'E_{label}']:.5f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    # as-built: one layer-by-layer build, then both tests from copies of that state
    t0 = time.perf_counter()
    md = fresh()
    am = AMProcess(md, layer_height=layer_voxels * h, inherent_strain=eps_star, n_inc=2, solver="auto")
    am.build(verbose=verbose)
    md.release(am.plate_nodes, (0, 1))          # cut the in-plane weld, keep u_z = 0
    md.solve(solver="auto", name="release")
    vm = np.asarray(tf.fem.constitutive.von_mises(md.stress().mean(axis=1)))
    _, alpha = md.plastic_strain()
    res["rs_vm_p50"], res["rs_vm_p95"] = float(np.percentile(vm, 50)), float(np.percentile(vm, 95))
    res["rs_plastic_fraction"] = float((alpha.mean(axis=1) > 0).mean())
    res["build_seconds"] = time.perf_counter() - t0
    built = md.save_state()
    for sign, label in ((+1, "tension"), (-1, "compression")):
        t0 = time.perf_counter()
        md.load_state(built)
        e, s = test_curve(md, top, sign)
        res[f"E_{label}_RS"] = float(np.polyfit(e, s, 1)[0]) / IN718_EP.E
        res[f"curve_{label}_RS"] = [e.tolist(), s.tolist()]
        print(f"{topo} rho={rho} {label + '_RS':16s}: E/E0 = {res[f'E_{label}_RS']:.5f}  "
              f"({time.perf_counter() - t0:.0f}s, build {res['build_seconds']:.0f}s)", flush=True)
    for label in ("tension", "compression"):
        res[f"reduction_{label}"] = 1 - res[f"E_{label}_RS"] / res[f"E_{label}"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", default="P-S,IWP-S")
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=24, help="voxels per 10 mm cell (24 -> 0.42 mm ~ the paper's 0.4 mm AM layer)")
    ap.add_argument("--layer-voxels", type=int, default=2)
    ap.add_argument("--eps", default="-0.005,-0.005,-0.0025")
    ap.add_argument("--out", default="results/literature/ahmed_barsoum_2022_rs.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    eps = tuple(float(v) for v in a.eps.split(",")) + (0.0, 0.0, 0.0)
    print(tf.device_info(), flush=True)
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}     # resumable
    for topo in a.topologies.split(","):
        if topo in allres:
            print(f"{topo}: cached", flush=True)
            continue
        allres[topo] = run(topo, a.rho, a.n, a.layer_voxels, eps, a.verbose)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(allres, open(a.out, "w"), indent=1)
        r = allres[topo]
        print(f"==> {topo}: modulus reduction from residual stress: tension {100 * r['reduction_tension']:.1f} %, "
              f"compression {100 * r['reduction_compression']:.1f} % (RS von Mises p95 {r['rs_vm_p95']:.0f} MPa, "
              f"{100 * r['rs_plastic_fraction']:.0f} % of elements yielded)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(allres, open(a.out, "w"), indent=1)
