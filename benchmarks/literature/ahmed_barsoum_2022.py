"""Benchmark against Ahmed, Barsoum & Abu Al-Rub, Metals 12 (2022) 1344.

Reproduces the *reference* (no residual stress) effective Young's modulus of
IN718 TPMS lattices -- P-S, G-S, IWP-S (sheets) and IWP-L (ligament) at
relative densities 10/20/30 % -- in the paper's set-up: 3 x 3 x 3 cells of
10 mm, loaded along z, and compares the Gibson-Ashby fits with their Table 3.

Two configurations:

* ``finite``   -- the paper's 30 mm specimen: bottom face u_z = 0 (symmetry),
  top face displaced rigidly in z (remote point, Ansys default "rigid"
  behaviour => u_x = u_y = 0 on the top face). P-S / IWP-S / IWP-L are
  mirror-symmetric about x = y = 15 mm and are run as quarter models; the
  chiral gyroid is run as a full model.
* ``periodic`` -- one unit cell with periodic boundary conditions (infinite
  lattice), to compare with the periodic-cell literature values that Table 3
  also lists (Lee et al. 2017 for P-S; Al-Ketan et al. 2018 for G-S, IWP).

Each case is solved on three voxel resolutions; the reported value is the
Richardson extrapolation (with GCI) of the three -- see common.richardson.

    python benchmarks/literature/ahmed_barsoum_2022.py --config periodic
    python benchmarks/literature/ahmed_barsoum_2022.py --config finite
    python benchmarks/literature/ahmed_barsoum_2022.py --report
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ResultStore, power_fit, richardson  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.homogenize import homogenize  # noqa: E402
from tpmsfem.geometry import TPMS, box  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402

HERE = os.path.dirname(__file__)
REF = json.load(open(os.path.join(HERE, "data", "ahmed_barsoum_2022.json")))
IN718 = tf.Material("IN718", 165_000.0, 0.3)
L = 10.0
TOPOLOGIES = {
    #        surface,   kind,      side     mirror-symmetric about cell mid-planes
    "P-S":   ("schwarz_p", "sheet", "lower", True),
    "G-S":   ("gyroid", "sheet", "lower", False),
    "IWP-S": ("iwp", "sheet", "lower", True),
    # ligament I-WP: the body-centred "I" labyrinth (f > c), struts along <111>
    "IWP-L": ("iwp", "network", "upper", True),
}
# voxels per 10 mm cell, chosen so the finest grid has >= ~3.5 voxels across the
# thinnest wall while keeping the CPU memory of the AMG solver below ~10 GB
RES_FINITE = {0.10: (40, 60, 90), 0.20: (28, 42, 63), 0.30: (24, 36, 54)}
RES_FINITE_FULL = {0.10: (32, 48, 72), 0.20: (24, 36, 54), 0.30: (20, 30, 45)}   # gyroid (no symmetry)
RES_PERIODIC = {0.10: (48, 72, 108), 0.20: (40, 60, 90), 0.30: (32, 48, 72)}


ORIGIN_SHIFT = float(os.environ.get("AB22_ORIGIN_SHIFT", "0"))   # fraction of L (phase sensitivity)


def lattice(topo, rho):
    surf, kind, side, _ = TOPOLOGIES[topo]
    o = ORIGIN_SHIFT * L
    return TPMS(surf, cell_size=L, kind=kind, density=rho, side=side, origin=(o, o, o))


def run_finite(topo, rho, n, solver):
    lat = lattice(topo, rho)
    sym = TOPOLOGIES[topo][3]
    h = L / n
    X = 15.0 if sym else 30.0
    mesh = voxelize(lat & box((0, 0, 0), (30, 30, 30)), (0, 0, 0), (X, X, 30.0), h)
    md = tf.SolidModel(mesh, IN718)
    x, y, z = mesh.points.T
    bot = np.flatnonzero(z < 1e-9)
    top = np.flatnonzero(z > 30 - 1e-9)
    md.fix(bot, (2,), 0.0)                       # symmetric BC on the base plane
    delta = -0.03                                # 0.1 % strain (linear => slope = modulus)
    md.fix(top, (0, 1), 0.0)                     # rigid remote point: no in-plane motion
    md.fix(top, (2,), delta)
    if sym:
        md.fix(np.flatnonzero(x > 15 - 1e-9), (0,), 0.0)
        md.fix(np.flatnonzero(y > 15 - 1e-9), (1,), 0.0)
    if solver == "auto" and md.ndof > 2_200_000:
        solver = "pcg"          # assembled AMG would exceed ~10 GB on the CPU; matrix-free Jacobi-PCG
    info = md.solve(solver=solver)
    F = md.reactions()[top, 2].sum() * (4 if sym else 1)
    E = (F / 900.0) / (delta / 30.0)
    return dict(E_rel=E / IN718.E, vf=mesh.n_cells * h**3 / (X * X * 30), elements=mesh.n_cells,
                dofs=md.ndof, seconds=info.seconds, krylov=info.krylov_iterations, solver=info.solver)


def run_periodic(topo, rho, n, solver):
    lat = lattice(topo, rho)
    mesh = voxelize(lat, (0, 0, 0), (L, L, L), L / n, periodic=(True, True, True))
    md = tf.SolidModel(mesh, IN718)
    t0 = time.perf_counter()
    # cubic symmetry of these TPMS: the normal block + one shear term give E, nu and Zener
    r = homogenize(md, components=(0, 1, 2, 3), solver=solver)
    return dict(E_rel=r.young("zz") / IN718.E, nu=r.poisson("zz", "xx"), zener=r.zener(),
                C=r.C.tolist(), vf=r.volume_fraction, elements=mesh.n_cells, dofs=md.ndof,
                seconds=time.perf_counter() - t0, krylov=sum(i.krylov_iterations for i in r.infos))


def compute(config, store, topos, rhos, solver):
    for topo in topos:
        for rho in rhos:
            if config == "finite":
                res = (RES_FINITE if TOPOLOGIES[topo][3] else RES_FINITE_FULL)[rho]
            else:
                res = RES_PERIODIC[rho]
            for n in res:
                key = f"{config}|{topo}|{rho:.2f}|{n}" + (f"|shift{ORIGIN_SHIFT}" if ORIGIN_SHIFT else "")
                if store.get(key):
                    continue
                t0 = time.perf_counter()
                out = (run_finite if config == "finite" else run_periodic)(topo, rho, n, solver)
                out["wall_s"] = time.perf_counter() - t0
                store.put(key, out)
                print(f"{config:8s} {topo:6s} rho={rho:.2f} n={n:3d}: E/E0={out['E_rel']:.5f} vf={out['vf']:.3f} "
                      f"[{out['elements']:,} el, {out['wall_s']:.0f}s]", flush=True)


def summarize(store, config, topos, rhos):
    rows = {}
    for topo in topos:
        for rho in rhos:
            if config == "finite":
                res = (RES_FINITE if TOPOLOGIES[topo][3] else RES_FINITE_FULL)[rho]
            else:
                res = RES_PERIODIC[rho]
            vals = [store.get(f"{config}|{topo}|{rho:.2f}|{n}") for n in res]
            if any(v is None for v in vals):
                continue
            h = [L / n for n in res][::-1]                       # fine -> coarse
            f = [v["E_rel"] for v in vals][::-1]
            rows[(topo, rho)] = dict(fine=f[0], rich=richardson(h, f), vf=vals[-1]["vf"],
                                     elements=vals[-1]["elements"],
                                     grids=" / ".join(f"{v['E_rel']:.4f}" for v in vals))
    return rows


def report(store, topos, rhos):
    lines = []
    for config in ("finite", "periodic"):
        rows = summarize(store, config, topos, rhos)
        if not rows:
            continue
        lines.append(f"\n### {'3x3x3 specimen (paper set-up)' if config == 'finite' else 'periodic unit cell'}\n")
        lines.append("| topology | rho | E/E0 on 3 grids | E/E0 best | uncertainty | published | diff |")
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        fits = {}
        for topo in topos:
            t3 = REF["table3_gibson_ashby_E_over_E0"][topo]
            for rho in rhos:
                r = rows.get((topo, rho))
                if r is None:
                    continue
                best = r["rich"]["f_ext"] if r["rich"]["monotone"] else r["fine"]
                if config == "finite":
                    pub = np.mean([t3["tension"][0] * rho ** t3["tension"][1],
                                   t3["compression"][0] * rho ** t3["compression"][1]])
                else:
                    pub = t3["literature"][0] * rho ** t3["literature"][1]
                fits.setdefault(topo, []).append((rho, best))
                unc = (f"GCI {100 * r['rich']['gci']:.1f} %" if r["rich"]["monotone"]
                       else f"spread {100 * r['rich']['spread']:.1f} %")
                lines.append(f"| {topo} | {rho:.2f} | {r['grids']} | {best:.4f} | {unc} | {pub:.4f} | "
                             f"{100 * (best / pub - 1):+.1f} % |")
        lines.append("\n| topology | C1 (ours) | n (ours) | C1, n published |")
        lines.append("|---|---:|---:|---|")
        for topo, pts in fits.items():
            if len(pts) < 3:
                continue
            C1, n = power_fit([p[0] for p in pts], [p[1] for p in pts])
            t3 = REF["table3_gibson_ashby_E_over_E0"][topo]
            pub = (f"tension {t3['tension'][0]}, {t3['tension'][1]}; compression {t3['compression'][0]}, {t3['compression'][1]}"
                   if config == "finite" else f"{t3['literature'][0]}, {t3['literature'][1]} ({t3['literature_ref']})")
            lines.append(f"| {topo} | {C1:.3f} | {n:.2f} | {pub} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["finite", "periodic"])
    ap.add_argument("--topologies", default="P-S,G-S,IWP-S,IWP-L")
    ap.add_argument("--densities", default="0.1,0.2,0.3")
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--store-dir", default="results/literature")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    topos = a.topologies.split(",")
    rhos = [float(r) for r in a.densities.split(",")]
    stores = {c: ResultStore(os.path.join(a.store_dir, f"ahmed_barsoum_2022_{c}.json")) for c in ("finite", "periodic")}
    print(tf.device_info(), flush=True)
    if a.config:
        compute(a.config, stores[a.config], topos, rhos, a.solver)
    if a.report or not a.config:
        merged = ResultStore(os.path.join(a.store_dir, "ahmed_barsoum_2022_finite.json"))
        merged.data = {**stores["finite"].data, **stores["periodic"].data}
        txt = report(merged, topos, rhos)
        print(txt)
        with open(os.path.join(a.store_dir, "ahmed_barsoum_2022.md"), "w") as fh:
            fh.write(txt + "\n")
