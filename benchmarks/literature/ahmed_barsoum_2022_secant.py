"""How much does the modulus depend on *where* the slope is taken?

Ahmed et al. (2022) extract E "from the slope of the stress-strain curves in
the elastic region" of an elasto-plastic run to 10 % strain. With IN718 and
the stress concentrations of thin TPMS walls, local yielding starts at small
macroscopic strains, so a slope taken over the first few tenths of a percent
is lower than the true initial modulus. This script computes the
elasto-plastic z-compression curve of the 3x3x3 specimen (quarter model) and
reports the initial tangent and secant moduli at several strains, to compare
with the published values.

    python benchmarks/literature/ahmed_barsoum_2022_secant.py [--n 36]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ahmed_barsoum_2022 as B  # noqa: E402
from ahmed_barsoum_2022_rs import IN718_EP  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.geometry import box  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402

STRAINS = [0.0002, 0.001, 0.0025, 0.005, 0.01]


def curve(topo, rho, n):
    mesh = voxelize(B.lattice(topo, rho) & box((0, 0, 0), (30, 30, 30)), (0, 0, 0), (15, 15, 30), B.L / n)
    md = tf.SolidModel(mesh, IN718_EP)
    x, y, z = mesh.points.T
    top = np.flatnonzero(z > 30 - 1e-9)
    md.fix(np.flatnonzero(x > 15 - 1e-9), (0,), 0.0)
    md.fix(np.flatnonzero(y > 15 - 1e-9), (1,), 0.0)
    md.fix(np.flatnonzero(z < 1e-9), (2,), 0.0)
    md.fix(top, (0, 1), 0.0)
    out = []
    for eps in STRAINS:
        md.fix(top, (2,), -eps * 30.0)
        md.solve(n_inc=2, solver="auto", name=f"{eps}")
        sig = -md.reactions()[top, 2].sum() * 4 / 900.0
        _, alpha = md.plastic_strain()
        out.append((eps, sig, float((alpha.max(axis=1) > 0).mean())))
        print(f"  {topo} rho={rho} eps={100 * eps:.2f} %: sigma={sig:.2f} MPa, yielded elements "
              f"{100 * out[-1][2]:.1f} %", flush=True)
    return mesh.n_cells, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topologies", default="P-S,IWP-S")
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=24, help="voxels per cell; the secant/initial RATIO is what is used")
    a = ap.parse_args()
    out = "results/literature/ahmed_barsoum_2022_secant.json"
    res = json.load(open(out)) if os.path.exists(out) else {}          # resumable
    for topo in a.topologies.split(","):
        key = f"{topo}|rho={a.rho:.2f}"
        if key in res:
            print(f"{key}: cached", flush=True)
            continue
        t0 = time.perf_counter()
        ne, pts = curve(topo, a.rho, a.n)
        e = np.array([p[0] for p in pts])
        s = np.array([p[1] for p in pts])
        E0 = s[0] / e[0] / B.IN718.E
        secant = {f"{100 * ei:.2f}%": si / ei / B.IN718.E for ei, si in zip(e, s)}
        t3 = B.REF["table3_gibson_ashby_E_over_E0"][topo]
        pub = np.mean([t3["tension"][0] * a.rho ** t3["tension"][1], t3["compression"][0] * a.rho ** t3["compression"][1]])
        ratio = {k: v / E0 for k, v in secant.items()}
        res[key] = dict(topology=topo, rho=a.rho, n=a.n, elements=ne, initial=E0, secant=secant, secant_over_initial=ratio,
                         published=pub,
                         curve=[list(p) for p in pts], seconds=time.perf_counter() - t0)
        print(f"==> {topo}: initial E/E0 {E0:.4f}; secant/initial " +
              ", ".join(f"{k}: {v:.3f}" for k, v in ratio.items()) + f"; published E/E0 {pub:.4f}", flush=True)
        os.makedirs("results/literature", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
