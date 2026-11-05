"""Benchmark against Van Viet, Zaki & El-Rich, PLoS One 20 (2025) e0332104.

Sectionally nonlinearly functionally graded (SNFG) gyroid cells for cranial
and femoral Ti implants: the sheet thickness parameter t varies along z
(Eqs. 1-2, Table 1). The effective Young's modulus perpendicular to the
grading (y) is computed by homogenisation with periodic BCs in x, y and
traction-free faces normal to z, as described in the paper, and compared with
the values stated in its text (Fig. 8 discussion).

Assumptions (not stated in the paper): one gyroid period per cell in each
direction, gyroid phase starting at z = 0, sheet solid where |f| <= t(z)
(t_max = 1.5 = max|f| of the gyroid gives a fully dense end, as the paper's
cortical end requires).

    python benchmarks/literature/van_viet_elrich_2025.py           # compute + report
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ResultStore, richardson  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.backend import array_module  # noqa: E402
from tpmsfem.fem.homogenize import homogenize  # noqa: E402
from tpmsfem.geometry.fields import Field  # noqa: E402
from tpmsfem.geometry.tpms import gyroid  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402

HERE = os.path.dirname(__file__)
REF = json.load(open(os.path.join(HERE, "data", "van_viet_elrich_2025.json")))
TI = tf.Material("Ti", 104_000.0, 0.3)
L = 10.0
T = REF["gyroid_thickness_parameters_table1"]
RES = (40, 60, 90)


def t_cranial(z, g0):
    xp = array_module(z)
    s = xp.abs((z - L / 2) / (L / 2))
    return (T["tmax"] - T["tmin"]) * s**g0 + T["tmin"]


def t_femur(z, gc, gt):
    xp = array_module(z)
    zc = xp.clip((z - L / 2) / (L / 2), 0.0, 1.0)
    zt = xp.clip(z / (L / 2), 0.0, 1.0)
    cort = (T["tmax"] - T["tminc_eq_tmaxt"]) * zc**gc + T["tminc_eq_tmaxt"]
    trab = (T["tminc_eq_tmaxt"] - T["tmin"]) * zt**gt + T["tmin"]
    return xp.where(z > L / 2, cort, trab)


def graded_gyroid(tfun):
    k = 2 * np.pi / L

    def fn(x, y, z):
        xp = array_module(x)
        return xp.abs(gyroid(k * x, k * y, k * z)) - tfun(z)

    return Field(fn, "SNFG-gyroid")


CASES = {f"cranial g0={g}": (lambda g: (lambda z: t_cranial(z, g)))(g) for g in (2, 3, 4, 5, 7)}
CASES.update({f"femur gc={g} gt=2": (lambda g: (lambda z: t_femur(z, g, 2)))(g) for g in (0.5, 1, 3, 7)})
PUBLISHED = {"cranial g0=2": 25.3, "cranial g0=7": 11.5, "femur gc=0.5 gt=2": 35.2, "femur gc=7 gt=2": 14.0}


def run(name, n, solver):
    mesh = voxelize(graded_gyroid(CASES[name]), (0, 0, 0), (L, L, L), L / n, periodic=(True, True, False))
    md = tf.SolidModel(mesh, TI)
    t0 = time.perf_counter()
    r = homogenize(md, components=(0, 1, 5), solver=solver)
    return dict(E_x=r.young("xx") / 1000, E_y=r.young("yy") / 1000, G_xy=r.shear("xy") / 1000,
                nu_yx=r.poisson("yy", "xx"), vf=r.volume_fraction, elements=mesh.n_cells,
                dofs=md.ndof, seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--store", default="results/literature/van_viet_elrich_2025.json")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    store = ResultStore(a.store)
    print(tf.device_info(), flush=True)
    if not a.report_only:
        for name in CASES:
            for n in RES:
                key = f"{name}|{n}"
                if store.get(key):
                    continue
                out = run(name, n, a.solver)
                store.put(key, out)
                print(f"{name:20s} n={n:3d}: E_y={out['E_y']:.2f} GPa E_x={out['E_x']:.2f} vf={out['vf']:.3f} "
                      f"[{out['elements']:,} el, {out['seconds']:.0f}s]", flush=True)
    lines = ["| case | vol. fraction | E_y on 3 grids (n=40/60/90) [GPa] | E_y best [GPa] | uncertainty | published [GPa] | diff |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for name in CASES:
        vals = [store.get(f"{name}|{n}") for n in RES]
        if any(v is None for v in vals):
            continue
        h = [L / n for n in RES][::-1]
        f = [v["E_y"] for v in vals][::-1]
        rr = richardson(h, f)
        best = rr["f_ext"] if rr["monotone"] else f[0]
        pub = PUBLISHED.get(name)
        unc = (f"GCI {100 * rr['gci']:.1f} %" if rr["monotone"] else f"spread {100 * rr['spread']:.1f} %")
        grids = " / ".join(f"{v['E_y']:.2f}" for v in vals)
        lines.append(f"| {name} | {vals[-1]['vf']:.3f} | {grids} | {best:.2f} | {unc} | "
                     + (f"{pub:.1f} | {100 * (best / pub - 1):+.1f} % |" if pub else "- | - |"))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.splitext(a.store)[0] + ".md", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
