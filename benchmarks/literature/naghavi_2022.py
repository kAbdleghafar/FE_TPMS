"""Benchmark against Naghavi, ..., Liu (UCL), Bioengineering 9 (2022) 504.

"Mechanical characterisation and numerical modelling of TPMS-based gyroid and
diamond Ti6Al4V scaffolds for bone implants", doi:10.3390/bioengineering9100504
(open access, PMC9598079).

Sheet gyroid (G600/G800/G1000) and sheet diamond (D800/D900/D1100) scaffolds,
0.3 mm walls, unit cells 1.8-2.8 mm, porosity 53-67 %, compressed as cylinders
(gyroid D 11.04 x H 16.56 mm, diamond D 14.05 x H 21.08 mm) between rigid tied
platens. Their FE: Abaqus C3D10, 0.065 mm elements (15-32 M elements); solid
modulus = their measured compressive bulk modulus E_s = 35.77 GPa (also used for
their Gibson-Ashby fit); nu = 0.3. Table 6 gives FE and experimental moduli.

tpmsfem:
1. ``periodic``: E_zz of the infinite lattice (one periodic cell, 3 grids,
   Richardson) for all six designs;
2. ``specimen``: the full cylinder with tied rigid platens on coarse grids,
   giving the finite-specimen factor F = E_specimen / E_periodic on the same
   grid (discretisation errors largely cancel in F);
3. prediction E = F x E_periodic(converged), compared with Table 6.

Assumptions: relative density = 1 - design porosity (Table 3); lattice origin
at the specimen's bounding-box corner; top platen moves axially only.

    python benchmarks/literature/naghavi_2022.py --config periodic
    python benchmarks/literature/naghavi_2022.py --config specimen
    python benchmarks/literature/naghavi_2022.py --report
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ResultStore, richardson  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.homogenize import homogenize  # noqa: E402
from tpmsfem.geometry import TPMS, cylinder  # noqa: E402
from tpmsfem.mesh import voxelize  # noqa: E402

TI64 = tf.Material("Ti6Al4V_bulk_compression", 35_770.0, 0.3)     # E_s used by the authors
SPEC = {"gyroid": dict(D=11.04, H=16.56), "diamond": dict(D=14.05, H=21.08)}
#                surface   cell [mm] porosity  E_FE   E_exp  [GPa]   (Tables 3 and 6)
DESIGNS = {
    "G600":  ("gyroid", 1.808, 0.540, 9.54, 8.46),
    "G800":  ("gyroid", 2.208, 0.620, 6.96, 6.81),
    "G1000": ("gyroid", 2.608, 0.674, 5.39, 5.69),
    "D800":  ("diamond", 2.195, 0.527, 10.78, 10.22),
    "D900":  ("diamond", 2.400, 0.564, 9.89, 9.37),
    "D1100": ("diamond", 2.811, 0.622, 8.06, 7.59),
}
RES_PERIODIC = (40, 60, 90)                   # voxels per unit cell
SPECIMEN_H = {"G800": (0.20, 0.15, 0.12), "D900": (0.24, 0.18, 0.144)}   # voxel sizes [mm]


def lattice(name, origin=(0, 0, 0)):
    surf, a, P, _, _ = DESIGNS[name]
    return TPMS(surf, cell_size=a, kind="sheet", density=1.0 - P, origin=origin)


def run_periodic(name, n, solver):
    a = DESIGNS[name][1]
    mesh = voxelize(lattice(name), (0, 0, 0), (a, a, a), a / n, periodic=(True, True, True))
    md = tf.SolidModel(mesh, TI64)
    r = homogenize(md, components=(0, 1, 2, 3), solver=solver)
    return dict(E=r.young("zz") / 1000, vf=r.volume_fraction, elements=mesh.n_cells)


def run_specimen(name, h, solver):
    surf = DESIGNS[name][0]
    D, H = SPEC[surf]["D"], SPEC[surf]["H"]
    R = D / 2
    H = round(H / h) * h                       # commensurate grid (platen node sets exact)
    lat = lattice(name, origin=(-R, -R, 0.0))
    body = cylinder((0, 0, -1.0), (0, 0, H + 1.0), R) & lat
    mesh = voxelize(body, (-R, -R, 0), (R, R, H), h)
    md = tf.SolidModel(mesh, TI64)
    z = mesh.points[:, 2]
    bot, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > H - 1e-9)
    md.fix(bot)                                # bottom platen: encastre (tied)
    md.fix(top, (0, 1), 0.0)                   # top platen: rigid, axial motion only
    delta = 1e-3 * H
    md.fix(top, (2,), -delta)
    if solver == "auto" and md.ndof > 2_500_000:
        solver = "pcg"
    info = md.solve(solver=solver)
    F = -md.reactions()[top, 2].sum()
    E = F * H / (np.pi * R**2 * delta)
    vf = mesh.n_cells * h**3 / (np.pi * R**2 * H)
    # periodic value on the SAME voxel size (for the finite-specimen factor)
    a = DESIGNS[name][1]
    n = max(8, round(a / h))
    per = run_periodic(name, n, solver if solver != "pcg" else "auto")
    return dict(E=E / 1000, vf=vf, elements=mesh.n_cells, E_periodic_same_h=per["E"], n_cell=n,
                factor=E / 1000 / per["E"], seconds=info.seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["periodic", "specimen"])
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    store = ResultStore("results/literature/naghavi_2022.json")
    print(tf.device_info(), flush=True)
    if a.config == "periodic":
        for name in a.designs.split(","):
            for n in RES_PERIODIC:
                key = f"periodic|{name}|{n}"
                if store.get(key) is None:
                    t0 = time.perf_counter()
                    store.put(key, run_periodic(name, n, a.solver))
                    print(f"periodic {name} n={n}: E={store.get(key)['E']:.3f} GPa vf={store.get(key)['vf']:.3f} "
                          f"[{store.get(key)['elements']:,} el, {time.perf_counter() - t0:.0f}s]", flush=True)
    if a.config == "specimen":
        for name, hs in SPECIMEN_H.items():
            for h in hs:
                key = f"specimen|{name}|{h}"
                if store.get(key) is None:
                    t0 = time.perf_counter()
                    store.put(key, run_specimen(name, h, a.solver))
                    r = store.get(key)
                    print(f"specimen {name} h={h}: E={r['E']:.3f} GPa (periodic same h {r['E_periodic_same_h']:.3f}, "
                          f"factor {r['factor']:.3f}) vf={r['vf']:.3f} [{r['elements']:,} el, {time.perf_counter() - t0:.0f}s]",
                          flush=True)
    if a.report or not a.config:
        report(store)


def periodic_best(store, name):
    surf, a, P, _, _ = DESIGNS[name]
    vals = [store.get(f"periodic|{name}|{n}") for n in RES_PERIODIC]
    if any(v is None for v in vals):
        return None, None
    rho = 1 - P
    corr = [v["E"] * (rho / v["vf"]) ** 1.4 for v in vals]      # density correction (local exponent ~1.4)
    rr = richardson([a / n for n in RES_PERIODIC][::-1], corr[::-1])
    return (rr["f_ext"] if rr["monotone"] else corr[-1]), corr


def report(store):
    """Finite-specimen factor F = E_specimen(finest grid) / E_periodic(converged) from the
    full cylinder models (G800, D900); the spread of the specimen values over the three
    grids is the uncertainty. Other designs: F of the same topology x converged periodic E."""
    factors, spec = {}, {}
    for name, hs in SPECIMEN_H.items():
        vals = [store.get(f"specimen|{name}|{h}") for h in hs]
        vals = [v["E"] for v in vals if v]
        Eper, _ = periodic_best(store, name)
        if vals and Eper:
            spec[name] = (vals[-1], (max(vals) - min(vals)) / vals[-1], vals)
            factors[DESIGNS[name][0]] = vals[-1] / Eper
    lines = ["| design | rho | E periodic, 3 grids [GPa] | E periodic best | specimen factor F | "
             "E predicted [GPa] | FE (Abaqus C3D10) | experiment | vs FE | vs experiment |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, (surf, a, P, Efe, Eexp) in DESIGNS.items():
        Eper, corr = periodic_best(store, name)
        if Eper is None:
            continue
        F = factors.get(surf, float("nan"))
        if name in spec:
            Ep, unc, vals = spec[name]
            how = f"**{Ep:.2f}** (cylinder model; grids {' / '.join(f'{v:.2f}' for v in vals)})"
        else:
            Ep, how = F * Eper, f"{F * Eper:.2f}"
        lines.append(f"| {name} | {1 - P:.3f} | {' / '.join(f'{c:.2f}' for c in corr)} | {Eper:.2f} | {F:.3f} | "
                     f"{how} | {Efe:.2f} | {Eexp:.2f} | {100 * (Ep / Efe - 1):+.1f} % | {100 * (Ep / Eexp - 1):+.1f} % |")
    txt = "\n".join(lines)
    print(txt)
    open("results/literature/naghavi_2022.md", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
