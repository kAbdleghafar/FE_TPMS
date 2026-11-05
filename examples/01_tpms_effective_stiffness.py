"""Effective compressive stiffness of TPMS lattices (unit-cell study).

For each surface/density a cube of n x n x n cells between two thin solid
plates is compressed by a prescribed displacement; E_eff = sigma / strain from
the reaction force. Fits the Gibson-Ashby scaling E_eff/E_s = C * rho^n.

Also runs a mesh-convergence study and reports Roache's GCI -- do this for any
resolution you use in production.

    python examples/01_tpms_effective_stiffness.py --h 0.1 --cells 2
"""
import argparse
import json
import os
import time

import numpy as np

import tpmsfem as tf
import tpmsfem.post  # noqa: F401
from tpmsfem.bench.metrics import gci
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

p = argparse.ArgumentParser()
p.add_argument("--cell", type=float, default=4.0, help="unit cell size [mm]")
p.add_argument("--cells", type=int, default=2, help="cells per edge")
p.add_argument("--h", type=float, default=0.1, help="voxel size [mm]")
p.add_argument("--ratio", type=float, default=1.5, help="refinement ratio of the convergence study")
p.add_argument("--surfaces", default="gyroid,diamond,schwarz_p")
p.add_argument("--densities", default="0.2,0.3,0.4")
p.add_argument("--solver", default="auto")
p.add_argument("--out", default="results/01")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
print(tf.device_info())

from tpmsfem.fem.materials import TI6AL4V_LPBF as TI  # noqa: E402

size = a.cell * a.cells
plate = max(0.5, 2 * a.h * a.ratio**2)   # physical platen thickness, resolved on all grids
strain = 1e-3


def wall_thickness(surface, rho, n=96):
    """Approximate sheet thickness = rho / (TPMS surface area per unit volume)."""
    from skimage import measure

    from tpmsfem.geometry.tpms import SURFACES
    g = np.arange(n + 1) / n * 2 * np.pi
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    v, f, _, _ = measure.marching_cubes(SURFACES[surface](X, Y, Z), 0.0, spacing=(a.cell / n,) * 3)
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    return rho * a.cell**3 / area


def effective_modulus(surface, rho, h, save=None):
    h = size / max(1, round(size / h))          # commensurate grid: nodes on z = 0 and z = size
    lat = TPMS(surface, cell_size=a.cell, kind="sheet", density=rho)
    mesh = voxelize(lattice_cube(lat, size, plate=plate), (0, 0, 0), (size,) * 3, h)
    md = tf.SolidModel(mesh, TI.elastic())
    z = mesh.points[:, 2]
    bot, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > size - 1e-9)
    md.fix(bot)
    md.fix(top, (0, 1), 0.0)
    md.fix(top, (2,), -strain * size)
    info = md.solve(solver=a.solver)
    F = -md.reactions()[top, 2].sum()
    # lattice region only: subtract the (stiff, thin) plates in series
    E_eff = (F / size**2) / strain
    vol_frac = mesh.n_cells * h**3 / size**3
    if save:
        tf.post.save_vtu(md, save)
    return E_eff, vol_frac, mesh, info


rows = []
for surf in a.surfaces.split(","):
    for rho in map(float, a.densities.split(",")):
        t0 = time.perf_counter()
        E, vf, mesh, info = effective_modulus(surf, rho, a.h, save=os.path.join(a.out, f"{surf}_{rho:.2f}.vtu"))
        rows.append(dict(surface=surf, rho=rho, voxel_fraction=vf, E_eff=E, E_rel=E / TI.E,
                         n_cells=mesh.n_cells, dofs=3 * mesh.n_points, krylov=info.krylov_iterations,
                         seconds=time.perf_counter() - t0))
        t = wall_thickness(surf, rho)
        rows[-1]["wall_mm"] = t
        print(f"{surf:10s} rho={rho:.2f} wall~{t:.2f} mm ({t / a.h:.1f} voxels) E_eff={E:9.1f} MPa  "
              f"E/Es={E / TI.E:.4f}  [{mesh.n_cells:,} el, {info.solver}, {time.perf_counter() - t0:.1f}s]")

print("\nGibson-Ashby fit  E/Es = C rho^n  (rho includes the plates' share; use for trends)")
for surf in a.surfaces.split(","):
    r = [x for x in rows if x["surface"] == surf]
    n, logC = np.polyfit(np.log([x["rho"] for x in r]), np.log([x["E_rel"] for x in r]), 1)
    print(f"  {surf:10s} C={np.exp(logC):.3f} n={n:.2f}")

r = a.ratio
hs = [a.h, r * a.h, r * r * a.h]
t = wall_thickness("gyroid", 0.3)
print(f"\nMesh convergence (gyroid, rho=0.3, wall~{t:.2f} mm): voxel h = " + ", ".join(f"{h:.3f}" for h in hs)
      + "  (" + ", ".join(f"{t / h:.1f}" for h in hs) + " voxels across the wall)")
Es = [effective_modulus("gyroid", 0.3, h)[0] for h in hs]
g = gci(Es[0], Es[1], Es[2], r=r)
print(f"  E(h)={Es[0]:.1f}, E(r*h)={Es[1]:.1f}, E(r^2*h)={Es[2]:.1f} -> observed order p={g['p']:.2f}, "
      f"extrapolated {g['f_exact']:.1f}, GCI_fine={100 * g['gci_fine']:.2f} %")
json.dump(dict(rows=rows, convergence=dict(h=hs, E=Es, **g)), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
