"""Performance scaling: matrix-free operator and solver time vs. problem size.

Runs a gyroid lattice block of growing size on the active backend and reports
the matrix-free K*v throughput, Jacobi-PCG and (CPU) AMG solve times, and peak
memory. Run it once per backend to get the CPU-vs-GPU comparison:

    TPMSFEM_BACKEND=numpy python benchmarks/perf_scaling.py --sizes 4,6,8
    TPMSFEM_BACKEND=cupy  python benchmarks/perf_scaling.py --sizes 4,8,12,16,20

Results are appended to results/perf_<backend>.json.
"""
import argparse
import json
import os
import resource
import time

import numpy as np

import tpmsfem as tf
from tpmsfem.backend import backend_name, on_gpu, synchronize
from tpmsfem.fem.materials import TI6AL4V_LPBF
from tpmsfem.geometry import TPMS, lattice_cube
from tpmsfem.mesh import voxelize

p = argparse.ArgumentParser()
p.add_argument("--sizes", default="4,6,8", help="block edge lengths [mm]")
p.add_argument("--h", type=float, default=0.1)
p.add_argument("--solvers", default="gmg,pcg,amg")
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
print(tf.device_info())


def gpu_mem():
    if not on_gpu():
        return None
    import cupy as cp

    return cp.get_default_memory_pool().total_bytes() / 2**20


rows = []
for L in map(float, a.sizes.split(",")):
    lat = TPMS("gyroid", cell_size=2.0, kind="sheet", density=0.35)
    t0 = time.perf_counter()
    mesh = voxelize(lattice_cube(lat, L, plate=0.4), (0, 0, 0), (L,) * 3, L / round(L / a.h))
    t_mesh = time.perf_counter() - t0
    for solver in a.solvers.split(","):
        if solver not in ("pcg", "gmg") and on_gpu():
            continue
        md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
        z = mesh.points[:, 2]
        md.fix(np.flatnonzero(z < 1e-9))
        md.fix(np.flatnonzero(z > L - 1e-9), (2,), -1e-3 * L)
        v = md.xp.ones(md.ndof)
        md.tangent(v)
        synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            md.tangent(v)
        synchronize()
        t_mv = (time.perf_counter() - t0) / n
        info = md.solve(solver=solver)
        synchronize()
        row = dict(backend=backend_name(), L=L, elements=md.ne, dofs=md.ndof, solver=solver,
                   t_mesh=t_mesh, t_matvec=t_mv, Melem_per_s=md.ne / t_mv / 1e6,
                   t_solve=info.seconds, iterations=info.krylov_iterations,
                   host_peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   gpu_pool_MiB=gpu_mem())
        rows.append(row)
        print(f"L={L:5.1f} {md.ne:>10,} el {md.ndof:>11,} dofs | K*v {1e3 * t_mv:8.2f} ms "
              f"({row['Melem_per_s']:.1f} Melem/s) | {solver}: {info.seconds:8.2f} s, {info.krylov_iterations} its")

path = os.path.join(a.out, f"perf_{backend_name()}.json")
json.dump(rows, open(path, "w"), indent=1)
print("wrote", path)
