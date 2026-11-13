"""AM-Bench case-B track: the free-surface LBM melt pool (tpmsfem.process.freesurface) against
the conduction model (MeltPoolModel) on the same problem as compare_openlb_track.py:
constant properties, no surface losses, the same moving Gaussian source with absorption depth.

    python tools/lbm/compare_freesurface_track.py frozen     # heat only (flow frozen): solver check
    python tools/lbm/compare_freesurface_track.py flow       # with the melt flowing (stage 1: no Marangoni)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_openlb_track import CP, DELTA, K, LAT, LX, LY, LZ, RHO, T0, TL, TS, A, P, V, W, X0, X1, pool  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.materials import ThermalProperties  # noqa: E402
from tpmsfem.process import Laser, MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process.freesurface import FreeSurfaceMeltPool  # noqa: E402

OUT = "tools/lbm/caseB_freesurface_comparison.json"


def material():
    th = ThermalProperties.from_si([0.0, 20000.0], [K, K], [CP, CP], RHO, LAT, TS, TL)
    return tf.Material("const", 200_000.0, 0.3, thermal=th)


def run(mode, res_um=5.0, dt=None, gas_cells=6, viscosity=5e-3, sigma=1.8, mushy_C=1e9):
    h = res_um * 1e-6
    nx, ny, nz = int(round(LX * 1e-3 / h)), int(round(LY * 1e-3 / h)), int(round(LZ * 1e-3 / h))
    shape = (nx + 2, ny + 2, nz + 2 + gas_cells)
    metal = np.zeros(shape, bool)
    metal[1:-1, 1:-1, 1:nz + 1] = True
    frozen = mode == "frozen"
    dt = dt or (5e-7 if frozen else 1e-7)
    md = FreeSurfaceMeltPool(material(), h, metal, dt=dt, viscosity=viscosity, sigma=sigma, mushy_C=mushy_C,
                             T0=T0, T_max=20000.0)
    y_track = h + 0.5 * LY * 1e-3
    path = [ScanVector(h + X0 * 1e-3, y_track, h + X1 * 1e-3, y_track, V * 1e-3)]
    res = md.run(Laser(P, A, W * 1e-3, depth=DELTA * 1e-3), path, frozen_flow=frozen, verbose=True)
    T_lbm = res.T[1:-1, 1:-1, 1:nz + 1]
    # conduction model
    th_mat = material()
    mp = MeltPoolModel(th_mat, h * 1e3, length=LX, width=LY, depth=LZ, symmetric_y=False, T0=T0, emissivity=0.0,
                       h_conv=0.0, evaporation=None, T_max=20000.0)
    t0 = time.perf_counter()
    r2 = mp.run(Laser(P, A, W, depth=DELTA), [ScanVector(X0, 0.0, X1, 0.0, V)])
    T_fd = np.asarray(r2.T)
    hot = (T_fd > 500.0) | (T_lbm > 500.0)
    d = T_lbm - T_fd
    out = dict(mode=mode, cell_um=res_um, dt_s=dt, tau=md.tau, sigma_lattice=md.sigma_lat, cells=int(np.prod(shape)),
               lbm=dict(pool_um=pool(T_lbm, h * 1e3), T_max=float(T_lbm.max()), u_max_m_s=res.u_max,
                        mass_error=res.mass_error, energy_balance=res.energy_balance, steps=res.steps,
                        seconds=res.seconds, mlups=res.mlups,
                        surface_um=[float((res.surface.min() - h) * 1e6), float((res.surface.max() - h) * 1e6)]),
               conduction=dict(pool_um=pool(T_fd, h * 1e3), T_max=float(T_fd.max()), seconds=time.perf_counter() - t0),
               difference=dict(rms_K_where_hot=float(np.sqrt(np.mean(d[hot] ** 2))), max_abs_K=float(np.abs(d).max()),
                               rel_rms=float(np.sqrt(np.mean(d[hot] ** 2)) / np.mean(T_fd[hot] - T0))))
    print(json.dumps(out, indent=1))
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    store[mode] = out
    json.dump(store, open(OUT, "w"), indent=1)
    np.savez_compressed(OUT.replace(".json", f"_{mode}.npz"), T_lbm=T_lbm.astype(np.float32), T_fd=T_fd.astype(np.float32),
                        fill=res.fill.astype(np.float32), surface=res.surface.astype(np.float32))
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "frozen")
