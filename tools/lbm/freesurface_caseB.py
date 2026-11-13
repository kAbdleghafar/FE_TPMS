"""AM-Bench 2018-02 case B (IN625 plate, 179.2 W, 0.8 m/s) with the free-surface lattice
Boltzmann melt pool: Marangoni flow, recoil pressure, evaporation and radiation.

Variants (same absorptivity and absorption depth as the calibrated conduction model):
  flat        flow on, but no Marangoni stress (surface tension, recoil, evaporation only)
  marangoni   plus Marangoni stress, dsigma/dT = -1.1e-4 N/(m K) (IN718-type, Mills 2002)

    python tools/lbm/freesurface_caseB.py marangoni [cell_um]
Measured (NIST AMMT): width 123.5 +- 6.5 um, depth 36.0 +- 1.9 um, length 359 +- 20 um.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "literature"))
import in625_material as IM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, ScanVector  # noqa: E402
from tpmsfem.process.freesurface import FreeSurfaceMeltPool  # noqa: E402

OUT = "tools/lbm/caseB_freesurface_physics.json"
P, V, W_BEAM, A, DELTA = 179.2, 0.8, 85e-6, 0.3234, 15.3e-6
MEASURED = dict(width=(123.5, 6.5), depth=(36.0, 1.9), length=(359.0, 20.0))
LX, LY, LZ, GAS_UM = 1.0e-3, 0.3e-3, 0.2e-3, 60e-6
X0, X1 = 0.15e-3, 0.85e-3


def fusion_zone(res, nz_metal, h, x_sec):
    """Width and depth [um] of the region whose peak temperature reached the solidus, in the
    cross-section at x_sec; depth below the original surface."""
    i = int(round(x_sec / h)) + 1
    Tp = res.T_peak[i]                                       # (ny, nz)
    melted = Tp >= res_solidus
    cols = np.nonzero(melted.any(axis=1))[0]
    width = (cols.max() - cols.min() + 1) * h * 1e6 if len(cols) else 0.0
    rows = np.nonzero(melted.any(axis=0))[0]
    depth = (nz_metal + 1 - rows.min()) * h * 1e6 if len(rows) else 0.0
    return float(width), float(depth)


def run(variant, cell_um=10.0, dt=2e-7, mu=5e-3, sigma=1.8, dsdT=-1.1e-4):
    global res_solidus
    h = cell_um * 1e-6
    nx, ny, nz = int(round(LX / h)), int(round(LY / h)), int(round(LZ / h))
    ng = int(round(GAS_UM / h))
    shape = (nx + 2, ny + 2, nz + ng + 2)
    metal = np.zeros(shape, bool)
    metal[1:-1, 1:-1, 1:nz + 1] = True
    mat = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
    md = FreeSurfaceMeltPool(mat, h, metal, dt=dt, viscosity=mu, sigma=sigma, mushy_C=1e9, T0=25.0, T_max=6000.0,
                             dsigma_dT=dsdT if variant == "marangoni" else 0.0, evaporation=Evaporation(),
                             emissivity=0.4)
    res_solidus = md.Ts
    y = h + 0.5 * LY
    res = md.run(Laser(P, A, W_BEAM, depth=DELTA), [ScanVector(h + X0, y, h + X1, y, V)], t_extra=60e-6, verbose=True,
                 snapshot_every=0)
    x_sec = X0 + 0.75 * (X1 - X0)
    w, d = fusion_zone(res, nz, h, x_sec)
    surf = (res.surface[1:-1, 1:-1] - (nz + 1) * h) * 1e6
    out = dict(variant=variant, cell_um=cell_um, dt=dt, tau=md.tau, sigma_lattice=md.sigma_lat, width_um=w, depth_um=d,
               T_peak_max=float(res.T_peak.max()), u_max_m_s=res.u_max, surface_um=[float(surf.min()), float(surf.max())],
               mass_error=res.mass_error, energy_balance=res.energy_balance, steps=res.steps, seconds=res.seconds,
               mlups=res.mlups, measured=MEASURED)
    print(json.dumps(out, indent=1), flush=True)
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    store[f"{variant}|{cell_um:g}um"] = out
    json.dump(store, open(OUT, "w"), indent=1)
    np.savez_compressed(OUT.replace(".json", f"_{variant}_{cell_um:g}um.npz"), T_peak=res.T_peak.astype(np.float32),
                        T=res.T.astype(np.float32), fill=res.fill.astype(np.float32), flag=res.flag,
                        surface=res.surface.astype(np.float32))
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "marangoni", float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
