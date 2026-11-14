"""AM-Bench 2018-02 single tracks A, B, C (IN625 plate) with the free-surface lattice Boltzmann
melt pool, for calibrating one absorptivity across all three cases.

The flow model carries Marangoni convection, recoil and evaporation itself, so the absorption
depth is kept physical and small (5 um) instead of being a fitting knob as in the conduction
model (15.3 um there).

    python tools/lbm/freesurface_ambench.py <case> <absorptivity> [cell_um] [dsigma_dT | S<activity>]

dsigma_dT: a constant [N/(m K)], or "S0.01" for the Sahoo model with sulfur activity 0.01
(surface-active sulfur: inward Marangoni flow below a crossover temperature, outward above;
Fe-S adsorption constants of Sahoo et al. 1988 with the slope of pure nickel, 3.8e-4).
Heat is carried with the lattice mass flux (energy_transport="lbm").
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "literature"))
import in625_material as IM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, ScanVector  # noqa: E402
from tpmsfem.process.freesurface import FreeSurfaceMeltPool, SahooSurfaceTension  # noqa: E402

OUT = "tools/lbm/ambench_freesurface.json"
CASES = {"A": dict(P=137.9, v=0.4, width=(147.9, 3.7), depth=(42.5, 1.7)),
         "B": dict(P=179.2, v=0.8, width=(123.5, 6.5), depth=(36.0, 1.9)),
         "C": dict(P=179.2, v=1.2, width=(106.0, 1.4), depth=(29.6, 0.6))}
W_BEAM, DELTA = 85e-6, 5e-6
LX, LY, LZ, GAS = 1.0e-3, 0.3e-3, 0.2e-3, 60e-6
X0, X1 = 0.15e-3, 0.85e-3


def pool_size(Tp, nz, h, Ts, x_sec):
    """Width at the surface and depth on the centreline [um], interpolated at the solidus."""
    i = int(round(x_sec / h)) + 1
    sec = Tp[i, 1:-1, 1:nz + 1]
    col = sec[sec.shape[0] // 2]
    z = (np.arange(nz) + 0.5) * h - nz * h
    k = np.nonzero(col >= Ts)[0].min()
    depth = -(z[k] - (col[k] - Ts) / (col[k] - col[k - 1]) * h) if k > 0 else nz * h
    surf = sec[:, -1]
    y = (np.arange(len(surf)) + 0.5) * h
    m = np.nonzero(surf >= Ts)[0]
    a, b = m.min(), m.max()
    yl = y[a] - (surf[a] - Ts) / (surf[a] - surf[a - 1]) * h
    yr = y[b] + (surf[b] - Ts) / (surf[b] - surf[b + 1]) * h
    return float(yr - yl) * 1e6, float(depth) * 1e6


def run(case, A, cell_um=10.0, dt=2e-7, dsdT=-1.1e-4):
    c = CASES[case]
    h = cell_um * 1e-6
    nx, ny, nz, ng = int(round(LX / h)), int(round(LY / h)), int(round(LZ / h)), int(round(GAS / h))
    metal = np.zeros((nx + 2, ny + 2, nz + ng + 2), bool)
    metal[1:-1, 1:-1, 1:nz + 1] = True
    mat = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
    marangoni = SahooSurfaceTension(a_s=float(dsdT[1:]), A=3.8e-4) if isinstance(dsdT, str) else dsdT
    md = FreeSurfaceMeltPool(mat, h, metal, dt=dt, viscosity=5e-3, sigma=1.8, mushy_C=1e9, T0=25.0, T_max=8000.0,
                             dsigma_dT=marangoni, evaporation=Evaporation(), emissivity=0.4, energy_transport="lbm")
    y = h + 0.5 * LY
    res = md.run(Laser(c["P"], A, W_BEAM, depth=DELTA), [ScanVector(h + X0, y, h + X1, y, c["v"])], t_extra=60e-6)
    w, d = pool_size(res.T_peak, nz, h, md.Ts, X0 + 0.75 * (X1 - X0))
    out = dict(case=case, A=A, dsigma_dT=dsdT, cell_um=cell_um, width_um=w, depth_um=d, measured_width=c["width"],
               measured_depth=c["depth"], T_peak_max=float(res.T_peak.max()), u_max_m_s=res.u_max,
               recoil_clipped=md.recoil_clipped, limited=md.lbm.limited, mass_error=res.mass_error,
               energy_balance=res.energy_balance, energy_residual=res.energy_residual, transport="lbm",
               steps=res.steps, seconds=res.seconds)
    print(json.dumps(out), flush=True)
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    tag = dsdT if isinstance(dsdT, str) else f"{dsdT:+.1e}"
    store[f"{case}|A={A:.3f}|dsdT={tag}|{cell_um:g}um|lbm"] = out
    json.dump(store, open(OUT, "w"), indent=1)
    return out


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 10.0,
        dsdT=(sys.argv[4] if sys.argv[4].startswith("S") else float(sys.argv[4])) if len(sys.argv) > 4 else -1.1e-4)
