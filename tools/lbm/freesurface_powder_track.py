"""Single LPBF track on a powder layer vs on the bare plate, with the free-surface lattice
Boltzmann melt pool (IN625 properties, Marangoni, recoil, evaporation, radiation, gravity).

Powder: log-normal sizes (d50 30 um, 15-45 um), rained and rolled onto the plate, recoater at
40 um (tpmsfem.process.powder). Laser 200 W, 0.8 m/s, 1/e^2 radius 40 um, absorptivity 0.40
(Beer-Lambert along the metal), absorption depth 5 um. 5 um cells, 0.05 us steps (recoil-driven
flow reaches ~10 m/s).

    python tools/lbm/freesurface_powder_track.py powder|plate
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
from tpmsfem.process.powder import rain_powder, voxelise  # noqa: E402

OUT = "tools/lbm/powder_track.json"
H = 5e-6
LX, LY, PLATE, LAYER, GAS = 0.6e-3, 0.25e-3, 80e-6, 40e-6, 40e-6
P, V, RADIUS, A, DELTA = 200.0, 0.8, 40e-6, 0.40, 5e-6
X0, X1 = 0.1e-3, 0.5e-3


def geometry(case, seed=3):
    nx, ny = int(round(LX / H)), int(round(LY / H))
    npl, nla, nga = int(round(PLATE / H)), int(round(LAYER / H)), int(round(GAS / H))
    shape = (nx + 2, ny + 2, npl + nla + nga + 2)
    fill = np.zeros(shape)
    fill[1:-1, 1:-1, 1:npl + 1] = 1.0
    bed = None
    if case == "powder":
        floor = (npl + 1) * H
        bed = rain_powder((H, H + LX, H, H + LY), floor, LAYER, seed=seed)
        fill = np.maximum(fill, voxelise(bed, H, shape))
    return fill, bed, npl


def run(case):
    fill, bed, npl = geometry(case)
    mat = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
    md = FreeSurfaceMeltPool(mat, H, fill, dt=5e-8, viscosity=5e-3, sigma=1.8, mushy_C=1e9, T0=25.0, T_max=6000.0,
                             dsigma_dT=-1.1e-4, evaporation=Evaporation(), emissivity=0.4)
    s0 = np.asarray(md.surface_height()).copy()
    y = H + 0.5 * LY
    res = md.run(Laser(P, A, RADIUS, depth=DELTA), [ScanVector(H + X0, y, H + X1, y, V)], t_extra=60e-6,
                 verbose=True)
    plate_top = (npl + 1) * H
    surf = res.surface
    # melted region of the plate (peak T >= solidus below the plate top)
    Tp = res.T_peak
    sec = slice(int(round((X0 + 0.25 * (X1 - X0)) / H)) + 1, int(round((X0 + 0.75 * (X1 - X0)) / H)) + 1)
    melted = Tp[sec, 1:-1, 1:npl + 1] >= md.Ts
    width = float(np.mean([(np.nonzero(m.any(axis=1))[0].ptp() + 1) * H * 1e6 if m.any() else 0.0
                           for m in melted]))
    depth = float(np.mean([(npl - np.nonzero(m.any(axis=0))[0].min()) * H * 1e6 if m.any() else 0.0 for m in melted]))
    cl = (surf[sec, 1 + int(round(0.5 * LY / H))] - plate_top) * 1e6
    out = dict(case=case, cells=int(np.prod(fill.shape)), particles=0 if bed is None else len(bed),
               packing=None if bed is None else bed.packing_density, width_in_plate_um=width, depth_in_plate_um=depth,
               track_height_um=dict(mean=float(cl.mean()), sd=float(cl.std()), min=float(cl.min()), max=float(cl.max())),
               T_peak_max=float(Tp.max()), u_max_m_s=res.u_max, mass_error=res.mass_error,
               energy_balance=res.energy_balance, steps=res.steps, seconds=res.seconds, mlups=res.mlups)
    print(json.dumps(out, indent=1), flush=True)
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    store[case] = out
    json.dump(store, open(OUT, "w"), indent=1)
    np.savez_compressed(OUT.replace(".json", f"_{case}.npz"), surface0=s0.astype(np.float32),
                        surface=surf.astype(np.float32), fill=res.fill.astype(np.float16),
                        T_peak=Tp.astype(np.float32), plate_top=plate_top, h=H,
                        centres=np.zeros((0, 3)) if bed is None else bed.centres,
                        radii=np.zeros(0) if bed is None else bed.radii)
    return out


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "powder")
