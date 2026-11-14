"""Porosity process maps for laser powder-bed fusion of IN625 and CoCrFeMnNi.

For every (laser power, scan speed) the calibrated conduction melt-pool model (MeltPoolModel,
with a powder layer, evaporation and radiation) gives the melt-pool width W, depth D (from
the top of the powder) and length. From those:

* lack of fusion: Tang, Pistorius & Beuth (2017) criterion (h/W)^2 + (L/D)^2 <= 1 and the
  geometric unfused fraction (tpmsfem.process.defects.lof_porosity, 10 % track scatter);
* keyhole: the keyhole number Ke of Gan et al. (2021): conduction < 1.4 <= transition <= 6 <
  keyhole; keyhole pores expected at high Ke (small fluctuations and almost no porosity below
  ~16, chaotic above ~30 in their data).

Validation (IN625): Locatelli et al., Key Eng. Mater. 1047 (2026) 31: Print Genius 150,
30 um layers, 50 um hatch, plate at 80 C, P = 100-300 W, v = 0.5-2.0 m/s. Reported: lack of
fusion at 100 W, worsening with speed; keyhole porosity at 250-300 W and 0.5 m/s; ~99.9 %
dense at 1.0-1.5 m/s and P > 100 W. The spot size is not reported: 1/e^2 radii of 30 and
40 um are run.

Assumptions: absorptivity and absorption depth of the AM-Bench case-B calibration for IN625
(0.3234, 15.3 um; obtained with an 85 um spot) and the uncalibrated values of the Cantor
study (0.35, 15 um); a 2-D lack-of-fusion estimate without scan rotation.

    python benchmarks/literature/porosity_maps.py run IN625 30      # spot radius in um
    python benchmarks/literature/porosity_maps.py figure
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STORE = "benchmarks/literature/results/porosity_maps.json"
FIG = "docs/img/porosity_maps.png"
POWERS = [100, 125, 150, 175, 200, 225, 250, 275, 300]
SPEEDS = [400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500]          # mm/s
LOCATELLI = dict(layer=0.030, hatch=0.050, T0=80.0,
                 lof=[(100, 1000), (100, 1500), (100, 2000)],          # stated in the text
                 keyhole=[(250, 500), (300, 500)],
                 dense=[(p, v) for p in (150, 200, 250, 300) for v in (1000, 1500)])


def material(name):
    from tpmsfem.fem.materials import Material
    from tpmsfem.process import Evaporation
    if name == "IN625":
        import in625_material as IM
        mat = Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
        return mat, 0.3234, 0.0153, Evaporation()
    import cantor_material as CM
    mat = CM.material()
    return mat, 0.35, 0.015, Evaporation(T_boil=2061.0, L_v=4.02e6, M=0.05494, p0=0.2 * 101325.0)


def track(args):
    """One single track over a powder layer on a plate: W, D (from the powder top), L [um]."""
    name, P, v, radius, layer, T0, h = args
    from tpmsfem.process import Laser, MeltPoolModel, ScanVector
    from tpmsfem.process.meltpool import fusion_zone, pool_length

    mat, A, delta, evap = material(name)
    x0, x1 = 0.10, 0.90
    md = MeltPoolModel(mat, h, length=1.05, width=0.16, depth=0.20, powder=layer, T0=T0, emissivity=0.4,
                       evaporation=evap)
    t_end = (x1 - x0) / v
    t0 = time.perf_counter()
    res = md.run(Laser(P, A, radius, depth=delta), [ScanVector(x0, 0.0, x1, 0.0, v)], t_extra=2e-4,
                 snapshot_times=[t_end - 1e-9])
    w, d = fusion_zone(res, x0 + 0.7 * (x1 - x0))
    L = pool_length(res.snapshots[0][1], res) if res.snapshots else float("nan")
    return dict(material=name, P=P, v=v, radius_um=radius * 1e3, layer_um=layer * 1e3, T0=T0,
                # fusion_zone measures depth below the plate top; the lack-of-fusion criterion
                # wants it from the top of the powder layer (Tang et al.: D vs layer L)
                width_um=1e3 * (w or 0.0), depth_um=1e3 * ((d or -layer) + layer), depth_into_plate_um=1e3 * (d or -layer),
                length_um=1e3 * L,
                T_peak=float(res.T_peak.max()), seconds=time.perf_counter() - t0)


def key(r):
    return f"{r['material']}|r={r['radius_um']:.0f}|L={r['layer_um']:.0f}|T0={r['T0']:.0f}|P={r['P']}|v={r['v']:.0f}"


def run(name, radius_um, layer=0.030, T0=80.0, h=0.005, workers=4):
    store = json.load(open(STORE)) if os.path.exists(STORE) else {}
    jobs = []
    for P in POWERS:
        for v in SPEEDS:
            k = f"{name}|r={radius_um:.0f}|L={layer * 1e3:.0f}|T0={T0:.0f}|P={float(P)}|v={v:.0f}"
            if k not in store:
                jobs.append((name, float(P), float(v), radius_um * 1e-3, layer, T0, h))
    print(f"{len(jobs)} tracks to run", flush=True)
    with Pool(workers) as pool:
        for r in pool.imap_unordered(track, jobs):
            store = json.load(open(STORE)) if os.path.exists(STORE) else {}
            store[key(r)] = r
            json.dump(store, open(STORE, "w"), indent=1)
            print(f"  {r['P']:.0f} W {r['v']:.0f} mm/s: W {r['width_um']:.0f} D {r['depth_um']:.0f} "
                  f"L {r['length_um']:.0f} um, T_peak {r['T_peak']:.0f} C ({r['seconds']:.0f} s)", flush=True)


def props(name, T=700.0):
    """rho [kg/m^3], c_p [J/(kg K)], alpha [m^2/s], liquidus [C] at temperature T (for Ke)."""
    mat = material(name)[0]
    th = mat.thermal
    rho = th.rho * 1e12
    cp = float(np.interp(T, th.T, th.cp)) * 1e-6
    k = float(th.conductivity(T))
    return rho, cp, k / (rho * cp), float(th.T_liquidus)


def maps(name, radius_um, layer, hatch, T0=80.0):
    """Arrays over (P, v): W, D, the LoF criterion, the LoF porosity [%] and Ke."""
    from tpmsfem.process.defects import keyhole_number, lof_criterion, lof_porosity

    store = json.load(open(STORE))
    A = material(name)[1]
    rho, cp, alpha, Tl = props(name)
    shape = (len(POWERS), len(SPEEDS))
    W, D, crit, por, Ke = (np.full(shape, np.nan) for _ in range(5))
    for i, P in enumerate(POWERS):
        for j, v in enumerate(SPEEDS):
            r = store.get(f"{name}|r={radius_um:.0f}|L={layer * 1e3:.0f}|T0={T0:.0f}|P={float(P)}|v={v:.0f}")
            if not r:
                continue
            W[i, j], D[i, j] = r["width_um"], r["depth_um"]
            crit[i, j] = lof_criterion(W[i, j], D[i, j], hatch * 1e3, layer * 1e3)
            por[i, j] = 100.0 * lof_porosity(W[i, j], D[i, j], hatch * 1e3, layer * 1e3, cv=0.10, n=160, samples=8)
            Ke[i, j] = keyhole_number(P, v * 1e-3, A, radius_um * 1e-6, rho, cp, Tl, T0, alpha)
    return dict(W=W, D=D, crit=crit, por=por, Ke=Ke)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], float(sys.argv[3]), layer=float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.030)
    elif sys.argv[1] == "figure":
        import porosity_maps_figure as F
        F.figure()
