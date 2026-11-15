"""Grain structure of the CoCrFeMnNi gyroid lattice over many layers: 3-D grain CA.

Section 7 of docs/process_models.md grew grains for one layer only, and noted that the strong
<001> build texture reported for LPBF CoCrFeMnNi needs many layers of competitive growth.
Here the 3-D grain CA (tpmsfem.process.grain_ca) builds 20 layers of the graded gyroid sheet
of that study:
* 4 mm cell, wall thickness graded from 0.2 mm (x = 0) to 0.6 mm (x = 4 mm);
* 200 W, 0.75 m/s, 30 um layers, 85 um hatch, one contour 35 um inside the surface, no
  rotation;
* two 1 x 1 mm windows, thin walls (x = 0.2-1.2 mm) and thick walls (x = 2.6-3.6 mm), from
  a build height of 0.7 to 1.3 mm.

Thermal input. The steady single track of the validation run (powder layer on a plate at
80 C; melt pool 97 um wide, 24 um into the plate), shifted along every clipped scan vector
of every layer (grain_ca.track_events). Not included:
* the shorter pools of short vectors;
* preheating between tracks;
* the different pool over powder at down-facing surfaces. Powder cells that the pool reaches
  join the part, but with the plate-pool shape.

Grains: the part below the first layer has random 20 um grains (as in section 7); KGT tip
kinetics of the CrFeCo-(MnNi) pseudo-binary (40 wt%, k = 0.74, freezing range 60 K); no
nucleation. CA cells 2.5 um.

    python benchmarks/literature/cantor_gyroid_ca.py section          # Cantor track (~5 min, cached)
    python benchmarks/literature/cantor_gyroid_ca.py run thin|thick|thin_a1     # _a1: heat-flow bias 1
    python benchmarks/literature/cantor_gyroid_ca.py figure
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ca_multilayer as ML  # noqa: E402
import cantor_gyroid_solidification as CG  # noqa: E402
import nist_ambench2018_tracks_ca as CB  # noqa: E402

from tpmsfem.process import grain_ca as CA  # noqa: E402

SECTION = "benchmarks/literature/results/cantor_track_section.npz"
OUT = "benchmarks/literature/results/cantor_gyroid_ca.json"
FIELDS = "benchmarks/literature/results/cantor_gyroid_ca_{name}.npz"
FIG = "docs/img/cantor_gyroid_ca.png"
DX = 2.5e-6
N_LAYERS, Z_START = 20, 0.7                      # mm: top of the part when the first layer is spread
WINDOWS = {"thin": (0.2, 1.2, 1.0, 2.0), "thick": (2.6, 3.6, 0.55, 1.55)}   # x0, x1, y0, y1 [mm]
SUB = 60e-6                                      # part below the first layer inside the CA grid


def section():
    if os.path.exists(SECTION):
        d = np.load(SECTION)
        return CA.TrackSection(d["l"], d["z"], d["T_peak"], d["tau_m"], d["tau_l"], d["cooling_rate"],
                               float(d["T_melt"]), float(d["speed"]))
    from tpmsfem.process import MeltPoolModel, ScanVector

    md = MeltPoolModel(CG.MAT, CG.H, length=1.2, width=0.2, depth=0.15, powder=CG.LAYER, T0=CG.T_PLATE,
                       emissivity=0.4, evaporation=CG.EVAP)
    res = md.run(CG.laser(), [ScanVector(0.15, 0.0, 1.05, 0.0, CG.V_MM_S)], t_extra=2e-4)
    sec = CA.TrackSection.from_meltpool(res, 0.7, CG.V_MM_S, 0.15)
    sec.z = sec.z - CG.LAYER * 1e-3              # heights relative to the top of the powder layer
    melted = sec.T_peak >= sec.T_melt
    zz = np.broadcast_to(sec.z[None, :], melted.shape)
    print(f"Cantor track: melted down to {zz[melted].min() * 1e6:.0f} um below the powder top, "
          f"tau_l up to {sec.tau_l[melted].max() * 1e6:.0f} us", flush=True)
    np.savez_compressed(SECTION, l=sec.l, z=sec.z, T_peak=sec.T_peak, tau_m=sec.tau_m, tau_l=sec.tau_l,
                        cooling_rate=sec.cooling_rate, T_melt=sec.T_melt, speed=sec.speed)
    return sec


def kinetics(dT0=60.0, c0=40.0, k=0.74):
    m = -dT0 * k / (c0 * (1.0 - k))
    return CA.TipKinetics.kgt(c0, k, m, 3e-9, 3.65e-7)


def run(name):
    """``name``: thin or thick, optionally with a heat-flow bias suffix, e.g. thin_a1."""
    sec = section()
    kin = kinetics()
    gy = CG.gyroid()
    window, alpha = (name.split("_a")[0], float(name.split("_a")[1])) if "_a" in name else (name, 0.0)
    x0, x1, y0, y1 = WINDOWS[window]
    n = int(round((x1 - x0) * 1e-3 / DX))
    nz = int(round((SUB + N_LAYERS * CG.LAYER * 1e-3 + 10e-6) / DX))
    zb = Z_START * 1e-3 - SUB                    # bottom of the grid [m]
    grid = CA.CAGrid((x0 * 1e-3 + 0.5 * DX, y0 * 1e-3 + 0.5 * DX, zb + 0.5 * DX), (n, n, nz), DX)
    X, Y, Z = np.meshgrid(*(grid.centres(a) * 1e3 for a in range(3)), indexing="ij")
    part = np.asarray(gy(X, Y, Z)) < 0.0
    rng = np.random.default_rng(0)
    grain, rot = CA.voronoi_substrate(grid, 20e-6, rng, mask=part & (Z <= Z_START))
    del X, Y
    ca = CA.GrainCA(grid, grain, rot, kin, seed=0, heat_flow_bias=alpha)
    t = 0.0
    t_wall = time.perf_counter()
    out = dict(window=window, heat_flow_bias=alpha, bounds_mm=WINDOWS[window], layers=[])
    for layer in range(N_LAYERS):
        z_top = Z_START + (layer + 1) * CG.LAYER                       # mm
        path, st = CG.slice_scan_path(gy, z_top - 0.5 * CG.LAYER, (x0 - 0.05, x1 + 0.05, y0 - 0.05, y1 + 0.05),
                                      hatch=CG.HATCH, speed=CG.V_MM_S, contour_offset=CG.CONTOUR_OFFSET)
        tracks = []
        for v in path:
            L = np.hypot(v.x1 - v.x0, v.y1 - v.y0) * 1e-3
            if v.power is None or v.power > 0:
                tracks.append(CA.Track((v.x0 * 1e-3, v.y0 * 1e-3), (v.x1 * 1e-3, v.y1 * 1e-3), z_top * 1e-3, t,
                                       v.speed * 1e-3))
            t += L / (v.speed * 1e-3)
        ev = CA.track_events(grid, sec, tracks)
        ca.run(ev)
        t += 5e-3
        print(f"  {name}: layer {layer + 1}/{N_LAYERS}, {len(tracks)} vectors, {len(ev['cell'])} events, {ca.steps} steps "
              f"[{time.perf_counter() - t_wall:.0f} s]", flush=True)
    res = ca.result()
    for layer in range(N_LAYERS):
        k_lo = int(round((SUB + layer * CG.LAYER * 1e-3) / DX))
        st = ML.layer_stats(res, k_lo, k_lo + int(round(CG.LAYER * 1e-3 / DX)))
        st.update(layer=layer + 1, z_mm=Z_START + (layer + 0.5) * CG.LAYER)
        out["layers"].append(st)
        print(f"    layer {layer + 1}: <100>||BD {100 * st['frac_100_bd_15']:.0f} %, width {st['width_area_weighted_um']:.1f} um, "
              f"{st['grains']} grains", flush=True)
    out.update(steps=res.steps, seconds=time.perf_counter() - t_wall, ca_stats=res.stats,
               substrate=ML.layer_stats(res, 2, int(round(SUB / DX)) - 2))
    kz = int(round((SUB + (N_LAYERS - 0.5) * CG.LAYER * 1e-3) / DX))
    jm = n // 2
    np.savez_compressed(FIELDS.format(name=name), plan_bd=CB._ipf3(res, res.grain[:, :, kz], (0.0, 0.0, 1.0)),
                        xz_bd=CB._ipf3(res, res.grain[:, jm, :], (0.0, 0.0, 1.0)),
                        part_plan=part[:, :, kz], part_xz=part[:, jm, :],
                        top_rot=res.orientations[ML._ids(res.grain[:, :, int(round((SUB + 10 * CG.LAYER * 1e-3) / DX)):kz]
                                                          [::3, ::3, ::3])].astype(np.float32), dx=np.array(DX))
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    store[name] = out
    json.dump(store, open(OUT, "w"), indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "section":
        section()
    elif cmd == "run":
        run(sys.argv[2])
    elif cmd == "figure":
        import cantor_gyroid_ca_figure as F
        F.figure()
