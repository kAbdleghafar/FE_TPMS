"""Heat accumulation over many layers in the LPBF CoCrFeMnNi gyroid sheet, and its effect on
the solidification conditions (follow-up of cantor_gyroid_solidification.py, which simulated
one layer on an 80 C substrate).

Part scale (tpmsfem.thermal.HeatModel):
- one 4 x 4 mm gyroid cell column, 10 mm tall, walls graded 0.2 -> 0.6 mm, on a 2 mm plate
  whose bottom is held at 80 C; the surrounding powder is part of the model (2 % of the
  solid conductivity, 55 % of its density); the side faces are adiabatic (neighbouring
  cells); the top loses heat by convection and radiation;
- voxels of 1/15 mm; the 30 um layers are lumped into 0.133 mm lumps (4.4 layers);
- each lump receives the energy of its layers where the laser puts it: the contour + hatch
  path of the lump's slice (process.scanpath) deposits ``f_net A P / v`` per unit length
  into the solid voxels below it. The lump's energy arrives in 4 pulses, 1.1 layer times
  apart (each spread over one heat step, a third of that interval), so the temperature just
  before a pulse is what a new layer meets. Depositing a lump's energy at once and waiting
  4.4 layer times would understate it. ``f_net`` is the share
  of the absorbed energy that stays in the part (the rest evaporates or radiates from the
  melt pool); it is taken from a single-track melt-pool run;
The temperature of the top of the part at the end of each lump (just before the next pulse) is the preheat the next layer
sees, recorded separately for the thin-wall and thick-wall windows of the single-layer study.

Melt-pool scale: the two windows of the single-layer study are re-run at the layer 9.3 mm above
the plate (the same slice as 1.3 mm, two cells higher) with that preheat, followed by the
phase field at each class's fusion-boundary conditions.

    python benchmarks/literature/cantor_gyroid_multilayer.py --part build [--layer-time 10]
    python benchmarks/literature/cantor_gyroid_multilayer.py --part windows
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
import cantor_material as CM  # noqa: E402
from common import ResultStore  # noqa: E402

from tpmsfem.fem.materials import Material, ThermalProperties  # noqa: E402
from tpmsfem.mesh.voxel import mask_to_mesh  # noqa: E402
from tpmsfem.process import MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process.scanpath import slice_scan_path  # noqa: E402
from tpmsfem.thermal.heat import HeatModel  # noqa: E402

STORE = "benchmarks/literature/results/cantor_gyroid_multilayer.json"
N_XY, VOX = 60, 4.0 / 60
PLATE, HEIGHT, LUMP_VOX = 2.0, 10.0, 2
N_PULSE, N_SUB = 4, 3                            # energy pulses per lump; heat steps per pulse
POWDER_K, POWDER_RHO = 0.02, 0.55
Z_WINDOW = 9.3                                   # layer re-run at the melt-pool scale [mm]


def powder_material():
    d = CM.LI2021
    T = np.r_[d["T_C"], 3500.0]
    k = np.r_[d["k_W_mK"], d["k_W_mK"][-1]] * POWDER_K
    cp = np.r_[d["cp_J_kgK"], d["cp_J_kgK"][-1]]
    return Material("CoCrFeMnNi powder", 1.0, 0.3,
                    thermal=ThermalProperties.from_si(T, k, cp, d["density_kg_m3"] * POWDER_RHO))


def net_fraction(store):
    """Share of the absorbed laser energy left in the material after a single track."""
    if store.get("net_fraction") is None:
        md = MeltPoolModel(CG.MAT, CG.H, length=1.2, width=0.25, depth=0.2, powder=CG.LAYER, T0=CG.T_PLATE,
                           emissivity=0.4, evaporation=CG.EVAP)
        H0 = float(md.H.sum()) * CG.H ** 3 * 2.0
        res = md.run(CG.laser(), [ScanVector(0.15, 0.0, 1.05, 0.0, CG.V_MM_S)], t_extra=2e-4)
        gain = float(md.H.sum()) * CG.H ** 3 * 2.0 - H0
        store.put("net_fraction", dict(absorbed_mJ=float(res.absorbed_energy), retained_mJ=gain,
                                       f_net=gain / float(res.absorbed_energy)))
    return store.get("net_fraction")


def lump_energy(gy, z_mid, f_net):
    """Energy [mJ] of ONE layer at height z_mid per voxel column (N_XY, N_XY), from the scan path."""
    path, st = slice_scan_path(gy, z_mid, (0.0, CG.L_CELL, 0.0, CG.L_CELL), hatch=CG.HATCH, speed=CG.V_MM_S,
                               contour_offset=CG.CONTOUR_OFFSET)
    e = np.zeros((N_XY, N_XY))
    lin = f_net * CG.ABS * CG.P_W * 1e3 / CG.V_MM_S               # mJ per mm of lit path
    for v in path:
        if v.power == 0.0:
            continue
        L = float(np.hypot(v.x1 - v.x0, v.y1 - v.y0))
        n = max(int(np.ceil(L / (0.5 * VOX))), 1)
        s = (np.arange(n) + 0.5) / n
        ix = np.clip(((v.x0 + s * (v.x1 - v.x0)) / VOX).astype(int), 0, N_XY - 1)
        iy = np.clip(((v.y0 + s * (v.y1 - v.y0)) / VOX).astype(int), 0, N_XY - 1)
        np.add.at(e, (ix, iy), lin * L / n)
    return e, st


def part_build(store, layer_time):
    key = f"build|layer_time={layer_time:g}"
    if store.get(key) is not None:
        return store.get(key)
    f_net = net_fraction(store)["f_net"]
    gy = CG.gyroid()
    nzp, nzl = int(round(PLATE / VOX)), int(round(HEIGHT / VOX))
    shape = (N_XY, N_XY, nzp + nzl)
    c = (np.arange(N_XY) + 0.5) * VOX
    zc = -PLATE + (np.arange(nzp + nzl) + 0.5) * VOX
    X, Y, Z = np.meshgrid(c, c, zc, indexing="ij")
    solid = (Z < 0.0) | (np.asarray(gy(X, Y, Z)) < 0.0)
    mesh = mask_to_mesh(np.ones(shape, bool), (0.0, 0.0, -PLATE), VOX, materials=np.where(solid, 1, 2))
    ijk = mesh.cell_data["ijk"]
    hm = HeatModel(mesh, {1: CG.MAT, 2: powder_material()}, T_initial=CG.T_PLATE)
    hm.set_surface_losses(h=1e-5, emissivity=0.4, T_amb=CG.T_PLATE, face_filter=lambda ctr, n: n[:, 2] > 0.5)
    hm.fix(np.flatnonzero(mesh.points[:, 2] < -PLATE + 1e-6), CG.T_PLATE)
    lump = np.where(ijk[:, 2] < nzp, -1, (ijk[:, 2] - nzp) // LUMP_VOX)
    hm.deactivate(np.flatnonzero(lump >= 0))
    is_solid = np.asarray(mesh.cell_data["material"]) == 1
    n_lumps = nzl // LUMP_VOX
    lump_h = LUMP_VOX * VOX
    layers_per_lump = lump_h / CG.LAYER
    lump_time = layers_per_lump * layer_time
    dt = lump_time / (N_PULSE * N_SUB)                     # one step size: one AMG set-up per lump
    win = {n: (ijk[:, 0] * VOX >= w[0]) & (ijk[:, 0] * VOX < w[1]) & (ijk[:, 1] * VOX >= w[2]) & (ijk[:, 1] * VOX < w[3])
           for n, w in CG.WINDOWS.items()}
    recs = []
    t0 = time.perf_counter()
    for k in range(n_lumps):
        cells = np.flatnonzero(lump == k)
        hm.activate(cells, T_new=CG.T_PLATE)
        z_mid = (k + 0.5) * lump_h
        e, st = lump_energy(gy, z_mid, f_net)
        sc = cells[is_solid[cells]]
        n_s = np.zeros((N_XY, N_XY))
        np.add.at(n_s, (ijk[sc, 0], ijk[sc, 1]), 1.0)
        e_tot = e.sum()
        e_in = np.where(n_s > 0, e, 0.0)
        e_in *= e_tot / max(e_in.sum(), 1e-30)                 # energy that fell beside the voxel solid
        # the lump's layers arrive as N_PULSE pulses one (rescaled) layer time apart, so that the
        # temperature just before a pulse is the temperature a new layer meets
        q = (e_in / np.maximum(n_s, 1.0))[ijk[sc, 0], ijk[sc, 1]] * layers_per_lump / N_PULSE / (VOX ** 3 * dt)
        T_hot = 0.0
        for _ in range(N_PULSE):
            hm.q_vol[:] = 0.0
            hm.q_vol[sc] = q
            hm.step(dt)
            T_hot = max(T_hot, float(hm.T.max()))
            hm.q_vol[:] = 0.0
            for _ in range(N_SUB - 1):
                hm.step(dt)
        top = sc[ijk[sc, 2] == ijk[sc, 2].max()]
        Te = hm.T[mesh.cells[top]].mean(axis=1)
        rec = dict(lump=k, z_top=(k + 1) * lump_h, energy_per_layer_mJ=float(e_tot), T_after_source=T_hot,
                   T_top_mean=float(Te.mean()), T_top_max=float(Te.max()),
                   scan_length_mm=st["contour_length_mm"] + st["hatch_length_mm"])
        for n, w in win.items():
            sel = win[n][top]
            rec[f"T_top_{n}"] = float(Te[sel].mean()) if sel.any() else None
        recs.append(rec)
        if k % 5 == 0 or k == n_lumps - 1:
            print(f"lump {k + 1:3d}/{n_lumps}: z = {rec['z_top']:5.2f} mm, {rec['energy_per_layer_mJ']:.0f} mJ/layer, "
                  f"after source {T_hot:5.0f} C, before the next lump: mean {rec['T_top_mean']:5.1f} C, thin "
                  f"{rec['T_top_thin'] or float('nan'):5.1f}, thick {rec['T_top_thick'] or float('nan'):5.1f} "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    out = dict(layer_time=layer_time, lump_time=lump_time, f_net=f_net, voxel=VOX, lump_height=lump_h, records=recs)
    store.put(key, out)
    return out


def preheat(store, layer_time, name, z=Z_WINDOW):
    b = store.get(f"build|layer_time={layer_time:g}")
    zs = np.array([r["z_top"] for r in b["records"]])
    Ts = np.array([r[f"T_top_{name}"] if r[f"T_top_{name}"] is not None else np.nan for r in b["records"]])
    ok = np.isfinite(Ts)
    return float(np.interp(z - CG.LAYER, zs[ok], Ts[ok]))


def part_windows(store, layer_time, T_override=None, tag=None):
    """Melt-pool windows at z = 9.3 mm with the part-scale preheat, then the phase field."""
    res_store = ResultStore(CG.STORE.replace(".json", "_preheat.json"))
    for name in CG.WINDOWS:
        T0 = preheat(store, layer_time, name) if T_override is None else T_override
        label = tag or f"lt{layer_time:g}"
        key = f"window|{name}|{label}"
        if res_store.get(key) is None:
            print(f"window {name}, preheat {T0:.0f} C ({label})", flush=True)
            w = CG.run_window(name, T0=T0, verbose=False)
            w["T0"] = T0
            res_store.put(key, w)
        w = res_store.get(key)
        for c in ("supported", "down-facing"):
            d = w[c]
            print(f"  {name:5s} {c:11s} T0 = {w['T0']:.0f} C: G p50 {d['G']['p50']:.2e} K/m, R p50 {d['R']['p50']:.3f} m/s, "
                  f"cooling {d['cr']['p50']:.1e} K/s", flush=True)
            pk = f"pf|{name}|{c}|{label}"
            if res_store.get(pk) is None and d["n_columns"] >= 20:
                out, _ = CG.run_pf(d["G"]["p50"], d["R"]["p50"], 60.0)
                res_store.put(pk, out)
            o = res_store.get(pk)
            if o:
                print(f"     PF: spacing {o['spacing_um']:.2f} um, cores {o['cs_rel']:.3f}, max between cells "
                      f"{o['intercell_max_rel']:.2f} x nominal", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="build", choices=["build", "windows"])
    ap.add_argument("--layer-time", type=float, default=10.0)
    ap.add_argument("--preheat", type=float, default=None, help="override the preheat [C] (scenario)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    store = ResultStore(STORE)
    if a.part == "build":
        part_build(store, a.layer_time)
    else:
        part_windows(store, a.layer_time, a.preheat, a.tag)
    json.dump(store.data, open(STORE, "w"), indent=1)


if __name__ == "__main__":
    main()
