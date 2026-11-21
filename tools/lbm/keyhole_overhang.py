"""Closed keyhole 3 (flow): a keyhole in a thin CoCrFeMnNi slab over loose powder, the
down-facing side of a gyroid wall. Does the keyhole stay closed (its tip in the slab) or break
through into the powder below (open, full penetration)?

Geometry (bottom up): a 30 um base plate, a powder bed (``--powder``, rained and rolled,
d50 30 um), the solid slab (``--slab`` thick, the wall below the new layer) and gas. The laser
scans the bare slab top at the gyroid study's settings: 200 W, 0.75 m/s, r0 35 um.

Physics as in keyhole_ti64.py (ray-traced Fresnel laser, flat-melt absorptivity 0.33, bubble
pressures, recoil, evaporation, Marangoni, sub-grid viscosity, dt = dx / 250 m/s), with the
CoCrFeMnNi thermal data of the gyroid study and Fe-Ni-Cr-type melt properties (assumed: sigma
1.8 N/m, d sigma/dT -1.1e-4 N/(m K), viscosity 5 mPa s); evaporation and recoil of Mn at its
20 at% (as in the gyroid study).

The gas in the powder under the slab connects to the chamber through the rest of the bed, so
gas reaching the side faces is ambient (BubbleModel(open_sides=True)); gas pockets enclosed by
the melt are bubbles. The break-through is when the keyhole's gas (connected to the top)
reaches below the slab (``open_below``). Diagnostics every 1 us: keyhole depth, break-through,
bubbles; at the end, the powder melted below the slab (dross) and the pores.

Gan et al.'s law puts the keyhole at ~190 um (fluctuating 125-250 um) at these settings, so
slabs of 90, 150 and 250 um (plus the 30 um layer of the real build: here the slab top is the
surface) bracket open, intermittent and closed.

    python tools/lbm/keyhole_overhang.py --slab 150e-6              # GPU: TPMSFEM_BACKEND=cupy
    python tools/lbm/keyhole_overhang.py --slab 90e-6 --dx 5e-6 --track 100e-6   # CPU check
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks", "literature"))
import cantor_material as CM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.backend import asnumpy  # noqa: E402
from tpmsfem.process import Evaporation, Laser, ScanVector  # noqa: E402
from tpmsfem.process.freesurface import FreeSurfaceMeltPool  # noqa: E402
from tpmsfem.process.keyhole import BubbleModel, RayTracer, keyhole_metrics, melt_diagnostics, open_gas_reaches  # noqa: E402
from tpmsfem.process.powder import rain_powder, voxelise  # noqa: E402

P, V, R0, ETA_M = 200.0, 0.75, 35e-6, 0.33
EVAP = Evaporation(T_boil=2061.0, L_v=4.02e6, M=0.05494, p0=0.2 * 101325.0)     # Mn at 20 at% (gyroid study)
MELT = dict(sigma=1.8, dsigma_dT=-1.1e-4, viscosity=5e-3)                          # assumed (Fe-Ni-Cr-type melt)


def build(a):
    if a.backend:
        tf.set_backend(a.backend)
    dx = a.dx
    lx, ly = a.lead + a.track + a.tail, 2 * a.half_width
    nb, npw, nsl, ng = (int(round(v / dx)) for v in (a.base, a.powder, a.slab, a.gas))
    shape = (int(round(lx / dx)) + 2, int(round(ly / dx)) + 2, nb + npw + nsl + ng + 2)
    fill = np.zeros(shape)
    fill[1:-1, 1:-1, 1:nb + 1] = 1.0                                          # base plate
    z_pow = (nb + 1) * dx                                                     # powder floor
    z_slab = (nb + npw + 1) * dx                                              # slab bottom = powder top
    bed = rain_powder((dx, dx + lx, dx, dx + ly), z_pow, a.powder, seed=a.seed)
    fill = np.maximum(fill, voxelise(bed, dx, shape))
    fill[1:-1, 1:-1, nb + npw + 1:nb + npw + nsl + 1] = 1.0                   # the slab
    md = FreeSurfaceMeltPool(CM.material(), dx, fill, dt=a.dt or dx / 250.0, viscosity=MELT["viscosity"],
                             sigma=MELT["sigma"], mushy_C=1e9, T0=a.T0, T_max=9000.0, dsigma_dT=MELT["dsigma_dT"],
                             evaporation=EVAP, recoil_factor=0.54, emissivity=0.4, energy_transport="lbm")
    md.rays = RayTracer(A0=ETA_M)
    md.ray_every = a.ray_every
    md.bubbles = BubbleModel(min_cells=2, open_sides=True)      # the powder's gas is open through the bed
    md.bubble_every = a.bubble_every
    md.lbm.smagorinsky = a.smagorinsky
    y = dx + 0.5 * ly
    path = [ScanVector(dx + a.lead, y, dx + a.lead + a.track, y, V)]
    info = dict(P=P, v=V, r0=R0, eta_m=ETA_M, dx=dx, dt=md.dt, shape=list(shape), slab=a.slab, powder=a.powder,
                base=a.base, T0=a.T0, z_slab_bottom=z_slab, z_surface=z_slab + nsl * dx, y=y,
                powder_packing=float(bed.packing_density), particles=int(len(bed.radii)), melt=MELT)
    return md, path, Laser(P, 0.0, R0), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", type=float, default=150e-6)
    ap.add_argument("--powder", type=float, default=120e-6)
    ap.add_argument("--base", type=float, default=30e-6)
    ap.add_argument("--gas", type=float, default=50e-6)
    ap.add_argument("--dx", type=float, default=2.5e-6)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--track", type=float, default=300e-6)
    ap.add_argument("--lead", type=float, default=80e-6)
    ap.add_argument("--tail", type=float, default=100e-6)
    ap.add_argument("--half-width", type=float, default=125e-6)
    ap.add_argument("--cool", type=float, default=100e-6)
    ap.add_argument("--T0", type=float, default=80.0, help="part temperature [C]")
    ap.add_argument("--ray-every", type=int, default=10)
    ap.add_argument("--bubble-every", type=int, default=10)
    ap.add_argument("--smagorinsky", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", type=float, default=20.0)
    ap.add_argument("--redo", action="store_true", help="rerun a case that has already finished")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "keyhole_runs"))
    a = ap.parse_args()
    md, path, laser, info = build(a)
    os.makedirs(a.out, exist_ok=True)
    tag = f"overhang_slab{a.slab * 1e6:.0f}um_dx{a.dx * 1e6:g}um"
    js, npz, ck = (os.path.join(a.out, f"{tag}{s}") for s in (".json", "_sections.npz", "_checkpoint.npz"))
    if not a.redo and not os.path.exists(ck) and os.path.exists(js) and json.load(open(js)).get("finished"):
        print(f"{tag}: already finished ({js}); --redo to run it again", flush=True)
        return
    log, sections = [], []
    if os.path.exists(ck):
        md.load_state(ck)
        log = json.load(open(js)).get("log", []) if os.path.exists(js) else []
        print(f"resumed at t = {md.t * 1e6:.1f} us", flush=True)
    dx = a.dx
    k_slab = int(round(info["z_slab_bottom"] / dx))          # first slab cell index
    jc = int(round(info["y"] / dx))
    every = max(1, int(round(1e-6 / md.dt)))
    last, t_wall = {}, [time.time(), time.time()]

    def save(finished=False):
        info.update(log=log, finished=finished)
        json.dump(info, open(js, "w"), indent=1)
        if sections:
            np.savez_compressed(npz, t=np.array([s[0] for s in sections]), fill=np.stack([s[1] for s in sections]),
                                T=np.stack([s[2] for s in sections]))

    def cb(m, t, x_l, y_l):
        row = dict(t=t, x_l=x_l, laser_on=x_l is not None)
        if x_l is not None:
            last.update(x=x_l, y=y_l)
        if last:
            row.update(keyhole_metrics(m, last["x"], last["y"], R0, info["z_surface"]))
        if x_l is not None:
            row.update({f"ray_{k}": v for k, v in m.ray_stats.items()})
        row["open_below"], row["open_below_cells"] = open_gas_reaches(m.lbm.flag, k_slab)
        b = m.bubbles
        row["bubbles"] = [dict(volume_um3=x["volume"] * (dx * 1e6) ** 3, centroid_um=[c * dx * 1e6 for c in x["centroid"]],
                               frozen=x["frozen"], pore=x["pore"], pressure=x["pressure"]) for x in b.bubbles]
        row["formed"], row["released"] = b.formed, b.released
        row.update(melt_diagnostics(m))          # T_max over metal; u_max of the pool, droplets apart
        row.update(limited=int(m.lbm.limited), recoil_clipped=int(m.recoil_clipped),
                   mass=float(m.lbm.total_mass()), wall_s=time.time() - t_wall[0])
        log.append(row)
        T_sec = np.where(asnumpy(m.metal()[:, jc, :]), asnumpy(m.T[:, jc, :]), np.nan)
        sections.append((t, asnumpy(m.lbm.fill[:, jc, :]).astype(np.float16), T_sec.astype(np.float16)))
        if len(log) % 10 == 0:
            print(f"t {t * 1e6:7.1f} us  depth {row.get('depth', float('nan')) * 1e6:6.1f} um  "
                  f"open below {row.get('open_below')} ({row.get('open_below_cells', 0)} cells)  "
                  f"bubbles {len(row['bubbles'])} (pores {sum(x['pore'] for x in row['bubbles'])})  "
                  f"T_max {row['T_max']:6.0f} C  u_max {row['u_max']:5.1f} m/s  drops {row['drops']}  "
                  f"lim {row['limited']}  ({row['wall_s']:.0f} s)", flush=True)
        if time.time() - t_wall[1] > 60.0 * a.checkpoint:
            md.save_state(ck)
            save()
            t_wall[1] = time.time()

    print(f"{tag}: cells {info['shape']} ({np.prod(info['shape']) / 1e6:.2f} M), {info['particles']} particles "
          f"(packing {info['powder_packing']:.2f}), dt {md.dt * 1e9:.0f} ns", flush=True)
    res = md.run(laser, path, t_extra=a.cool, callback=cb, callback_every=every)
    # powder melted below the slab (dross): cells under the slab that reached the liquidus
    Tl = md.Tl
    melted_below = (res.T_peak[:, :, 1:k_slab] > Tl) & (res.fill[:, :, 1:k_slab] > 0.5)
    info.update(mass_error=res.mass_error, energy_balance=res.energy_balance, energy_residual=res.energy_residual,
                steps=res.steps, seconds=res.seconds, mlups=res.mlups,
                dross_um3=float(melted_below.sum() * (dx * 1e6) ** 3),
                dross_depth_um=float((k_slab - 1 - np.nonzero(melted_below.any(axis=(0, 1)))[0].min()) * dx * 1e6)
                if melted_below.any() else 0.0)
    np.savez_compressed(os.path.join(a.out, f"{tag}_final.npz"), flag=res.flag.astype(np.int8),
                        fill=res.fill.astype(np.float16), T_peak=res.T_peak.astype(np.float16))
    save(finished=True)
    if os.path.exists(ck):
        os.remove(ck)
    print("wrote", js, {k: info[k] for k in ("dross_um3", "dross_depth_um", "mass_error", "energy_residual")}, flush=True)


if __name__ == "__main__":
    main()
