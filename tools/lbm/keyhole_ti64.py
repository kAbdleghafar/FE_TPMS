"""Keyhole stage 2: single tracks on bare Ti-6Al-4V with the free-surface melt pool, ray-traced
laser (Fresnel, multiple reflections) and bubble pressures, against the operando X-ray keyhole
depths of Cunningham et al., Science 363 (2019) 849 (as compiled by Gan et al. 2021,
benchmarks/literature/data/gan2021_keyhole_xray.json).

Physics: Gan et al.'s Ti-6Al-4V property set (benchmarks/literature/ti64_material.py); recoil
0.54 p_sat, evaporation cooling, Marangoni (-2.6e-4 N/(m K)), radiation (0.3), gravity, heat
carried with the lattice mass. Laser: 1/e^2 radius 47.5 um, rays every ``--ray-every`` steps,
flat-melt absorptivity 0.26 (k = n). Bubbles: isothermal ideal gas, ambient pressure at
pinch-off, updated every ``--bubble-every`` steps. Time step dt = dx / (250 m/s): lattice Mach
0.1 at 25 m/s, the speeds recoil reaches at the keyhole wall, and room for recoil pressures up to
~80 atm (wall temperatures to ~5,100 C) within the lattice's density-jump cap. With dx / (100
m/s) the cap and the speed limiter act in hundreds of cells and the keyhole sprays into
fragments (5 um CPU check).

Cases (r0 = 47.5 um; measured depth, Gan's Ke):
  T1 129.4 W 0.4 m/s (85 um, Ke 5.8)    T2 209.3 W 1.1 m/s (79 um, Ke 4.2)
  T3 260.2 W 0.6 m/s (169 um, Ke 10.9)  T4 312.5 W 0.9 m/s (172 um, Ke 9.8)
  T5 364.8 W 0.7 m/s (257 um, Ke 15.1)  T6 468.0 W 0.7 m/s (351 um, Ke 20.7)

Closed keyholes (``--scenario``):
  track     a straight track, the laser switched off abruptly at its end (skywriting) and the
            closing keyhole followed for ``--cool`` (default);
  ramp      the same with the power ramped linearly to zero over the last ``--ramp`` of track;
  turn      a galvanometer turn point (Martin et al., Nat. Commun. 10 (2019) 1987): deceleration
            at ``--accel``, a ``--hatch`` shift, acceleration back, at constant power;
  turn-ke   the same turn with the power lowered as the beam slows to keep Gan's keyhole number
            constant (floored at 20 % so the pool stays molten): a mitigation schedule.
Martin et al.'s turn cases: M100, M200, M300 (W) at 1.0 m/s with a 50 um beam (r0 25 um) on
Ti-6Al-4V; they saw pores at all powers with constant power, none with power modulation.

Every ``--every`` steps a line of diagnostics is appended to the JSON (keyhole depth, front-wall
tangent, opening, absorbed fraction, bounces, bubbles, peak temperature and speed, limiter and
recoil-cap counts), and a centreline section (fill, temperature) to the NPZ. A checkpoint is
written every ``--checkpoint`` minutes; rerunning the same command resumes from it.

    python tools/lbm/keyhole_ti64.py T3 --dx 2.5e-6          # GPU (CuPy): set TPMSFEM_BACKEND=cupy
    python tools/lbm/keyhole_ti64.py T3 --dx 5e-6 --track 150e-6   # small CPU check
    python tools/lbm/keyhole_ti64.py M200 --scenario turn             # Martin turn point
    python tools/lbm/keyhole_ti64.py T3 --scenario ramp --ramp 100e-6
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks", "literature"))
import ti64_material as TM  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.backend import asnumpy  # noqa: E402
from tpmsfem.process import Laser, ScanVector  # noqa: E402
from tpmsfem.process.freesurface import FreeSurfaceMeltPool  # noqa: E402
from tpmsfem.process import defects as DF  # noqa: E402
from tpmsfem.process.keyhole import BubbleModel, RayTracer, keyhole_metrics  # noqa: E402
from tpmsfem.process.scanpath import ramp_down, turn_point  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "benchmarks", "literature", "data", "gan2021_keyhole_xray.json")
CASES = {"T1": (129.36, 0.4), "T2": (209.302, 1.1), "T3": (260.174, 0.6), "T4": (312.5, 0.9),
         "T5": (364.826, 0.7), "T6": (468.023, 0.7)}
MARTIN = {"M100": (100.0, 1.0), "M200": (200.0, 1.0), "M300": (300.0, 1.0)}     # 50 um beam, turn points
R0 = 47.5e-6
R0_MARTIN = 25e-6


def ke_steady(P, v, r0, T0=25.0):
    """Gan et al.'s keyhole number of a steady track on Ti-6Al-4V (liquid properties)."""
    d = TM.GAN2021
    kw = dict(eta_min=d["eta_min"], rho=d["rho_liquid"], cp=830.0, alpha=33.4 / (d["rho_liquid"] * 830.0),
              T_l=d["T_liquidus_K"] - 273.15, T0=T0)
    return float(DF.keyhole_state(P, v, r0, **kw)["Ke"]), kw


def measured(P, v):
    for r in json.load(open(DATA))["rows"]:
        if r["material"] == "Ti-6Al-4V" and abs(r["P_W"] - P) < 0.05 and abs(r["v_m_s"] - v) < 1e-6 and r["r0_m"] == R0:
            return r
    raise KeyError((P, v))


def build(args):
    if args.backend:
        tf.set_backend(args.backend)
    if args.case in CASES:
        P, v = CASES[args.case]
    elif args.case in MARTIN:
        P, v = MARTIN[args.case]
    else:
        P, v = args.P, args.v
    r0 = args.r0 or (R0_MARTIN if args.case in MARTIN else R0)
    scen = args.scenario or ("turn" if args.case in MARTIN else "track")
    meas = measured(P, v) if args.case in CASES else None
    e_exp = meas["e_m"] if meas else float(DF.keyhole_depth(ke_steady(P, v, r0)[0], r0))
    dx = args.dx
    turn = scen.startswith("turn")
    accel = args.accel
    T_dec = v / accel
    if turn:                                     # full speed, deceleration to the reversal, and back
        run_len = args.approach * v + 0.5 * v * T_dec
        lx = args.lead + run_len + args.tail
        e_exp = 2.0 * e_exp                      # the keyhole deepens in the turn
    else:
        lx = args.lead + args.track + args.tail
    if args.half_width is None:                  # 3 radii + 30 um either side of the track(s)
        args.half_width = max(100e-6, 3.0 * r0 + 30e-6) if args.case in MARTIN else 150e-6
    if args.plate is None and turn:              # Martin et al.: 500 um plates; 400 um holds the turn keyhole
        args.plate = 400e-6
    plate = args.plate or max(250e-6, 1.5 * e_exp + 80e-6)
    ly = 2 * args.half_width + (args.hatch if turn else 0.0)
    n = (int(round(lx / dx)) + 2, int(round(ly / dx)) + 2, int(round((plate + args.gas) / dx)) + 2)
    npl = int(round(plate / dx))
    metal = np.zeros(n, bool)
    metal[1:-1, 1:-1, 1:npl + 1] = True
    d = TM.GAN2021
    dt = args.dt or dx / 250.0
    md = FreeSurfaceMeltPool(TM.material(), dx, metal, dt=dt, viscosity=d["viscosity_Pa_s"], sigma=d["sigma_N_m"],
                             mushy_C=1e9, T0=25.0, T_max=9000.0, dsigma_dT=d["dsigma_dT"], evaporation=TM.evaporation(),
                             recoil_factor=0.54, emissivity=d["emissivity"], energy_transport="lbm")
    md.rays = RayTracer(A0=d["eta_min"], k_over_n=args.k_over_n)
    md.ray_every = args.ray_every
    if not args.no_bubbles:
        md.bubbles = BubbleModel(gamma=1.0)
        md.bubble_every = args.bubble_every
    if args.recoil_cap is not None:
        md.recoil_cap = args.recoil_cap
    md.lbm.smagorinsky = args.smagorinsky
    y = dx + 0.5 * ly
    x0 = dx + args.lead
    extra = {}
    if scen == "track":
        path = [ScanVector(x0, y, x0 + args.track, y, v)]
    elif scen == "ramp":
        path = ramp_down(x0, y, x0 + args.track, y, v, P, args.ramp, n=20)
        extra.update(ramp=args.ramp)
    elif turn:
        y = dx + args.half_width                  # first track; the return track is a hatch higher
        x_turn = x0 + run_len
        power = None
        if scen == "turn-ke":
            Ke0, kw = ke_steady(P, v, r0)
            floor = 0.2 * P
            power = lambda t, vv: float(np.clip(DF.keyhole_power(Ke0, max(vv, 1e-3), r0, **kw), floor, P))  # noqa: E731
            extra.update(Ke_target=Ke0)
        path, t_start = turn_point(x_turn, y, v, args.hatch, accel, approach=args.approach, depart=args.depart,
                                   dt_seg=args.seg_time, power=power)
        extra.update(accel=accel, hatch=args.hatch, x_turn=x_turn, t_turn=-t_start,
                     approach=args.approach, depart=args.depart)
    else:
        raise ValueError(scen)
    z_surf = (npl + 1) * dx
    info = dict(case=args.case, scenario=scen, P=P, v=v, r0=r0, dx=dx, dt=dt, shape=list(n), plate=plate,
                z_surface=z_surf, y=y, measured_depth=meas["e_m"] if meas else None,
                measured_eta=meas["eta"] if meas else None, Ke=ke_steady(P, v, r0)[0], **extra)
    return md, path, Laser(P, 0.0, r0), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--P", type=float)
    ap.add_argument("--v", type=float)
    ap.add_argument("--r0", type=float, default=None, help="1/e^2 beam radius [m]")
    ap.add_argument("--scenario", default=None, choices=["track", "ramp", "turn", "turn-ke"])
    ap.add_argument("--ramp", type=float, default=100e-6, help="ramp-down length [m] (scenario ramp)")
    ap.add_argument("--accel", type=float, default=1.11e3, help="mirror deceleration [m/s^2] (turns)")
    ap.add_argument("--hatch", type=float, default=50e-6)
    ap.add_argument("--approach", type=float, default=100e-6, help="time at full speed before the turn [s]")
    ap.add_argument("--depart", type=float, default=200e-6, help="time at full speed after the turn [s]")
    ap.add_argument("--seg-time", type=float, default=5e-6, help="time per path segment in turns [s]")
    ap.add_argument("--dx", type=float, default=2.5e-6)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--track", type=float, default=500e-6)
    ap.add_argument("--lead", type=float, default=100e-6)
    ap.add_argument("--tail", type=float, default=150e-6)
    ap.add_argument("--half-width", type=float, default=None, help="[m] default 150 um; 100 um for M cases")
    ap.add_argument("--plate", type=float, default=None)
    ap.add_argument("--gas", type=float, default=50e-6)
    ap.add_argument("--cool", type=float, default=100e-6, help="cooling time after the track [s]")
    ap.add_argument("--every", type=int, default=0, help="diagnostics every N steps (default: 1 us)")
    ap.add_argument("--ray-every", type=int, default=10)
    ap.add_argument("--bubble-every", type=int, default=10)
    ap.add_argument("--k-over-n", type=float, default=1.0)
    ap.add_argument("--recoil-cap", type=float, default=None)
    ap.add_argument("--smagorinsky", type=float, default=0.15, help="sub-grid (LES) constant; 0: off")
    ap.add_argument("--no-bubbles", action="store_true")
    ap.add_argument("--checkpoint", type=float, default=20.0, help="minutes between checkpoints")
    ap.add_argument("--redo", action="store_true", help="rerun a case that has already finished")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "keyhole_runs"))
    args = ap.parse_args()
    md, path, laser, info = build(args)
    os.makedirs(args.out, exist_ok=True)
    tag = (f"{args.case}_dx{args.dx * 1e6:g}um" if info["scenario"] == "track"
           else f"{args.case}_{info['scenario']}_dx{args.dx * 1e6:g}um")
    js, npz, ck = (os.path.join(args.out, f"{tag}{s}") for s in (".json", "_sections.npz", "_checkpoint.npz"))
    if not args.redo and not os.path.exists(ck) and os.path.exists(js) and json.load(open(js)).get("finished"):
        print(f"{tag}: already finished ({js}); --redo to run it again", flush=True)
        return
    log, sections = [], []
    if os.path.exists(ck):
        md.load_state(ck)
        if os.path.exists(js):
            log = json.load(open(js)).get("log", [])
        if os.path.exists(npz):
            z = np.load(npz)
            sections = [(float(t), z["fill"][i], z["T"][i]) for i, t in enumerate(z["t"])]
        print(f"resumed at t = {md.t * 1e6:.1f} us", flush=True)
    every = args.every or max(1, int(round(1e-6 / info["dt"])))
    jc = int(round(info["y"] / info["dx"]))
    t_wall = [time.time(), time.time()]

    def save():
        info.update(log=log, finished=False)
        json.dump(info, open(js, "w"), indent=1)
        if sections:
            np.savez_compressed(npz, t=np.array([s[0] for s in sections]),
                                fill=np.stack([s[1] for s in sections]), T=np.stack([s[2] for s in sections]))

    last = {}

    def cb(m, t, x_l, y_l):
        row = dict(t=t, x_l=x_l, laser_on=x_l is not None)
        if x_l is not None:
            last.update(x=x_l, y=y_l)
            row.update(keyhole_metrics(m, x_l, y_l, info["r0"], info["z_surface"]))
            row.update({f"ray_{k}": v for k, v in m.ray_stats.items()})
        elif last:                                   # laser off: follow the closing keyhole where the beam stopped
            row.update(keyhole_metrics(m, last["x"], last["y"], info["r0"], info["z_surface"]))
            row["x_last"] = last["x"]
        if m.bubbles is not None:
            row["bubbles"] = [dict(volume_um3=b["volume"] * (m.dx * 1e6) ** 3, pressure=b["pressure"],
                                   centroid_um=[c * m.dx * 1e6 for c in b["centroid"]],
                                   depth_um=1e6 * info["z_surface"] - b["centroid"][2] * m.dx * 1e6,
                                   T_wall_max=b["T_wall_max"], frozen=b["frozen"], pore=b["pore"])
                              for b in m.bubbles.bubbles]
            row["bubbles_formed"], row["bubbles_released"] = m.bubbles.formed, m.bubbles.released
        T = m.T
        row.update(T_max=float(T.max()), u_max=float((m.lbm.u ** 2).sum(axis=0).max()) ** 0.5 * m.dx / m.dt,
                   limited=int(m.lbm.limited), recoil_clipped=int(m.recoil_clipped),
                   mass=float(m.lbm.total_mass()), wall_s=time.time() - t_wall[0])
        log.append(row)
        sections.append((t, asnumpy(m.lbm.fill[:, jc, :]).astype(np.float16), asnumpy(T[:, jc, :]).astype(np.float16)))
        if len(log) % 10 == 0:
            d = row.get("depth", float("nan"))
            print(f"t {t * 1e6:7.1f} us  depth {d * 1e6:6.1f} um  absorbed {row.get('ray_absorbed', float('nan')):.2f}  "
                  f"bubbles {len(row.get('bubbles', []))} (pores {sum(b['pore'] for b in row.get('bubbles', []))})  "
                  f"T_max {row['T_max']:6.0f} C  u_max {row['u_max']:5.1f} m/s  "
                  f"lim {row['limited']} clip {row['recoil_clipped']}  ({row['wall_s']:.0f} s)", flush=True)
        if time.time() - t_wall[1] > 60.0 * args.checkpoint:
            md.save_state(ck)
            save()
            t_wall[1] = time.time()

    print(f"{tag}: {info['P']} W, {info['v']} m/s, cells {info['shape']} ({np.prod(info['shape']) / 1e6:.2f} M), "
          f"dt {info['dt'] * 1e9:.0f} ns, tau {md.tau:.4f}", flush=True)
    res = md.run(laser, path, t_extra=args.cool, callback=cb, callback_every=every)
    info.update(log=log, finished=True, mass_error=res.mass_error, energy_balance=res.energy_balance,
                energy_residual=res.energy_residual, absorbed_J=res.absorbed, steps=res.steps, seconds=res.seconds,
                mlups=res.mlups)
    final = dict(flag=res.flag.astype(np.int8), fill=res.fill.astype(np.float16), T_peak=res.T_peak.astype(np.float16))
    np.savez_compressed(os.path.join(args.out, f"{tag}_final.npz"), **final)
    save()
    info["finished"] = True
    json.dump(info, open(js, "w"), indent=1)
    if os.path.exists(ck):
        os.remove(ck)
    print("wrote", js, flush=True)


if __name__ == "__main__":
    main()
