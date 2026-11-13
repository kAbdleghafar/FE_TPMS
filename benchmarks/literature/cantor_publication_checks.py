"""Checks needed before the CoCrFeMnNi gyroid study (cantor_gyroid_*.py) can be published.

1. Pulsed validation. The Renishaw AM250 of Piglione et al. (2018) exposes point by point:
   80 us on each point, then a laser-off move of 60 um. The study treated this as a continuous
   scan at 0.75 m/s. Here the track is re-run as point exposures, with a move-and-settle delay
   of 1, 20 and 50 us (the delay is not reported), and as continuous scans at the matching
   mean speeds. The phase field then runs at each fusion-boundary median.
2. Absorptivity. The validation track and the thin-wall gyroid window are re-run at other
   absorptivities. Does the down-facing contrast survive?
3. Phase-field numerics. The cell spacing is re-run against
   - the interface width W0;
   - the window width and the growth length;
   - the random seed;
   - the starting spacing: a front seeded with n cells either keeps them, or eliminates or
     splits cells. That gives the band of spacings the model can sustain, which a measured
     spacing should lie in.
4. Material parameters of the pseudo-binary: D, k and the anisotropy.
5. Residual strain. A 2 mm column is built with 0.1 mm voxels and 0.4 mm lumps (as in the
   study), with 0.2 mm lumps, and with 0.05 mm voxels.

    python benchmarks/literature/cantor_publication_checks.py --part pulsed
    python benchmarks/literature/cantor_publication_checks.py --part all     # ~10 h on 4 cores
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
from common import ResultStore  # noqa: E402

from tpmsfem.process import Laser, MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402
from tpmsfem.process.meltpool import fusion_zone  # noqa: E402

STORE = "benchmarks/literature/results/cantor_publication_checks.json"
FIG = "docs/img/cantor_publication_checks.png"
POINT_UM, EXPOSURE_US = 60.0, 80.0
DELAYS_US = (1.0, 20.0, 50.0)
ABSORPTIVITIES = (0.30, 0.45, 0.60)
WINDOW_ABS = (0.30, 0.50)
# fusion-boundary medians of the study (dT0 = 60 K)
SUPPORTED = (3.0e7, 0.0274)        # validation single track = thin and thick walls over solid
DOWN = (2.5e7, 0.010)              # thin-wall window, over powder
BAND_SUPPORTED = (0.4, 0.6, 0.75, 1.0, 1.5)
BAND_DOWN = (0.6, 1.0, 1.2, 1.5, 2.0)


# ------------------------------------------------------------------------- phase field
def pf(G, R, *, dT0=60.0, width=6e-6, W0=10e-9, grow=10e-6, height=2e-6, D=3e-9, k=0.74, eps4=0.02, seed=0,
       lam0=None, n_chunks=25, prominence=0.2):
    """2-D phase field at (G, R); spacing from boundary counting averaged over the last 40 %.
    ``lam0`` [m]: seed the front with cells of this spacing instead of random noise."""
    alloy = PF.DiluteAlloy(c_inf=40.0, k=k, freezing_range=dT0, D=D, gibbs_thomson=3.65e-7, eps4=eps4)
    m = PF.DirectionalPF(alloy, G, R, width, height, W0_m=W0, seed=seed, noise=0.02 if lam0 is None else 0.002)
    n0 = None
    if lam0 is not None:
        n0 = max(int(round(width / lam0)), 1)
        x = (np.arange(m.nx) + 0.5) / m.nx
        phi = np.asarray(m.phi)
        m.phi = m.xp.asarray(np.clip(phi + 0.3 * np.cos(2 * np.pi * n0 * x)[None, :] * (1.0 - phi ** 2), -1, 1))
    t0 = time.perf_counter()
    hist = []
    for _ in range(n_chunks):
        r = m.run(grow / R / n_chunks, record_every=500)
        hist.append(r.boundaries(prominence=prominence))
    late = [max(h, 1.0) for h in hist[int(0.6 * n_chunks):]]
    Lx = m.nx * m.dx * m.W0
    tips = r.tips()
    liq = r.c[r.phi < 0.0]
    sc = S.SolidificationConstants(freezing_range=dT0, D_liquid=D, gibbs_thomson=3.65e-7, partition=k)
    return dict(G=G, R=R, dT0=dT0, width_um=width * 1e6, W0_nm=W0 * 1e9, grow_um=grow * 1e6, height_um=height * 1e6,
                D=D, k=k, eps4=eps4, seed=seed, n0=n0, prominence=prominence, lam0_um=None if lam0 is None else Lx / n0 * 1e6,
                n_history=hist, n_final=hist[-1], spacing_um=Lx / float(np.mean(late)) * 1e6,
                cs_rel=float(np.nanmedian([t["cs"] for t in tips])) if tips else float("nan"),
                intercell_max_rel=float(liq.max()) if liq.size else float("nan"),
                D_over_V_in_W0=D / R / W0, hunt_um=float(S.primary_spacing(G, R, sc, "hunt")),
                seconds=time.perf_counter() - t0)


def pf_cached(store, key, G, R, **kw):
    if store.get(key) is None:
        print(f"  phase field {key} ...", flush=True)
        store.put(key, pf(G, R, **kw))
    o = store.get(key)
    seed = f" (seeded with {o['n0']} cells, {o['lam0_um']:.2f} um)" if o.get("n0") else ""
    print(f"  {key}: spacing {o['spacing_um']:.2f} um{seed}, cells {o['n_history'][0]:.0f} -> {o['n_final']:.0f}, "
          f"Hunt {o['hunt_um']:.2f}, {o['seconds'] / 60:.0f} min", flush=True)
    return o


# ------------------------------------------------------------------------ melt pool
def pulsed_path(x0, x1, point=POINT_UM * 1e-3, exposure=EXPOSURE_US * 1e-6, delay=1e-6):
    """Point exposures along y = 0: the beam dwells ``exposure`` on each point (a 1 nm vector),
    then moves to the next with the laser off in ``delay``."""
    eps = 1e-6
    pts = np.arange(x0, x1 + 1e-9, point)
    vecs = []
    for i, x in enumerate(pts):
        vecs.append(ScanVector(x, 0.0, x + eps, 0.0, eps / exposure))
        if i < len(pts) - 1:
            vecs.append(ScanVector(x + eps, 0.0, pts[i + 1], 0.0, (pts[i + 1] - x - eps) / delay, 0.0))
    return vecs


def track(path, absorptivity=CG.ABS):
    md = MeltPoolModel(CG.MAT, CG.H, length=1.2, width=0.2, depth=0.15, powder=CG.LAYER, T0=CG.T_PLATE,
                       emissivity=0.4, evaporation=CG.EVAP)
    t0 = time.perf_counter()
    res = md.run(Laser(CG.P_W, absorptivity, CG.R_SPOT, depth=CG.DELTA), path, t_extra=2e-4)
    mid = (res.x[:, None, None] > 0.4) & (res.x[:, None, None] < 0.8) & np.ones(res.T.shape, bool)
    fb = CG.fusion_boundary(res, mid)
    secs = [fusion_zone(res, xs) for xs in np.arange(0.4, 0.8001, 0.01)]
    w = [s[0] for s in secs if s[0] is not None]
    d = [s[1] for s in secs if s[1] is not None]
    return dict(G=CG.summary(fb["G"]), R=CG.summary(fb["R"]), cr=CG.summary(fb["cr"]),
                width_um=1e3 * float(np.mean(w)), width_um_range=[1e3 * min(w), 1e3 * max(w)],
                depth_um=1e3 * float(np.mean(d)), depth_um_range=[1e3 * min(d), 1e3 * max(d)],
                T_peak_max=float(res.T_peak.max()), energy_balance=float(res.energy_balance),
                seconds=time.perf_counter() - t0)


def track_cached(store, key, path, **kw):
    if store.get(key) is None:
        print(f"  melt pool {key} ...", flush=True)
        store.put(key, track(path, **kw))
    v = store.get(key)
    print(f"  {key}: pool {v['width_um']:.0f} x {v['depth_um']:.0f} um, G p50 {v['G']['p50']:.2e} K/m, "
          f"R p50 {v['R']['p50']:.4f} m/s, cooling {v['cr']['p50']:.1e} K/s", flush=True)
    return v


def part_pulsed(store):
    print("1. pulsed validation (Piglione et al. 2018: 0.52-0.64 um, mean 0.59)", flush=True)
    for dl in DELAYS_US:
        v = track_cached(store, f"pulsed|delay={dl:.0f}", pulsed_path(0.15, 1.05, delay=dl * 1e-6))
        pf_cached(store, f"pf|pulsed|delay={dl:.0f}", v["G"]["p50"], v["R"]["p50"])
    for dl in DELAYS_US[1:]:
        speed = POINT_UM / (EXPOSURE_US + dl) * 1e3                                       # mm/s
        v = track_cached(store, f"continuous|v={speed:.0f}", [ScanVector(0.15, 0.0, 1.05, 0.0, speed)])
        pf_cached(store, f"pf|continuous|v={speed:.0f}", v["G"]["p50"], v["R"]["p50"])


def part_absorptivity(store):
    print("2a. absorptivity, validation track", flush=True)
    for a in ABSORPTIVITIES:
        v = track_cached(store, f"absorb|A={a:.2f}", [ScanVector(0.15, 0.0, 1.05, 0.0, CG.V_MM_S)], absorptivity=a)
        pf_cached(store, f"pf|absorb|A={a:.2f}", v["G"]["p50"], v["R"]["p50"])


def part_window(store):
    print("2b. absorptivity, thin-wall gyroid window: supported vs down-facing", flush=True)
    for a in WINDOW_ABS:
        key = f"window|thin|A={a:.2f}"
        if store.get(key) is None:
            old = CG.ABS
            CG.ABS = a
            try:
                store.put(key, CG.run_window("thin", verbose=False))
            finally:
                CG.ABS = old
        w = store.get(key)
        for c in ("supported", "down-facing"):
            d = w[c]
            print(f"  A = {a:.2f} {c}: {d['n_columns']} columns, G {d['G']['p50']:.2e}, R {d['R']['p50']:.4f}",
                  flush=True)
            pf_cached(store, f"pf|window|A={a:.2f}|{c}", d["G"]["p50"], d["R"]["p50"])


def part_numerics(store):
    print("3. phase-field numerics", flush=True)
    for name, (G, R) in (("supported", SUPPORTED), ("down", DOWN)):
        pf_cached(store, f"pf|num|{name}|base", G, R)
        pf_cached(store, f"pf|num|{name}|seed=1", G, R, seed=1)
        pf_cached(store, f"pf|num|{name}|wide", G, R, width=12e-6, grow=20e-6 if name == "supported" else 15e-6)
    pf_cached(store, "pf|num|supported|W0=7", *SUPPORTED, W0=7e-9)
    pf_cached(store, "pf|num|supported|tall", *SUPPORTED, height=3e-6)


def part_band(store):
    print("3b. stability band (seeded spacings)", flush=True)
    for name, (G, R), lams in (("supported", SUPPORTED, BAND_SUPPORTED), ("down", DOWN, BAND_DOWN)):
        for lam in lams:
            pf_cached(store, f"pf|band|{name}|lam0={lam:.2f}", G, R, lam0=lam * 1e-6)


def part_material(store):
    print("4. pseudo-binary parameters (supported conditions)", flush=True)
    G, R = SUPPORTED
    pf_cached(store, "pf|mat|D=6e-9", G, R, D=6e-9)
    pf_cached(store, "pf|mat|k=0.81", G, R, k=0.81)
    # k = 0.88: the walls reach only ~1.15 c_inf, below the default 0.2 c_inf prominence; count at 0.05
    # (checked to give the same count as 0.2 at k = 0.74)
    pf_cached(store, "pf|mat|k=0.88|prom=0.05", G, R, k=0.88, prominence=0.05)
    pf_cached(store, "pf|mat|k=0.74|prom=0.05", G, R, prominence=0.05)
    pf_cached(store, "pf|mat|eps4=0.01", G, R, eps4=0.01)
    pf_cached(store, "pf|mat|eps4=0.04", G, R, eps4=0.04)


# ---------------------------------------------------------------------- residual strain
RES_CASES = {"vox0.1_lump0.4": (0.1, 0.4), "vox0.1_lump0.2": (0.1, 0.2), "vox0.05_lump0.4": (0.05, 0.4)}
RES_HEIGHT, RES_PLATE = 2.0, 0.5


def residual_stats(path, z_range=(0.5, 1.5)):
    d = np.load(path)
    c, mat = d["centroids"], d["material"]
    sel = (mat == 1) & (c[:, 2] > z_range[0]) & (c[:, 2] < z_range[1])
    out = {}
    for tag in ("on_plate", "removed"):
        e = d[f"eps_el_{tag}"][sel] * 1e6
        s = d[f"stress_{tag}"][sel]
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2)
                     + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        out[tag] = dict(ezz_p5_p95=[float(np.percentile(e[:, 2], 5)), float(np.percentile(e[:, 2], 95))],
                        exx_p5_p95=[float(np.percentile(e[:, 0], 5)), float(np.percentile(e[:, 0], 95))],
                        abs_ezz_mean=float(np.abs(e[:, 2]).mean()), vm_p50=float(np.median(vm)),
                        vm_p95=float(np.percentile(vm, 95)))
    peeq = d["peeq"][sel] * 100
    out["peeq_percent_p50_p95"] = [float(np.median(peeq)), float(np.percentile(peeq, 95))]
    out["n_cells"] = int(sel.sum())
    out["seconds"] = float(d["seconds"])
    return out


def part_residual(store, cases=None):
    import cantor_gyroid_residual as CR

    print("5. residual strain: voxel and lump size (2 mm column on a 0.5 mm plate)", flush=True)
    for name, (vox, lump) in RES_CASES.items():
        if cases and name not in cases:
            continue
        key = f"residual|{name}"
        if store.get(key) is not None:
            continue
        out = f"benchmarks/literature/results/cantor_residual_check_{name}.npz"
        if not os.path.exists(out):
            CR.VOX, CR.LUMP, CR.HEIGHT, CR.PLATE = vox, lump, RES_HEIGHT, RES_PLATE
            CR.OUT, CR.CKPT = out, out.replace(".npz", "_checkpoint.pkl")
            CR.run()
            if os.path.exists(CR.CKPT):
                os.remove(CR.CKPT)
        store.put(key, residual_stats(out))
    for name in RES_CASES:
        o = store.get(f"residual|{name}")
        if o:
            r = o["removed"]
            print(f"  {name}: removed eps_zz p5/p95 {r['ezz_p5_p95'][0]:.0f}/{r['ezz_p5_p95'][1]:.0f}, mean |eps_zz| "
                  f"{r['abs_ezz_mean']:.0f}, von Mises p50 {r['vm_p50']:.0f} MPa, peeq p50 "
                  f"{o['peeq_percent_p50_p95'][0]:.2f} %, {o['seconds'] / 60:.0f} min", flush=True)


PARTS = dict(pulsed=part_pulsed, absorptivity=part_absorptivity, window=part_window, numerics=part_numerics,
             band=part_band, material=part_material, residual=part_residual)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "figure", *PARTS])
    ap.add_argument("--cases", nargs="*", help="residual cases to run")
    a = ap.parse_args()
    store = ResultStore(STORE)
    if a.part == "residual":
        part_residual(store, a.cases)
    elif a.part in PARTS:
        PARTS[a.part](store)
    elif a.part == "all":
        for f in PARTS.values():
            f(store)
    if a.part in ("all", "figure"):
        import cantor_publication_figure as FG
        FG.figure(store)


if __name__ == "__main__":
    main()
