"""NIST AM-Bench 2018 AMB2018-02: single laser tracks on bare IN625 plate (melt-pool scale).

Measurements (NIST AMMT testbed; https://www.nist.gov/ambench/amb2018-02-description and the
CHAL-AMB2018-02-MP-xsection / -MP-length / -CR result pages):

* Yb fibre laser 1070 nm, spot D4sigma = 170 um (FWHM 100 um) -> Gaussian 1/e^2 radius 85 um;
  true powers A 137.9 W / 400 mm/s, B 179.2 W / 800 mm/s, C 179.2 W / 1200 mm/s
  (the commanded 150 / 195 / 195 W were mis-calibrated); bare, bead-blasted IN625 plate at ~25 C.
* melt-pool cross-section width / depth at mid-track (optical, class mean +- SD over traces);
* melt-pool length (solidus 1290 C, NIR thermography);
* surface cooling rate on the centreline, 1290 -> 1190 C.

Protocol: case B is NIST's calibration case. Two parameters of the conduction model are
calibrated on its width and depth -- the absorptivity A and an absorption depth delta over
which the absorbed power is deposited (exponential in depth; the usual effective treatment of
a shallow vapour depression, peak intensity ~1.6 MW/cm^2 here) -- and cases A and C, the
lengths and the cooling rates are predictions. Surface evaporation caps the peak temperature
near boiling. (Enhancing the liquid conductivity in depth instead needs a factor of ~7 to
reach the measured depth; the trials are kept in the result store.)

    python benchmarks/literature/nist_ambench2018_tracks.py --calibrate
    python benchmarks/literature/nist_ambench2018_tracks.py --predict
    python benchmarks/literature/nist_ambench2018_tracks.py --report
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import in625_material as IM  # noqa: E402
from common import ResultStore  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector  # noqa: E402
from tpmsfem.process.meltpool import cooling_rate_history, fusion_zone, pool_length  # noqa: E402

W_BEAM = 0.085                                   # 1/e^2 radius [mm] (D4sigma 170 um)
CASES = {  # power [W], speed [mm/s]; measured width, depth [um] (mean, SD); length [um]; cooling rate [K/s]
    "A": dict(P=137.9, v=400.0, width=(147.9, 3.7), depth=(42.5, 1.7), length=(300, 12), cr=(1.16e6, 2.68e5)),
    "B": dict(P=179.2, v=800.0, width=(123.5, 6.5), depth=(36.0, 1.9), length=(359, 20), cr=(1.08e6, 5.88e5)),
    "C": dict(P=179.2, v=1200.0, width=(106.0, 1.4), depth=(29.6, 0.6), length=(370, 10), cr=(1.90e6, 5.52e5)),
}
STORE = "benchmarks/literature/results/ambench2018_tracks.json"
MAT = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())


def run_case(case, A, f_z, h=0.005, track=1.5, verbose=False, delta=0.0):
    c = CASES[case]
    x0, x1 = 0.15, 0.15 + track
    md = MeltPoolModel(MAT, h, length=x1 + 0.25, width=0.3, depth=0.2, T0=25.0,
                       liquid_k_factor=(1.0, 1.0, f_z), emissivity=0.4, evaporation=Evaporation())
    t_end_scan = (x1 - x0) / c["v"]
    x_probe = x0 + 0.7 * track
    res = md.run(Laser(c["P"], A, W_BEAM, depth=delta), [ScanVector(x0, 0.0, x1, 0.0, c["v"])], t_extra=0.3e-3,
                 probes={"centre": (x_probe, 0.0)}, snapshot_times=[t_end_scan - 1e-9], verbose=verbose)
    w, d = fusion_zone(res, x0 + 0.75 * track)
    wl, dl = fusion_zone(res, x0 + 0.75 * track, level=res.T_liquidus)
    L = pool_length(res.snapshots[0][1], res) if res.snapshots else np.nan
    t, Tp = res.probes["t"], res.probes["centre"]
    return dict(width=1e3 * w, depth=1e3 * d, width_liquidus=1e3 * wl, depth_liquidus=1e3 * dl, length=1e3 * L,
                cr100=float(cooling_rate_history(t, Tp, 1290.0, 1190.0)),
                cr290=float(cooling_rate_history(t, Tp, 1290.0, 1000.0)),
                T_peak_max=float(res.T_peak.max()), steps=res.steps, seconds=res.seconds,
                energy_balance=float(res.energy_balance), A=A, f_z=f_z, h=h, delta=delta)


def key_of(case, A, f_z, h, delta=0.0):
    return f"{case}|A={A:.4f}|fz={f_z:.3f}|h={h}|evap" + (f"|d={delta:.4f}" if delta else "")


def get(store, case, A, f_z, h=0.005, delta=0.0):
    key = key_of(case, A, f_z, h, delta)
    r = store.get(key)
    if r is None:
        r = run_case(case, A, f_z, h, delta=delta)
        store.put(key, r)
        print(f"  {key}: width {r['width']:.1f} um, depth {r['depth']:.1f} um, length {r['length']:.0f} um, "
              f"cr {r['cr100']:.2e} K/s, T_max {r['T_peak_max']:.0f} C  [{r['steps']} steps, {r['seconds']:.0f}s]",
              flush=True)
    return r


def calibrate(store, h=0.005, f_z=1.0):
    """Fit (A, delta) to case B's width and depth: for each absorption depth a secant search on
    A matches the width; delta is then interpolated to the depth and A refined."""
    tw, td = CASES["B"]["width"][0], CASES["B"]["depth"][0]

    def match_width(delta, A0=0.2, A1=0.26):
        r0, r1 = get(store, "B", A0, f_z, h, delta), get(store, "B", A1, f_z, h, delta)
        for _ in range(6):
            if abs(r1["width"] - tw) < 0.5:
                break
            A2 = round(float(np.clip(A1 + (tw - r1["width"]) * (A1 - A0) / (r1["width"] - r0["width"]), 0.05, 0.95)), 4)
            A0, r0, A1, r1 = A1, r1, A2, get(store, "B", A2, f_z, h, delta)
        return A1, r1

    pts = []
    for delta in (0.0, 0.02):
        A, r = match_width(delta)
        pts.append((delta, A, r["depth"]))
        print(f"delta = {1e3 * delta:.0f} um: A = {A:.3f} -> width {r['width']:.1f} um (target {tw}), "
              f"depth {r['depth']:.1f} um (target {td})", flush=True)
    for _ in range(3):
        (d0, A0_, z0), (d1, A1_, z1) = pts[-2], pts[-1]
        dn = round(float(np.clip(d1 + (td - z1) * (d1 - d0) / (z1 - z0), 0.0, 0.08)), 4)
        A, r = match_width(dn, A1_, A1_ * 1.03)
        pts.append((dn, A, r["depth"]))
        print(f"delta = {1e3 * dn:.1f} um: A = {A:.3f} -> width {r['width']:.1f} um, depth {r['depth']:.1f} um", flush=True)
        if abs(r["depth"] - td) < 0.5:
            break
    dn, A, _ = pts[-1]
    r = get(store, "B", A, f_z, h, dn)
    store.put("calibration", dict(A=A, f_z=f_z, delta=dn, h=h, width=r["width"], depth=r["depth"]))
    print(f"calibrated: A = {A:.3f}, delta = {1e3 * dn:.1f} um -> B width {r['width']:.1f} um, depth {r['depth']:.1f} um")


def contour_track(store):
    """Melt-pool width of the contour scan of the roughness benchmark (180 W, 1600 mm/s) with the
    case-B calibration (IN625 properties and the AMMT beam as a stand-in for IN718 on the
    Concept Laser M2): sets the bead radius of the surface model."""
    cal = store.get("calibration")
    if store.get("contour_180W_1600") is None:
        CASES["contour"] = dict(P=180.0, v=1600.0)
        r = run_case("contour", cal["A"], cal["f_z"], cal["h"], delta=cal.get("delta", 0.0))
        del CASES["contour"]
        store.put("contour_180W_1600", r)
    r = store.get("contour_180W_1600")
    print(f"contour 180 W / 1600 mm/s: width {r['width']:.1f} um, depth {r['depth']:.1f} um")


def report(store):
    cal = store.get("calibration")
    if cal is None:
        print("run --calibrate first")
        return
    lines = [f"Calibrated on case B: absorptivity A = {cal['A']:.3f}, absorption depth "
             f"{1e3 * cal.get('delta', 0.0):.1f} um (h = {1e3 * cal['h']:.0f} um).\n",
             "| case | P, v | quantity | measured (NIST AMMT) | tpmsfem | diff |", "|---|---|---|---:|---:|---:|"]
    for case, c in CASES.items():
        r = store.get(key_of(case, cal["A"], cal["f_z"], cal["h"], cal.get("delta", 0.0)))
        if r is None:
            continue
        tag = "calibration" if case == "B" else "prediction"
        for q, lab, unit, sc in (("width", "melt-pool width", "µm", 1), ("depth", "melt-pool depth", "µm", 1),
                                 ("length", "melt-pool length (solidus)", "µm", 1),
                                 ("cr", "cooling rate 1290→1190 °C", "K/s", 1)):
            m, sd = c[q]
            p = r["cr100"] if q == "cr" else r[q]
            fmt = (lambda v: f"{v:.2e}") if q == "cr" else (lambda v: f"{v:.1f}")
            calib = " (calibrated)" if case == "B" and q in ("width", "depth") else ""
            lines.append(f"| {case} ({tag}) | {c['P']} W, {c['v']:.0f} mm/s | {lab}{calib} | {fmt(m)} ± {fmt(sd)} {unit} | "
                         f"{fmt(p)} {unit} | {100 * (p / m - 1):+.0f} % |")
    txt = "\n".join(lines)
    print(txt)
    open("benchmarks/literature/results/ambench2018_tracks.md", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--contour", action="store_true")
    ap.add_argument("--try", dest="trial", nargs=2, type=float, metavar=("A", "F"))
    ap.add_argument("--h", type=float, default=0.005)
    a = ap.parse_args()
    store = ResultStore(STORE)
    print(tf.device_info(), flush=True)
    if a.trial:
        get(store, "B", a.trial[0], a.trial[1], a.h)
    if a.calibrate:
        calibrate(store, a.h)
    if a.predict:
        cal = store.get("calibration")
        for case in CASES:
            get(store, case, cal["A"], cal["f_z"], cal["h"], cal.get("delta", 0.0))
    if a.contour:
        contour_track(store)
    if a.report or a.predict:
        report(store)
