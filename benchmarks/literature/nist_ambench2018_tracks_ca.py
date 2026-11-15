"""AM-Bench 2018-02 case B: 3-D grain cellular automaton against NIST's EBSD section.

The CA (tpmsfem.process.grain_ca: decentred octahedra, KGT tip kinetics, optional
nucleation) regrows the melt pool of the calibrated case-B track (179.2 W, 800 mm/s) in 3-D,
from the steady-state thermal history of the melt-pool model, and is compared with the
measured transverse section (the data and the placement of nist_ambench2018_tracks_ebsd.py:
0.28 um pixels, fan centre, local surface).

Substrate, two variants:
* "extruded": the measured section, extended unchanged along the scan direction. This is
  the best case for a single section: every grain the melt pool can grow from is measured.
* "hybrid": the measured orientations within +-delta of the section plane (delta = half the
  measured median substrate grain size, as in the 3-D trace test), elsewhere a statistically
  equivalent substrate: 3-D Voronoi grains of the measured median size (2.7 um; 5
  realisations) or of the area-weighted size (7.9 um; 3 realisations), whose orientations are
  drawn from the measured substrate.

Kinetics: KGT for the Ni-Nb quasi-binary of the phase-field study (c0 = 4 wt%, k = 0.48,
m = -10.5 K/wt%, D = 3e-9 m^2/s, Gamma = 3.65e-7 K m): 31 K tip undercooling at 0.1 m/s,
consistent with the phase-field tips. Nucleation off, then a sensitivity run.

Grid: 0.5 um cells, 50 um along the scan (the section 40 um from the upstream end: grains
reach the section from at most ~11 um behind it in the 3-D trace test), 150 um wide, 45 um
deep.

    python benchmarks/literature/nist_ambench2018_tracks_ca.py section   # melt pool (~5 min), cached
    python benchmarks/literature/nist_ambench2018_tracks_ca.py run       # CA runs and comparison
    python benchmarks/literature/nist_ambench2018_tracks_ca.py figure
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpmsfem.process import ebsd as E  # noqa: E402
from tpmsfem.process import grain_ca as CA  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

SECTION = "benchmarks/literature/results/ambench2018_tracks_section.npz"
OUT = "benchmarks/literature/results/ambench2018_tracks_ca.json"
FIELDS = "benchmarks/literature/results/ambench2018_tracks_ca_fields.npz"
FIG = "docs/img/ca_ambench_caseB.png"
X0_MM, V_MMS, XSTART_MM = 0.7, 800.0, 0.15
DX = 0.5e-6
# EBSD sample frame (X right, Y down, Z out of the section) -> model frame (x scan, y, z up):
# X = y, Y = -z and, for a right-handed frame, Z = -x. Crystal axes (rows) transform as
# R_model = R_ebsd T^T. Whether the laser moved along +x or -x relative to the section's
# viewing side is not recorded, so both track directions are run.
T_EBSD = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
ALLOY = dict(c0=4.0, k=0.48, m=-10.5, D=3e-9, gibbs_thomson=3.65e-7)


def section():
    """Steady-state section of the case-B melt pool (cached)."""
    if os.path.exists(SECTION):
        d = np.load(SECTION)
        return CA.TrackSection(d["l"], d["z"], d["T_peak"], d["tau_m"], d["tau_l"], d["cooling_rate"],
                               float(d["T_melt"]), float(d["speed"]))
    import nist_ambench2018_tracks_ebsd as EB

    res, i = EB.simulate_section()
    sec = CA.TrackSection.from_meltpool(res, X0_MM, V_MMS, XSTART_MM)
    # steady-state check: the delays one section upstream
    up = CA.TrackSection.from_meltpool(res, X0_MM - 0.1, V_MMS, XSTART_MM)
    melted = sec.T_peak >= sec.T_melt
    drift = float(np.nanmax(np.abs(up.tau_l - sec.tau_l)[melted]))
    print(f"section: tau_l {sec.tau_l[melted].min() * 1e6:.0f}-{sec.tau_l[melted].max() * 1e6:.0f} us after the laser; "
          f"change over 0.1 mm upstream {drift * 1e6:.1f} us", flush=True)
    np.savez_compressed(SECTION, l=sec.l, z=sec.z, T_peak=sec.T_peak, tau_m=sec.tau_m, tau_l=sec.tau_l,
                        cooling_rate=sec.cooling_rate, T_melt=sec.T_melt, speed=sec.speed, steady_drift_s=drift)
    return sec


def ebsd_frame():
    """Binned EBSD orientations, valid mask, step [um], local surface row per column, fan
    centre [um] and the model's fusion zone on the pixel grid (as in the 2-D test)."""
    import nist_ambench2018_tracks_ebsd as EB
    from scipy.ndimage import median_filter

    EB.fetch()
    m = E.read_ctf(EB.CTF)
    g, valid, step = EB.binned(m)
    ny, nx = valid.shape
    prev = json.load(open(EB.OUT))
    top = np.array([np.argmax(valid[:, j]) if valid[:, j].any() else 0 for j in range(nx)], float)
    top = median_filter(top, size=21, mode="nearest")
    return g, valid, step, top, float(prev["fan_centre_um"])


def kinetics():
    return CA.TipKinetics.kgt(ALLOY["c0"], ALLOY["k"], ALLOY["m"], ALLOY["D"], ALLOY["gibbs_thomson"])


def build(sec, g_px, valid, step, top, xc, variant="extruded", seed=0, delta_um=None, d_sub_um=None, direction=1,
          dx=DX):
    """CA grid, initial grains and orientations (model frame), events, and the index of the
    section plane (40 um downstream of the upstream end of the grid)."""
    nxc, nyc, nzc = int(round(50e-6 / dx)), int(round(150e-6 / dx)), int(round(45e-6 / dx))
    x_s = 40e-6 if direction > 0 else (nxc * dx - 40e-6)
    grid = CA.CAGrid((0.0, -75e-6 + 0.5 * dx, -nzc * dx + 0.5 * dx), (nxc, nyc, nzc), dx)
    i_s = int(round(x_s / dx))
    y, z = grid.centres(1), grid.centres(2)
    ny_px, nx_px = valid.shape
    col = np.clip(np.round((y * 1e6 + xc) / step - 0.5).astype(int), 0, nx_px - 1)
    row = np.round(top[col][:, None] + (-z[None, :] * 1e6) / step - 0.5).astype(int)
    inside = (row >= 0) & (row < ny_px)
    rowc = np.clip(row, 0, ny_px - 1)
    colc = np.broadcast_to(col[:, None], rowc.shape)
    ok = inside & valid[rowc, colc]
    pid = (rowc * nx_px + colc).astype(np.int32)              # grain id = pixel id
    rot = g_px.reshape(-1, 3, 3) @ T_EBSD.T
    g2 = np.where(ok, pid, -1)
    grain = np.broadcast_to(g2[None], grid.shape).copy()
    if variant == "hybrid":
        rng = np.random.default_rng(seed)
        Y2, Z2 = np.meshgrid(y, z, indexing="ij")
        melted2, *_ = sec.sample(Y2, Z2)
        sub = ok & ~melted2.reshape(ok.shape)
        pool = rot[pid[sub]]
        gv, rv = CA.voronoi_substrate(grid, d_sub_um * 1e-6, rng, orientations=pool)
        far = np.abs(grid.centres(0) - x_s)[:, None, None] > delta_um * 1e-6
        far = far & (grain >= 0)
        grain = np.where(far, gv + len(rot), grain).astype(np.int32)
        rot = np.concatenate([rot, rv])
    a, b = x_s - direction * 0.55e-3, x_s + direction * 0.25e-3
    tracks = [CA.Track(start=(a, 0.0), end=(b, 0.0), z_top=0.0, t0=0.0)]
    ev = CA.track_events(grid, sec, tracks)
    return grid, grain, rot, ev, i_s


def compare(res, i_s, grid, g_px, valid, step, top, xc, sec):
    """Point-wise disorientation and grain statistics in the model fusion zone of the section."""
    ny_px, nx_px = valid.shape
    X = (np.arange(nx_px) + 0.5) * step
    Yp = (np.arange(ny_px) + 0.5) * step
    depth = Yp[:, None] - top[None, :] * step
    ym = (X[None, :] - xc) * 1e-6 * np.ones((ny_px, 1))
    zm = -depth * 1e-6
    melted, *_ = sec.sample(ym, zm)
    fz = melted.reshape(ny_px, nx_px) & (depth >= 0) & valid
    y0, z0 = grid.origin[1], grid.origin[2]
    j = np.clip(np.round((ym - y0) / grid.dx).astype(int), 0, grid.shape[1] - 1)
    k = np.clip(np.round((zm - z0) / grid.dx).astype(int), 0, grid.shape[2] - 1)
    gid = res.grain[i_s][j, k]
    ok = fz & (gid >= 0)
    pred = np.zeros_like(g_px)
    pred[ok] = res.orientations[gid[ok]] @ T_EBSD                # back to the EBSD frame
    mis = E.misorientation(g_px[ok], pred[ok])
    stats = {}
    for name, gg in (("measured", g_px), ("CA", pred)):
        lab = E.segment_grains(gg, ok, threshold=10.0, min_size=10)
        sh = E.grain_shapes(lab, (step, step))
        a = sh["area"]
        al = S.alignment(gg[ok], np.array([0.0, -1.0, 0.0]))
        stats[name] = dict(grains=int(lab.max() + 1), area_weighted_d_eq_um=float((sh["d_eq"] * a).sum() / a.sum()),
                           frac_100_build_15deg=float((al >= np.cos(np.radians(15))).mean()))
    return dict(fusion_zone_px=int(ok.sum()), frac_lt10=float((mis < 10).mean()), frac_lt5=float((mis < 5).mean()),
                median_deg=float(np.median(mis)), stats=stats), pred, ok


def run():
    sec = section()
    g_px, valid, step, top, xc = ebsd_frame()
    prev3d = json.load(open("benchmarks/literature/results/ambench2018_tracks_ebsd3d.json"))
    delta, d_sub = prev3d["delta_um"], prev3d["substrate_median_d_eq_um"]
    kin = kinetics()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
    cases = [("extruded", None, 0, 1), ("extruded, reversed scan", None, 0, -1)]
    cases += [("extruded, nucleation 1e14/m3 at 10 +- 2 K", (1e14, 10.0, 2.0), 0, 1),
              ("extruded, nucleation 1e15/m3 at 30 +- 2 K", (1e15, 30.0, 2.0), 0, 1)]
    cases += [(f"hybrid {s}", None, s, 1) for s in range(5)]
    d_aw = prev3d["substrate_area_weighted_d_eq_um"]
    cases += [(f"hybrid, {d_aw:.1f} um grains {s}", None, s, 1) for s in range(3)]
    cases += [("extruded, 0.25 um cells", None, 0, 1), ("extruded, fast kinetics (5 K at 0.1 m/s)", None, 0, 1)]
    fast = CA.TipKinetics.power_law(0.1 / 5.0 ** 2, 2.0, dT_max=200.0)
    for name, nuc, seed, direction in cases:
        if name in out:
            continue
        variant = "hybrid" if name.startswith("hybrid") else "extruded"
        d = d_aw if "um grains" in name else d_sub
        grid, grain, rot, ev, i_s = build(sec, g_px, valid, step, top, xc, variant, seed, delta, d, direction,
                                          dx=0.25e-6 if "0.25 um" in name else DX)
        t0 = time.perf_counter()
        res = CA.GrainCA(grid, grain, rot, fast if "fast" in name else kin, nucleation=nuc, seed=seed).run(ev).result()
        cmp_, pred, ok = compare(res, i_s, grid, g_px, valid, step, top, xc, sec)
        fz3 = res.melted
        gz = res.grain[fz3]
        cmp_.update(name=name, events=int(len(ev["cell"])), melted_cells=int(fz3.sum()), steps=res.steps,
                    seconds=time.perf_counter() - t0, ca_stats=res.stats,
                    nucleated_fraction=float(res.nucleated[gz[gz >= 0]].mean()) if (gz >= 0).any() else 0.0)
        print(f"{name}: {100 * cmp_['frac_lt10']:.1f} % within 10 deg, grains {cmp_['stats']['CA']['grains']} "
              f"(measured {cmp_['stats']['measured']['grains']}), d_eq {cmp_['stats']['CA']['area_weighted_d_eq_um']:.1f} um "
              f"(measured {cmp_['stats']['measured']['area_weighted_d_eq_um']:.1f}), nucleated {100 * cmp_['nucleated_fraction']:.1f} % "
              f"[{res.steps} steps, {cmp_['seconds']:.0f} s]", flush=True)
        out[name] = cmp_
        json.dump(out, open(OUT, "w"), indent=1)
        tag = {"extruded": "extruded", "hybrid 0": "hybrid", "extruded, nucleation 1e15/m3 at 30 +- 2 K": "nuc"}.get(name)
        if tag:
            ipf = S.ipf_color(np.where(ok[..., None, None], pred, np.eye(3)), axis=(0.0, -1.0, 0.0))
            fields[tag + "_section_ipf"] = np.where(ok[..., None], ipf, 1.0).astype(np.float16)
            # longitudinal section (x-z through the track centre) and plan view at z = -5 um, IPF along z
            gzc = res.grain[:, grid.shape[1] // 2, :]
            fields[tag + "_long_ipf"] = _ipf3(res, gzc, (0.0, 0.0, 1.0))
            kz = int(round((-5e-6 - grid.origin[2]) / grid.dx))
            fields[tag + "_plan_ipf"] = _ipf3(res, res.grain[:, :, kz], (0.0, 0.0, 1.0))
            np.savez_compressed(FIELDS, **fields)
    return out


def _ipf3(res, gid, axis):
    """IPF colours of a 2-D slice of grain ids along a sample axis (model frame)."""
    rgb = np.ones(gid.shape + (3,))
    ok = gid >= 0
    rgb[ok] = S.ipf_color(res.orientations[gid[ok]], axis=axis)
    return rgb.astype(np.float16)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "section":
        section()
    elif cmd == "run":
        run()
    elif cmd == "figure":
        import nist_ambench2018_tracks_ca_figure as F
        F.figure()
