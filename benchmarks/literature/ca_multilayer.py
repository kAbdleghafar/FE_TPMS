"""Grain structure of a multi-layer LPBF IN625 block from the 3-D grain CA: texture and grain
width against the scan rotation.

Process: the AM-Bench 2018 IN625 settings (195 W nominal, 179.2 W true, 800 mm/s; the
calibrated case-B melt pool, 123 x 36 um), 100 um hatch, 20 um layers, bidirectional tracks.
The scan direction rotates by 90 deg (x/y alternation) or 67 deg between layers.

Thermal input: the steady case-B track history (nist_ambench2018_tracks_ca.py section),
shifted along every track (grain_ca.track_events), so preheating between tracks and layers is
not included. The pool of the bare-plate track is placed with its top at the top of the
current layer, so it remelts 16 um of the layer below. The powder layer is not modelled: the
cells of a layer are added when the melt pool reaches them.

CA: 480 x 480 um at 2 um cells (240 x 240 um at 1 and 2 um cells as the resolution check),
40 um of substrate (Voronoi, 8 um grains, random orientations) plus 10 layers; KGT kinetics of
the Ni-Nb quasi-binary; no nucleation (see the case-B sensitivity).

Outputs per layer (the top 20 um slab of the deposit, below the last melt pool):
* texture: the share of the volume with a <100> within 15 deg of the build direction (BD)
  and of the first scan direction;
* grain width: area-weighted equivalent diameter of the grain sections in planes normal to BD.

    python benchmarks/literature/ca_multilayer.py run ROT [SIDE_um DX_um [ALPHA]]   # e.g. run 67 480 2 1
    python benchmarks/literature/ca_multilayer.py figure
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nist_ambench2018_tracks_ca as CB  # noqa: E402

from tpmsfem.process import grain_ca as CA  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

OUT = "benchmarks/literature/results/ca_multilayer.json"
FIELDS = "benchmarks/literature/results/ca_multilayer_{key}.npz"
FIG = "docs/img/ca_multilayer.png"
SUB, LAYER, N_LAYERS, HATCH = 40e-6, 20e-6, 10, 100e-6
MARGIN = 0.6e-3                               # track overrun beyond the block (pool length ~0.3 mm)


def key(rot_deg, side, dx, alpha=0.0):
    return f"rot{rot_deg:.0f}_side{side * 1e6:.0f}_dx{dx * 1e6:g}" + (f"_a{alpha:g}" if alpha else "")


def tracks_for_layer(layer, rot_deg, t0, speed, SIDE):
    """Bidirectional tracks covering the block, at angle layer * rot_deg, starting at t0."""
    th = np.radians((layer * rot_deg) % 360.0)
    d = np.array([np.cos(th), np.sin(th)])
    n = np.array([-d[1], d[0]])
    c = np.array([SIDE / 2, SIDE / 2])
    half_diag = SIDE / np.sqrt(2.0)
    reach = half_diag + 70e-6                 # pool half-width ~62 um
    offs = np.arange(-np.floor(reach / HATCH), np.floor(reach / HATCH) + 1) * HATCH
    z_top = SUB + (layer + 1) * LAYER
    tracks, t = [], t0
    length = 2 * (half_diag + MARGIN)
    for j, o in enumerate(offs):
        sgn = 1.0 if j % 2 == 0 else -1.0
        p0 = c + o * n - sgn * (half_diag + MARGIN) * d
        p1 = c + o * n + sgn * (half_diag + MARGIN) * d
        tracks.append(CA.Track(tuple(p0), tuple(p1), z_top, t))
        t += length / speed
    return tracks, t


def layer_stats(res, k_lo, k_hi):
    """Texture and grain width in the slab of cells k_lo <= k < k_hi."""
    g = res.grain[:, :, k_lo:k_hi]
    ok = g >= 0
    ids = g[ok]
    rot = res.orientations[ids]
    c15 = np.cos(np.radians(15.0))
    a_bd = S.alignment(rot, np.array([0.0, 0.0, 1.0]))
    a_sd = S.alignment(rot, np.array([1.0, 0.0, 0.0]))
    a_sd2 = S.alignment(rot, np.array([0.0, 1.0, 0.0]))
    # <110> along BD: the angle between BD and the nearest <110>
    v = np.abs(np.einsum("nij,j->ni", rot, np.array([0.0, 0.0, 1.0])))
    v = np.sort(v, axis=1)[:, ::-1]
    cos110 = (v[:, 0] + v[:, 1]) / np.sqrt(2.0)
    widths, areas = [], []
    from scipy.ndimage import label
    for k in range(g.shape[2]):
        sl = g[:, :, k]
        for gid in np.unique(sl[sl >= 0]):
            lab, n = label(sl == gid)
            if n == 0:
                continue
            cnt = np.bincount(lab.ravel())[1:]
            widths.extend(list(2.0 * np.sqrt(cnt / np.pi) * res.grid.dx * 1e6))
            areas.extend(list(cnt))
    w, a = np.array(widths), np.array(areas, float)
    return dict(frac_100_bd_15=float((a_bd >= c15).mean()), frac_100_sd_15=float((a_sd >= c15).mean()),
                frac_100_bd_and_sd_15=float(((a_bd >= c15) & ((a_sd >= c15) | (a_sd2 >= c15))).mean()),
                frac_110_bd_15=float((cos110 >= c15).mean()),
                mean_misalign_bd_deg=float(np.degrees(np.arccos(np.clip(a_bd, -1, 1))).mean()),
                grains=int(len(np.unique(ids))), width_area_weighted_um=float((w * a).sum() / a.sum()),
                width_median_um=float(np.median(w)))


def _ids(g):
    g = g.ravel()
    return g[g >= 0]


def run(rot_deg, SIDE=240e-6, DX=1e-6, alpha=0.0):
    sec = CB.section()
    kin = CB.kinetics()
    nz = int(round((SUB + N_LAYERS * LAYER + 10e-6) / DX))
    n = int(round(SIDE / DX))
    grid = CA.CAGrid((0.5 * DX, 0.5 * DX, 0.5 * DX), (n, n, nz), DX)
    rng = np.random.default_rng(0)
    z = grid.centres(2)
    mask = np.broadcast_to((z <= SUB)[None, None, :], grid.shape)
    grain, rot = CA.voronoi_substrate(grid, max(8e-6, 4 * DX), rng, mask=mask)
    ca = CA.GrainCA(grid, grain, rot, kin, seed=0, heat_flow_bias=alpha)
    t = 0.0
    out = dict(rot_deg=rot_deg, side_um=SIDE * 1e6, dx_um=DX * 1e6, heat_flow_bias=alpha, layers=[])
    t_wall = time.perf_counter()
    for layer in range(N_LAYERS):
        trs, t = tracks_for_layer(layer, rot_deg, t, sec.speed, SIDE)
        ev = CA.track_events(grid, sec, trs)
        ca.run(ev)
        t += 1e-3
        print(f"  rotation {rot_deg}: layer {layer + 1}/{N_LAYERS}, {len(trs)} tracks, {len(ev['cell'])} events, "
              f"{ca.steps} steps, {ca.n_grains} grains [{time.perf_counter() - t_wall:.0f} s]", flush=True)
    res = ca.result()
    for layer in range(N_LAYERS):
        k_lo = int(round((SUB + layer * LAYER) / DX))
        st = layer_stats(res, k_lo, k_lo + int(round(LAYER / DX)))
        st.update(layer=layer + 1, z_um=(SUB + (layer + 0.5) * LAYER) * 1e6)
        out["layers"].append(st)
        print(f"    layer {layer + 1}: <100>||BD {100 * st['frac_100_bd_15']:.0f} %, <110>||BD {100 * st['frac_110_bd_15']:.0f} %, "
              f"width {st['width_area_weighted_um']:.1f} um, {st['grains']} grains", flush=True)
    k_sub = int(round(SUB / DX)) - 10
    out["substrate"] = layer_stats(res, k_sub - 10, k_sub)
    out.update(steps=res.steps, seconds=time.perf_counter() - t_wall, ca_stats=res.stats)
    # fields for the figure: vertical sections and the top-layer plan, IPF along BD and SD
    kz = int(round((SUB + (N_LAYERS - 0.5) * LAYER) / DX))
    np.savez_compressed(FIELDS.format(key=key(rot_deg, SIDE, DX, alpha)),
                        xz_bd=CB._ipf3(res, res.grain[:, n // 2, :], (0.0, 0.0, 1.0)),
                        yz_bd=CB._ipf3(res, res.grain[n // 2, :, :], (0.0, 0.0, 1.0)),
                        plan_bd=CB._ipf3(res, res.grain[:, :, kz], (0.0, 0.0, 1.0)),
                        plan_sd=CB._ipf3(res, res.grain[:, :, kz], (1.0, 0.0, 0.0)),
                        top_rot=res.orientations[_ids(res.grain[:, :, int(round((SUB + 5 * LAYER) / DX)):kz][::4, ::4, ::4])]
                        .astype(np.float32), dx=np.array(DX))
    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    store[key(rot_deg, SIDE, DX, alpha)] = out
    json.dump(store, open(OUT, "w"), indent=1)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = [float(v) for v in sys.argv[2:]]
        run(a[0], *(v * 1e-6 for v in a[1:3]), *a[3:4])
    elif sys.argv[1] == "figure":
        import ca_multilayer_figure as F
        F.figure()
