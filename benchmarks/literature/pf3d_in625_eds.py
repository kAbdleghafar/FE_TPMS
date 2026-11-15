"""3-D multicomponent phase field of IN625 cells at the AM-Bench 2018 case-B mid-pool point,
with synthetic EDS maps (SEM and STEM) of Nb, Mo, Cr, Fe and Ni.

Conditions: the case-B centreline point 22.5 um below the surface used in pf_path_trapping.py
(G = 6.62e6 K/m, R = 0.092 m/s), IN625 of the AM-Bench feedstock as a dilute four-solute alloy
(tpmsfem.process.phasefield3d.IN625_AMBENCH: Nb, Mo with liquidus feedback; Cr, Fe partitioning
only), solute trapping with V_D = 1 m/s for every solute.

Jobs
  3d_W10   1.6 x 1.6 um periodic box, 1.2 um window, 5 um of growth, W0 = 10 nm (D/V = 3.3 W0;
           at this point the 2-D spacing is within 8 % of the W0 = 4.1 nm value, 5.3 of
           docs/process_models.md). <100> along the build direction.
  2d_W10   the same model in 2-D (one cell thick, 2.4 um wide, 6 um of growth): the 2-D/3-D
           comparison with identical physics and resolution.

    python benchmarks/literature/pf3d_in625_eds.py run 3d_W10      # checkpoints every chunk; resumes
    python benchmarks/literature/pf3d_in625_eds.py summarise 3d_W10  # extract from the last checkpoint
    python benchmarks/literature/pf3d_in625_eds.py figure

Checkpoints go to $PF3D_CKPT_DIR (default: the system temp dir); the full 3-D microstructure
(several hundred MB) stays there, and a compact extract (sections and a sub-block) is written
to results/.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

from tpmsfem.process import phasefield3d as P3

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
STORE = os.path.join(RES, "pf3d_in625_eds.json")
FIELDS = os.path.join(RES, "pf3d_in625_eds_{job}.npz")
FIG = os.path.join(RES, "pf3d_in625_eds.png")
FIG3D = os.path.join(RES, "pf3d_in625_eds_3d.png")
CKPT_DIR = os.environ.get("PF3D_CKPT_DIR", os.path.join(tempfile.gettempdir(), "pf3d_in625_ckpt"))

G, R, Z_UM = 6.618e6, 0.09211, -22.5          # case-B centreline point (pf_path_trapping.path_points)
V_D = 1.0
ALLOY = P3.IN625_AMBENCH.at_velocity(R, V_D)
JOBS = {
    # tag: (W0, (Lx, Ly), window height, growth, ny override, chunks)
    "3d_W10": (10e-9, (1.6e-6, 1.6e-6), 1.2e-6, 5.0e-6, None, 40),
    "2d_W10": (10e-9, (2.4e-6, 8e-9), 1.2e-6, 6.0e-6, 1, 10),
    "3d_W6.5": (6.5e-9, (1.2e-6, 1.2e-6), 1.2e-6, 4.0e-6, None, 80),
}


def model(tag):
    W0, size, h, grow, ny, chunks = JOBS[tag]
    return P3.DirectionalPF3D(ALLOY, G, R, size, h, W0_m=W0, ny=ny, seed=0)


def run(tag):
    W0, size, h, grow, ny, chunks = JOBS[tag]
    os.makedirs(CKPT_DIR, exist_ok=True)
    ck = os.path.join(CKPT_DIR, f"{tag}.npz")
    m = model(tag)
    done = 0
    if os.path.exists(ck):
        m.restore(ck)
        done = int(round(m.t * m.tau0 / (grow / R / chunks)))
        print(f"{tag}: resumed at chunk {done}/{chunks}, t = {m.t * m.tau0 * 1e6:.3f} us", flush=True)
    print(f"{tag}: grid {m.nx} x {m.ny} x {m.nz}, W0 = {m.W0 * 1e9:.1f} nm, D/V = {ALLOY.D / R / m.W0:.2f} W0", flush=True)
    t0 = time.time()
    while done < chunks:
        m.run(grow / R / chunks, record_every=500, adapt_dt=True)
        done += 1
        m.checkpoint(ck)
        print(f"{tag}: chunk {done}/{chunks}, tip {m.front[-1][1] * 1e6:.3f} um, {(time.time() - t0) / 3600:.2f} h",
              flush=True)
    r = m.result()
    summarise(tag, r)
    print(f"{tag}: done", flush=True)


def summarise(tag, r):
    """Metrics to the json store; a compact extract (sections, EDS maps, a coarsened 3-D block)
    to an npz. EDS maps are computed here from the full-resolution fields."""
    W0, size, h, grow, ny, chunks = JOBS[tag]
    names = r.names + [r.balance]
    tip = r.tip_row()
    dx = r.dx_m
    nsolid = r.solid[r.names[0]].shape[0] if r.solid else 0
    out = dict(tag=tag, z_um=Z_UM, G=G, R=R, V_D=V_D, W0_nm=W0 * 1e9, D_over_V_in_W0=ALLOY.D / R / W0,
               size_um=[s * 1e6 for s in size], height_um=h * 1e6, grown_um=float(r.front_history[-1][1] * 1e6),
               steps=r.steps, grid=[r.params["nx"], r.params["ny"], r.params["nz"]], dx_nm=dx * 1e9,
               freezing_range_K=ALLOY.freezing_range, k_bar=ALLOY.k_bar, solutes=r.params["solutes"])
    rows = slice(nsolid // 3, max(nsolid, 1))       # beyond the initial transient
    out["segregation_wt"] = {n: r.segregation(n, rows) for n in names}
    grown = {n: r.grown(n) for n in names}
    ext = dict(dx=dx, tip=tip, nsolid=nsolid, names=np.array(names))
    for n in (names if r.params["ny"] > 1 else ("Nb", "Mo")):
        ext[f"xz_{n}"] = grown[n][:, grown[n].shape[1] // 2, :].astype(np.float16)        # longitudinal section
    ext["xz_phi"] = r.grown_phi()[:, r.phi.shape[1] // 2, :].astype(np.float16)
    if r.params["ny"] == 1:
        from scipy.signal import find_peaks
        nb = grown["Nb"][:, 0, :]
        counts = []
        for d in np.linspace(0.3e-6, 0.6e-6, 7):        # boundaries crossed below the tip (as in 5.3)
            row = nb[tip - int(round(d / dx))] / ALLOY.solutes[0].c_inf
            e3 = np.concatenate([row, row, row])
            pk, _ = find_peaks(e3, prominence=0.2)
            counts.append(int(np.sum((pk >= len(row)) & (pk < 2 * len(row)))))
        n_b = float(np.mean(counts))
        out["spacing_2d_um"] = nb.shape[1] * dx * 1e6 / n_b if n_b else None
        out["boundaries_2d"] = n_b
    else:
        sp = []
        for d in (0.4e-6, 0.6e-6, 0.8e-6, 1.0e-6, 1.5e-6, 2.0e-6, 3.0e-6):
            row = tip - int(round(d / dx))
            if row >= 0:
                _, n_c, lam = r.cells(row)
                sp.append(dict(depth_um=d * 1e6, cells=n_c, spacing_um=lam * 1e6))
        out["spacing_3d"] = sp
        out.update(eds_extract(grown, dx, tip, ext))
        # coarsened 3-D views: Nb in the lowest 2 um below the tip, and the front in the window
        lo = max(tip - int(round(2.0e-6 / dx)), 0)
        ext["block_Nb"] = grown["Nb"][lo:tip + 4:2, ::2, ::2].astype(np.float16)
        ext["block_lo"] = lo
        ext["window_phi"] = r.phi[::2, ::2, ::2].astype(np.float16)
        ext["window_offset_rows"] = nsolid
    store = json.load(open(STORE)) if os.path.exists(STORE) else {}
    store[tag] = out
    json.dump(store, open(STORE, "w"), indent=1)
    np.savez_compressed(FIELDS.format(job=tag), **ext)
    print(json.dumps({k: v for k, v in out.items() if k in ("spacing_3d", "spacing_2d_um", "grown_um", "eds_contrast")}),
          flush=True)


#: EDS settings: SEM at 15 kV (a common setting) and 5 kV (low-kV mapping), STEM on a 100 nm foil
EDS_MODES = {"sem15": dict(kind="sem", beam_kV=15.0, total_counts=1500.0, pixel_m=16e-9),
             "sem5": dict(kind="sem", beam_kV=5.0, total_counts=600.0, pixel_m=16e-9),
             "stem": dict(kind="stem", foil_m=100e-9, beam_kV=200.0, total_counts=800.0, pixel_m=8e-9)}
SECTION_DEPTH = 0.8e-6          # transverse section, below the leading tips


def eds_extract(grown, dx, tip, ext):
    """Synthetic EDS maps of a transverse section (bulk below it) and a longitudinal section,
    with and without counting noise, and the Nb contrast each technique retains."""
    from tpmsfem.post import eds

    zs = tip - int(round(SECTION_DEPTH / dx))
    ext["xy_true_z"] = zs
    for n, g in grown.items():
        ext[f"xy_true_{n}"] = g[zs].astype(np.float16)
    lo = max(tip - int(round(3.0e-6 / dx)), 0)                   # block for the longitudinal maps
    blk = {n: g[lo:tip + 1] for n, g in grown.items()}
    ext["xz_lo"] = lo
    info, clean_nb = {}, {}
    for mode, cfg in EDS_MODES.items():
        cfg = dict(cfg)
        kind = cfg.pop("kind")
        fn = eds.sem_eds if kind == "sem" else eds.stem_eds
        for sec, (fields, idx, axis) in {"xy": (grown, zs, 0), "xz": (blk, blk["Nb"].shape[1] // 2, 1)}.items():
            clean = fn(fields, dx, idx, axis=axis, **{k: v for k, v in cfg.items() if k != "total_counts"})
            noisy = fn(fields, dx, idx, axis=axis, seed=1, **cfg)
            if sec == "xy":
                clean_nb[mode] = clean.maps["Nb"].astype(float)
            for n in grown:
                ext[f"eds_{mode}_{sec}_{n}"] = noisy.maps[n].astype(np.float16)
                ext[f"eds_{mode}_{sec}_clean_{n}"] = clean.maps[n].astype(np.float16)
            ext[f"eds_{mode}_{sec}_pixel"] = clean.pixel_m
            if sec == "xy":
                info[mode] = dict(lines={n: clean.lines[n][0] for n in grown},
                                  resolution_nm={n: clean.resolution_m[n] * 1e9 for n in grown})
    # Nb contrast retained: spread (p95 - p05) of the noise-free map / that of the true section
    true = grown["Nb"][zs].astype(float)
    t_spread = np.percentile(true, 95) - np.percentile(true, 5)
    contrast = {}
    for mode in EDS_MODES:
        a = clean_nb[mode]
        contrast[mode] = float((np.percentile(a, 95) - np.percentile(a, 5)) / t_spread)
    return dict(eds=info, eds_contrast=contrast, section_depth_um=SECTION_DEPTH * 1e6,
                nb_true_section_wt=dict(p05=float(np.percentile(true, 5)), p95=float(np.percentile(true, 95))))


def summarise_checkpoint(tag):
    m = model(tag)
    m.restore(os.path.join(CKPT_DIR, f"{tag}.npz"))
    summarise(tag, m.result())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "figure"
    if cmd == "run":
        run(sys.argv[2])
    elif cmd == "summarise":
        summarise_checkpoint(sys.argv[2])
    elif cmd == "figure":
        import pf3d_in625_eds_figure as F
        F.figure()
