"""Mesh-resolution study of the FCC -> HCP / twinning results (Level 0, docs §7.5; Level 1a, §7.6).

Re-runs the stress-free compression of the graded CoCrFeMnNi gyroid column (Level 0: where
stacking faults can extend) and of the Fe50Mn30Co10Cr10 gyroid (Level 1a: how much HCP forms)
at a chosen voxel size, with exactly the models of ``cantor_gyroid_faults.py`` and
``femncocr_gyroid_hcp.py``: same geometry, platens, material, strain steps, orientation samples,
thresholds and kinetic laws. Two things differ, neither of which changes a result:

* the fault criterion is evaluated over chunks of elements, so the (elements x orientations)
  arrays never exist in full (at 0.025 mm they would take 11 GB each);
* results go to ``results/fcc_hcp_resolution_<level>_vox<h>.json``, never over the published
  0.1 mm files.

Run on the GPU for the fine meshes (the FE solve is 10-18x faster there at these sizes):

    TPMSFEM_BACKEND=cupy python benchmarks/literature/fcc_hcp_resolution.py run 0 0.05
    TPMSFEM_BACKEND=cupy python benchmarks/literature/fcc_hcp_resolution.py run 1a 0.025
    python benchmarks/literature/fcc_hcp_resolution.py gci              # convergence table

At 0.1 mm the output matches the published files (check with ``compare``).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_faults as GF  # noqa: E402
import cantor_gyroid_solidification as CG  # noqa: E402
import femncocr_gyroid_hcp as FH  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.backend import backend_name  # noqa: E402
from tpmsfem.post import fcc_faults as FF  # noqa: E402
from tpmsfem.post import hcp_kinetics as HK  # noqa: E402

OUT = os.path.join(GF.RES, "fcc_hcp_resolution_{level}_vox{vox}.json")
CHUNK = int(os.environ.get("FAULTS_CHUNK", "20000"))     # elements per post-processing chunk


def _compression_model(mesh, material):
    md = tf.SolidModel(mesh, material)
    z = mesh.points[:, 2]
    bot, top = np.flatnonzero(z < 1e-9), np.flatnonzero(z > z.max() - 1e-9)
    md.fix(bot)
    md.fix(top, (0, 1), 0.0)
    return md, top


def _von_mises(sig):
    return np.sqrt(0.5 * ((sig[:, 0] - sig[:, 1]) ** 2 + (sig[:, 1] - sig[:, 2]) ** 2 + (sig[:, 2] - sig[:, 0]) ** 2)
                   + 3.0 * (sig[:, 3] ** 2 + sig[:, 4] ** 2 + sig[:, 5] ** 2))


def _fault_stats(sig, plastic, thick, ori, vol):
    """Level 0 per-step quantities of cantor_gyroid_faults.run(), accumulated over element chunks."""
    acc = {}
    vsum = vol.sum()
    gmax = -np.inf
    n = len(sig)
    sums = {f"{h}": 0.0 for h in ("thin", "thick")}
    counts = {"thin": int((~thick).sum()), "thick": int(thick.sum())}
    for k in range(0, n, CHUNK):
        s = slice(k, min(k + CHUNK, n))
        sg, pl, th, v = sig[s], plastic[s], thick[s], vol[s]
        tau_ca = np.empty((len(sg), GF.N_ORI))
        g_byun = np.empty((len(sg), GF.N_ORI))
        for w, sel in (("thin", ~th), ("thick", th)):
            if sel.any():
                tau_ca[sel] = FF.leading_partial_stress(sg[sel], ori[w])
                g_byun[sel] = FF.fault_energy_capacity(sg[sel], ori[w], criterion="byun")
        tau_rd = FF.leading_partial_stress(sg, ori["random"])
        for name, tau in (("ca", tau_ca), ("random", tau_rd)):
            for tc in GF.TAU_C:
                f_el = (tau >= tc).mean(axis=1) * pl
                acc[f"frac_{name}_{tc:.0f}"] = acc.get(f"frac_{name}_{tc:.0f}", 0.0) + float((f_el * v).sum())
                acc[f"elems_{name}_{tc:.0f}"] = acc.get(f"elems_{name}_{tc:.0f}", 0.0) + float((f_el > 0).sum())
        acc["frac_byun_ca_30"] = acc.get("frac_byun_ca_30", 0.0) + float((((g_byun >= GF.GAMMA_BYUN).mean(axis=1) * pl) * v).sum())
        gmax = max(gmax, float(g_byun.max()))
        f360 = (tau_ca >= 360.0).mean(axis=1) * pl
        sums["thin"] += float(f360[~th].sum())
        sums["thick"] += float(f360[th].sum())
    row = {}
    for key, val in acc.items():
        row[key] = val / vsum if key.startswith("frac_") else val / n
    row["byun_gamma_max"] = gmax
    for h in ("thin", "thick"):
        row[f"frac_ca_360_{h}"] = sums[h] / max(counts[h], 1)
    return row


def run(level: str, vox: float):
    GF.VOX = vox                                                     # build_mesh() reads it
    t0 = time.time()
    mesh = GF.build_mesh()
    cen = mesh.centroids()
    nvox = int(mesh.n_cells)
    print(f"Level {level}, voxel {vox} mm: {nvox:,} voxels, backend {backend_name()} ({tf.device_info()})", flush=True)
    if level == "0":
        md, top = _compression_model(mesh, GF.material())
        ori = GF.orientations()
        thick = cen[:, 0] >= 0.5 * CG.L_CELL
        vol = np.full(len(cen), vox ** 3)
        strains = GF.STRAINS
    else:
        md, top = _compression_model(mesh, FH.material())
        laws = {s: HK.femncocr_lpbf(s, FH.E_MPA) for s in ("exponential", "sigmoidal")}
        strains = FH.STRAINS
    print(f"  {md.ndof:,} dofs; setup {time.time() - t0:.0f} s", flush=True)
    rows = []
    for eps in strains:
        ts = time.time()
        md.fix(top, (2,), -eps * GF.HEIGHT)
        info = md.solve(n_inc=2, tol=1e-6)
        t_solve = time.time() - ts
        peeq = md.plastic_strain()[1].mean(axis=1)
        F = abs(md.reactions()[top, 2].sum())
        row = dict(strain=eps, nominal_stress_MPa=float(F / CG.L_CELL ** 2), seconds_solve=t_solve,
                   newton=int(getattr(info, "iterations", 0)), krylov=int(getattr(info, "krylov_iterations", 0)),
                   converged=bool(getattr(info, "converged", True)),
                   peeq_mean=float(peeq.mean()), peeq_p95=float(np.percentile(peeq, 95)),
                   peeq_p99=float(np.percentile(peeq, 99)), peeq_max=float(peeq.max()))
        if level == "0":
            sig = md.stress().mean(axis=1)
            plastic = peeq > 1e-6
            vm = _von_mises(sig)
            row.update(plastic_volume_frac=float(plastic.mean()), vm_p50=float(np.median(vm)),
                       vm_p99=float(np.percentile(vm, 99)), vm_max=float(vm.max()))
            tp = time.time()
            row.update(_fault_stats(sig, plastic, thick, ori, vol))
            row["seconds_post"] = time.time() - tp
            print(f"  eps {eps * 100:4.2f} %: nominal {row['nominal_stress_MPa']:6.1f} MPa, plastic "
                  f"{row['plastic_volume_frac'] * 100:5.1f} %, faults (CA, 360) {row['frac_ca_360'] * 100:5.2f} %, "
                  f"(CA, 240) {row['frac_ca_240'] * 100:5.2f} %, Byun max {row['byun_gamma_max']:.1f} mJ/m2 | "
                  f"solve {t_solve:.0f} s, post {row['seconds_post']:.0f} s", flush=True)
        else:
            row["yielded"] = float((peeq > 1e-6).mean())
            for s, law in laws.items():
                f = law.fraction(peeq)
                row[f"hcp_mean_{s}"] = float(f.mean())
                row[f"hcp_p95_{s}"] = float(np.percentile(f, 95))
                row[f"hcp_max_{s}"] = float(f.max())
                row[f"hcp_above10_{s}"] = float((f >= 0.10).mean())
            print(f"  eps {eps * 100:4.1f} %: nominal {row['nominal_stress_MPa']:6.1f} MPa, yielded "
                  f"{100 * row['yielded']:5.1f} %, PEEQ p95 {100 * row['peeq_p95']:5.2f} %, HCP mean "
                  f"{100 * row['hcp_mean_exponential']:5.2f} / {100 * row['hcp_mean_sigmoidal']:5.2f} % | "
                  f"solve {t_solve:.0f} s", flush=True)
        rows.append(row)
        out = dict(level=level, voxel_mm=vox, n_voxels=nvox, ndof=int(md.ndof),
                   relative_density=float(nvox * vox ** 3 / (CG.L_CELL ** 2 * GF.HEIGHT)),
                   backend=backend_name(), device=tf.device_info(), steps=rows, seconds=time.time() - t0,
                   complete=len(rows) == len(strains))
        json.dump(out, open(OUT.format(level=level, vox=f"{vox:g}"), "w"), indent=1)   # after every step
    print(f"wrote {OUT.format(level=level, vox=f'{vox:g}')} ({time.time() - t0:.0f} s)", flush=True)


def compare(level: str):
    """Largest difference between this driver at 0.1 mm and the published 0.1 mm results."""
    ref = json.load(open(GF.STORE if level == "0" else FH.STORE))["steps"]
    new = json.load(open(OUT.format(level=level, vox="0.1")))["steps"]
    worst = {}
    for a, b in zip(ref, new):
        for k, v in a.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in b:
                d = abs(b[k] - v) / max(abs(v), 1e-12)
                if d > worst.get(k, (0, 0))[0]:
                    worst[k] = (d, a["strain"])
    for k, (d, e) in sorted(worst.items(), key=lambda x: -x[1][0])[:12]:
        print(f"  {k:24s} max rel. diff {d:.1e} (at {e * 100:.2f} %)")


def gci(level: str, key: str, vox=(0.1, 0.05, 0.025), fs: float = 1.25):
    """Roache grid-convergence index on three meshes (refinement ratio 2): observed order p,
    Richardson extrapolation and GCI of the fine solution, per strain step."""
    runs = [{r["strain"]: r for r in json.load(open(OUT.format(level=level, vox=f"{h:g}")))["steps"]} for h in vox]
    print(f"Level {level}, {key}: coarse / medium / fine, observed order p, extrapolated, GCI_fine")
    rows = []
    for eps in runs[0]:
        if not all(eps in r for r in runs):
            continue
        f3, f2, f1 = (r[eps][key] for r in runs)                       # coarse, medium, fine
        e21, e32 = f2 - f1, f3 - f2
        p = np.log(abs(e32 / e21)) / np.log(2.0) if e21 != 0 and e32 != 0 and e32 / e21 > 0 else float("nan")
        ext = f1 + (f1 - f2) / (2.0 ** p - 1.0) if np.isfinite(p) and p > 0 else float("nan")
        g = fs * abs((f1 - f2) / f1) / (2.0 ** p - 1.0) if np.isfinite(p) and p > 0 and f1 != 0 else float("nan")
        rows.append(dict(strain=eps, coarse=f3, medium=f2, fine=f1, p=p, extrapolated=ext, gci_fine=g))
        print(f"  {eps * 100:5.2f} %: {f3:.4g} / {f2:.4g} / {f1:.4g}   p = {p:5.2f}   ext {ext:.4g}   GCI {g * 100:5.1f} %")
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run(sys.argv[2], float(sys.argv[3]))
    elif cmd == "compare":
        compare(sys.argv[2])
    elif cmd == "gci":
        for lv, keys in (("0", ("frac_ca_360", "frac_ca_240", "byun_gamma_max", "plastic_volume_frac", "nominal_stress_MPa")),
                         ("1a", ("hcp_mean_exponential", "hcp_p95_exponential", "peeq_p95", "nominal_stress_MPa"))):
            for k in keys:
                try:
                    gci(lv, k)
                except FileNotFoundError as e:
                    print("missing:", e.filename)
