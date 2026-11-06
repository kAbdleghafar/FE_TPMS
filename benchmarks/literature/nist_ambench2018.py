"""NIST AM-Bench 2018 (AMB2018-01): IN625 LPBF bridge -- residual stress & distortion.

Benchmark data (all public, NIST / UC Davis / LLNL / Cornell):

* geometry: AMB2018_01_Part.STL (75 x 5 x 12.5 mm bridge, 12 legs, 11 ridges)
* part deflection after EDM-cutting the 12 legs from the plate (end block stays
  attached): 1.276 mm at ridge 1 (part AMB2018-01-625-CBM-B1-P3), CHAL-AMB2018-01-PD
* contour-method sigma_xx map at x = 31 mm (part + plate strip,
  AMB2018-01-625-CBM-B2-P3; Hill Engineering / UC Davis; E = 207 GPa, nu = 0.278)
* neutron-diffraction elastic strains eps_xx, eps_yy, eps_zz (NIST NCNR BT8, 1.5 mm gauge)
* synchrotron EDXRD elastic strains eps_xx, eps_zz (CHESS), data doi:10.18434/mds2-3052

Protocol (standard for AM-Bench): ONE number is used for calibration -- the
ridge-1 deflection -- and the residual stress/strain fields are predicted.

Model: elastic inherent-strain layer-by-layer build (0.5 mm super-layers),
eps* = s (-1, -1, -kappa) in the build frame (in-plane isotropic because the
scan direction alternates between x and y every layer). Because the elastic
model is linear, two unit runs (in-plane, vertical) give the response for any
(s, kappa) by superposition; s is calibrated to the deflection for each kappa.

Variants: ``plate`` (h = 0.5 mm, elastic 81 x 12 x 13 mm IN625 plate strip,
bolted during the build, then cut free) and ``rigid`` (h = 0.25 mm, part welded
to a rigid plate) -- the first resolves the plate, the second the thin legs.

    python benchmarks/literature/nist_ambench2018.py --variant plate
    python benchmarks/literature/nist_ambench2018.py --variant rigid
    python benchmarks/literature/nist_ambench2018.py --report
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

import tpmsfem as tf
from tpmsfem.fem import constitutive as C
from tpmsfem.geometry import box
from tpmsfem.mesh import voxelize_stl
from tpmsfem.residual.am import AMProcess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "results/literature/ambench2018"
DATA = os.path.join(OUT, "data")
URLS = {
    "part.stl": "https://s3.amazonaws.com/nist-midas/1889/AMB2018_01_Part.STL",
    "contour_L7.txt": "https://www.nist.gov/document/amb2018-01-625-cbm-b2-p3contourl7stressestxt",
    "neutron.txt": "https://data.nist.gov/od/ds/ark:/88434/mds2-3052/IN625%20neutron%20RS%20results/IN625_Neutron_Strain_Results.txt",
    "xray.txt": "https://data.nist.gov/od/ds/ark:/88434/mds2-3052/IN625%20X-ray%20RS%20results/IN625_EDXRD_Strain_Results.txt",
}
IN625 = tf.Material("IN625", 207_000.0, 0.278)          # constants used by the measurement teams
DEFLECTION_RIDGE1 = 1.276                               # mm, CHAL-AMB2018-01-PD
PLATE = dict(x=(-3.0, 78.0), y=(-4.5, 7.5), thickness=13.0)   # strip 80.72 x 12.19 x 12.85 mm
CONTOUR_X, CONTOUR_Y0, CONTOUR_Z0 = 31.0, 4.5, 12.85    # section x; part offset in the contour frame
LEG_CUT_X = 55.0                                        # legs are at x < 54; end block at x >= 56
RIDGES_X = [0.5 + 7 * k for k in range(11)]
UNITS = {"inplane": (-1e-3, -1e-3, 0.0, 0, 0, 0), "vertical": (0.0, 0.0, -1e-3, 0, 0, 0)}


def fetch():
    os.makedirs(DATA, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            print("downloading", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "tpmsfem-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:
                fh.write(r.read())
    return {k: os.path.join(DATA, k) for k in URLS}


def build_mesh(variant, files):
    if variant == "plate":
        h = 0.5
        lo = (PLATE["x"][0], PLATE["y"][0], -PLATE["thickness"])
        hi = (PLATE["x"][1], PLATE["y"][1], 12.5)
        plate = box(lo, (PLATE["x"][1], PLATE["y"][1], 0.0))
        mesh = voxelize_stl(files["part.stl"], h, lo=lo, hi=hi, extra_fields=[(2, plate)])
    else:
        h = 0.25
        mesh = voxelize_stl(files["part.stl"], h, lo=(0, 0, 0), hi=(75, 5, 12.5))
    return mesh, h


def run_unit(variant, unit, files, solver="auto"):
    mesh, h = build_mesh(variant, files)
    md = tf.SolidModel(mesh, {1: IN625, 2: IN625})
    print(f"[{variant}/{unit}] {mesh.summary()}", flush=True)
    sub = np.flatnonzero(mesh.cell_data["material"] == 2) if variant == "plate" else None
    am = AMProcess(md, layer_height=0.5, inherent_strain=UNITS[unit], substrate=sub, solver=solver)
    t0 = time.perf_counter()
    am.build(verbose=False)
    if variant == "plate":
        am.remove_from_plate(verbose=False)          # strip cut from the 100 mm plate / unbolted
    as_built = md.stress().mean(axis=1)              # element-average stress (ne, 6)
    u_built = md.displacement().copy()
    # EDM: cut the 12 legs at the plate interface (end block stays attached)
    c = mesh.centroids()
    if variant == "plate":
        kerf = np.flatnonzero((mesh.cell_data["material"] == 1) & (c[:, 2] < h) & (c[:, 0] < LEG_CUT_X))
        md.deactivate(kerf)
    else:
        bot = np.flatnonzero((mesh.points[:, 2] < 1e-9) & (mesh.points[:, 0] < LEG_CUT_X))
        md.release(bot)
    md.solve(solver=solver, name="leg cut")
    du = md.displacement() - u_built
    p = mesh.points
    defl = []
    for xr in RIDGES_X:
        ridge = np.flatnonzero((np.abs(p[:, 2] - 12.5) < 1e-9) & (np.abs(p[:, 0] - xr) <= 0.5 + 1e-9)
                               & (p[:, 1] >= -1e-9) & (p[:, 1] <= 5 + 1e-9))
        d = du[ridge, 2].mean()
        if variant == "plate":   # CMM heights are relative to the plate top beside the part
            side = np.flatnonzero((np.abs(p[:, 2]) < 1e-9) & (np.abs(p[:, 0] - xr) <= 1.0)
                                  & ((p[:, 1] < -0.5) | (p[:, 1] > 5.5)))
            d -= du[side, 2].mean()
        defl.append(float(d))
    out = os.path.join(OUT, f"{variant}_{unit}.npz")
    np.savez_compressed(out, centroids=c, material=mesh.cell_data["material"], stress=as_built,
                        deflection=np.array(defl), h=h, seconds=time.perf_counter() - t0)
    print(f"[{variant}/{unit}] ridge deflections (unit eps*): {np.round(defl, 4).tolist()}  "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ analysis
def elastic_strain(s):
    E, nu = IN625.E, IN625.nu
    tr = s[:, 0] + s[:, 1] + s[:, 2]
    return ((1 + nu) * s - nu * tr[:, None] * np.r_[1, 1, 1, 0, 0, 0][None, :]) / E


def sample(cent, values, pts, half):
    """Average ``values`` over element centroids inside the gauge box centre +- half
    (NaN where the gauge contains no material of the model)."""
    out = np.full((len(pts), values.shape[1]), np.nan)
    for i, q in enumerate(pts):
        sel = np.all(np.abs(cent - q) <= half + 1e-9, axis=1)
        if sel.any():
            out[i] = values[sel].mean(axis=0)
    return out


MAX_ELASTIC_STRAIN = 5e-3     # measured |elastic strain| above this is rejected (IN625 yield strain ~0.4 %)


def clean(meas):
    meas = np.asarray(meas, float).copy()
    meas[np.abs(meas) > MAX_ELASTIC_STRAIN] = np.nan
    return meas


def metrics(pred, meas):
    ok = np.isfinite(pred) & np.isfinite(meas)
    p, m = pred[ok], meas[ok]
    return dict(n=int(ok.sum()), rmse=float(np.sqrt(np.mean((p - m) ** 2))), bias=float(np.mean(p - m)),
                r=float(np.corrcoef(p, m)[0, 1]), meas_rms=float(np.sqrt(np.mean(m**2))))


def analyse(variant, files, kappas=(0.0,)):
    """kappa (vertical/in-plane inherent strain ratio) has no effect: with
    strain-free layer birth a purely vertical eigenstrain is compatible and
    produces neither stress nor deflection (verified by the 'vertical' unit
    run), so the calibration is a single parameter."""
    U = {u: np.load(os.path.join(OUT, f"{variant}_{u}.npz")) for u in UNITS}
    cent, h = U["inplane"]["centroids"], float(U["inplane"]["h"])
    contour = np.loadtxt(files["contour_L7.txt"], comments="%")
    neutron = np.loadtxt(files["neutron.txt"], skiprows=1)            # X Z eyy exx ezz
    xray = np.loadtxt(files["xray.txt"], skiprows=1)                  # X Z exx ezz e45 gxz [microstrain]
    part_c = contour[contour[:, 1] > CONTOUR_Z0 + 0.1]
    cpts = np.column_stack([np.full(len(part_c), CONTOUR_X), part_c[:, 0] - CONTOUR_Y0, part_c[:, 1] - CONTOUR_Z0])
    npts = np.column_stack([neutron[:, 0], np.full(len(neutron), 2.5), neutron[:, 1]])
    xpts = np.column_stack([xray[:, 0], np.full(len(xray), 2.5), xray[:, 1]])
    part = U["inplane"]["material"] == 1
    res = {}
    for kappa in kappas:
        d = U["inplane"]["deflection"] + kappa * U["vertical"]["deflection"]
        s = DEFLECTION_RIDGE1 / d[0]
        sig = s * (U["inplane"]["stress"] + kappa * U["vertical"]["stress"])
        eps = elastic_strain(sig)
        cp = sample(cent[part], sig[part][:, [0]], cpts, np.array([h / 2, h / 2, h / 2]))[:, 0]
        plate_c = contour[contour[:, 1] < CONTOUR_Z0 - 0.1]
        ppts = np.column_stack([np.full(len(plate_c), CONTOUR_X), plate_c[:, 0] - CONTOUR_Y0, plate_c[:, 1] - CONTOUR_Z0])
        pp = (sample(cent[~part], sig[~part][:, [0]], ppts, np.array([h / 2, h / 2, h / 2]))[:, 0]
              if (~part).any() else np.full(len(plate_c), np.nan))
        nz = sample(cent[part], eps[part], npts, np.array([0.75, 0.75, 0.75]))
        xz = sample(cent[part], eps[part], xpts, np.array([max(0.125, h / 2), 0.75, max(0.125, h / 2)]))
        res[f"kappa={kappa:g}"] = dict(
            inherent_strain=[-s * 1e-3, -s * 1e-3, -kappa * s * 1e-3],
            deflection=(s * d).tolist(),
            contour_sxx=metrics(cp, part_c[:, 2]),
            contour_sxx_plate=metrics(pp, plate_c[:, 2]) if np.isfinite(pp).any() else None,
            contour_plate_points=dict(z=(plate_c[:, 1] - CONTOUR_Z0).tolist(), y=(plate_c[:, 0] - CONTOUR_Y0).tolist(),
                                      measured=plate_c[:, 2].tolist(), predicted=pp.tolist()),
            neutron_exx=metrics(nz[:, 0], clean(neutron[:, 3])), neutron_eyy=metrics(nz[:, 1], clean(neutron[:, 2])),
            neutron_ezz=metrics(nz[:, 2], clean(neutron[:, 4])),
            neutron_exx_body=metrics(nz[neutron[:, 1] >= 7, 0], clean(neutron[neutron[:, 1] >= 7, 3])),
            neutron_exx_legs=metrics(nz[neutron[:, 1] < 5, 0], clean(neutron[neutron[:, 1] < 5, 3])),
            xray_exx=metrics(xz[:, 0], clean(xray[:, 2] * 1e-6)), xray_ezz=metrics(xz[:, 2], clean(xray[:, 3] * 1e-6)),
            neutron_points=dict(x=neutron[:, 0].tolist(), z=neutron[:, 1].tolist(), exx_meas=clean(neutron[:, 3]).tolist(),
                                exx_pred=nz[:, 0].tolist(), ezz_meas=clean(neutron[:, 4]).tolist(), ezz_pred=nz[:, 2].tolist()),
            profile=dict(z=(part_c[:, 1] - CONTOUR_Z0).tolist(), y=(part_c[:, 0] - CONTOUR_Y0).tolist(),
                         measured=part_c[:, 2].tolist(), predicted=cp.tolist()),
        )
    return res


def report(files):
    lines = []
    for variant in ("plate", "rigid"):
        if not all(os.path.exists(os.path.join(OUT, f"{variant}_{u}.npz")) for u in UNITS):
            continue
        res = analyse(variant, files)
        json.dump(res, open(os.path.join(OUT, f"analysis_{variant}.json"), "w"), indent=1)
        h = float(np.load(os.path.join(OUT, f"{variant}_inplane.npz"))["h"])
        lines.append(f"\n### variant `{variant}` (h = {h} mm)\n")
        lines.append("| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        r = res["kappa=0"]
        lines.append(f"| calibrated in-plane inherent strain | | | | | eps*_xx = eps*_yy = {r['inherent_strain'][0]:.2e} |")
        for key, label, sc, u in (("contour_sxx", "sigma_xx at x = 31 mm (contour, UC Davis)", 1, "MPa"),
                                  ("contour_sxx_plate", "sigma_xx in the plate strip at x = 31 mm (contour)", 1, "MPa"),
                                  ("neutron_exx", "eps_xx (neutron, NIST NCNR)", 1e6, "µε"),
                                  ("neutron_exx_body", "  eps_xx, solid body z >= 7 mm (neutron)", 1e6, "µε"),
                                  ("neutron_exx_legs", "  eps_xx, legs z < 5 mm (neutron)", 1e6, "µε"),
                                  ("neutron_eyy", "eps_yy (neutron)", 1e6, "µε"),
                                  ("neutron_ezz", "eps_zz (neutron)", 1e6, "µε"),
                                  ("xray_exx", "eps_xx (synchrotron EDXRD, CHESS)", 1e6, "µε"),
                                  ("xray_ezz", "eps_zz (synchrotron EDXRD)", 1e6, "µε")):
            m = r[key]
            if m is None:
                continue
            lines.append(f"| {label} | {m['n']} | {m['meas_rms'] * sc:.0f} {u} | {m['rmse'] * sc:.0f} {u} | "
                         f"{m['bias'] * sc:+.0f} {u} | {m['r']:.2f} |")
        r0 = next(iter(res.values()))
        lines.append(f"\nPredicted ridge deflections 1..11 (mm, kappa={next(iter(res)).split('=')[1]}): "
                     + ", ".join(f"{v:.3f}" for v in r0["deflection"]))

    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "report.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["plate", "rigid"])
    ap.add_argument("--units", default="inplane,vertical")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = fetch()
    print(tf.device_info(), flush=True)
    if a.variant:
        for u in a.units.split(","):
            if not os.path.exists(os.path.join(OUT, f"{a.variant}_{u}.npz")):
                run_unit(a.variant, u, files)
    if a.report or not a.variant:
        report(files)
