"""AM-Bench 2018 IN625 bridge: sequentially coupled thermo-mechanical build.

Same geometry, plate strip, EDM leg cut and measurements as
``nist_ambench2018.py`` (elastic inherent strain), but the residual stress is
*predicted from the process*, with no calibrated parameter:

* heat: layer-lumped deposition of 0.5 mm super-layers (25 physical 20 um
  layers) at the liquidus, cooled by conduction into the part and the plate for
  the super-layer print time (25 x 52 s, Heigel et al. 2020), latent heat,
  radiation + convection from the upward-facing part surfaces, plate bottom at
  T_plate;
* mechanics: temperature-dependent elasto-plasticity -- MaCTO hardening of the
  AMB2018-01 build (TD, 25/250/500 C) and the baseline E(T), alpha(T) table
  (in625_material.py); birth stress-free at the deposition temperature,
  thermal strain frozen above the solidus;
* cool-down to 20 C, plate strip cut free (3-2-1 support), EDM cut of the 12
  legs -> ridge deflection, compared with the measured 1.276 mm.

    python benchmarks/literature/nist_ambench2018_tm.py --tag base            # ~1-2 h on 4 CPU cores
    python benchmarks/literature/nist_ambench2018_tm.py --tag base --layers 2 # timing test
    python benchmarks/literature/nist_ambench2018_tm.py --report
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import in625_material as IM  # noqa: E402
import nist_ambench2018 as AB  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.residual.thermomech import ThermoMechanicalAM  # noqa: E402
from tpmsfem.thermal import HeatModel  # noqa: E402

OUT = os.path.join(AB.OUT, "tm")
PHYS_LAYER, LAYER_TIME = 0.02, 52.0          # mm, s (Heigel et al. 2020)


def run(a, files):
    mesh, h = AB.build_mesh("plate", files)
    mats = {1: IM.part_material(a.direction, anneal=a.anneal, nu=a.nu), 2: IM.plate_material(nu=a.nu)}
    md = tf.SolidModel(mesh, mats, T_initial=a.T_plate)
    hm = HeatModel(mesh, mats, T_initial=a.T_plate)
    # heat leaves through the upward-facing part surfaces (top of each layer, arches);
    # powder-covered sides and the plate (cut from a 100 mm plate) are adiabatic
    hm.set_surface_losses(h=a.h_conv * 1e-3, emissivity=a.emissivity, T_amb=a.T_plate,
                          face_filter=lambda c, n: (n[:, 2] > 0.5) & (c[:, 2] > 1e-6))
    print(f"[tm/{a.tag}] {mesh.summary()}\n{md.summary()}", flush=True)
    sub = np.flatnonzero(mesh.cell_data["material"] == 2)
    tm = ThermoMechanicalAM(md, hm, layer_height=h, T_deposit=a.T_deposit,
                            layer_time=h / PHYS_LAYER * LAYER_TIME, n_steps=a.n_steps, first_step=a.first_step,
                            mech_every=a.mech_every, T_plate=a.T_plate, substrate=sub, solver=a.solver)
    t0 = time.perf_counter()
    tm.build(layers=a.layers)
    if a.layers is not None:
        print(f"[tm/{a.tag}] {a.layers} layers in {time.perf_counter() - t0:.0f}s", flush=True)
        return
    tm.cool_down(T_room=20.0)
    sig_plate = md.stress().mean(axis=1)                 # on the bolted plate, room temperature
    tm.remove_from_plate()
    as_built = md.stress().mean(axis=1)
    alpha = md.plastic_strain()[1].mean(axis=1)
    u_built = md.displacement().copy()
    c = mesh.centroids()
    kerf = np.flatnonzero((mesh.cell_data["material"] == 1) & (c[:, 2] < h) & (c[:, 0] < AB.LEG_CUT_X))
    md.deactivate(kerf)
    md.solve(solver=a.solver, name="leg cut")
    du = md.displacement() - u_built
    p = mesh.points
    defl = []
    for xr in AB.RIDGES_X:
        ridge = np.flatnonzero((np.abs(p[:, 2] - 12.5) < 1e-9) & (np.abs(p[:, 0] - xr) <= 0.5 + 1e-9)
                               & (p[:, 1] >= -1e-9) & (p[:, 1] <= 5 + 1e-9))
        side = np.flatnonzero((np.abs(p[:, 2]) < 1e-9) & (np.abs(p[:, 0] - xr) <= 1.0)
                              & ((p[:, 1] < -0.5) | (p[:, 1] > 5.5)))
        defl.append(float(du[ridge, 2].mean() - du[side, 2].mean()))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{a.tag}.npz"), centroids=c, material=mesh.cell_data["material"],
                        stress=as_built, stress_on_plate=sig_plate, alpha=alpha, deflection=np.array(defl), h=h,
                        seconds=time.perf_counter() - t0, args=json.dumps(vars(a)),
                        records=json.dumps([r.__dict__ for r in tm.records]))
    print(f"[tm/{a.tag}] ridge deflections: {np.round(defl, 3).tolist()} (measured ridge 1: "
          f"{AB.DEFLECTION_RIDGE1})  {time.perf_counter() - t0:.0f}s", flush=True)


def report(files):
    lines = []
    E20 = IM.E_of_T(20.0)
    elastic = json.load(open(os.path.join(AB.OUT, "analysis_plate.json")))["kappa=0"] \
        if os.path.exists(os.path.join(AB.OUT, "analysis_plate.json")) else None
    for path in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        R = np.load(path)
        tag = os.path.basename(path)[:-4]
        args = json.loads(str(R["args"]))
        part = R["material"] == 1
        r = AB.compare_fields(R["centroids"], float(R["h"]), part, R["stress"], R["deflection"], files,
                              E=E20, nu=args["nu"])
        r["plastic_strain_part_mean"] = float(R["alpha"][part].mean())
        json.dump(r, open(os.path.join(OUT, f"analysis_{tag}.json"), "w"), indent=1)
        lines.append(f"\n### thermo-mechanical `{tag}` ({float(R['seconds']) / 60:.0f} min; T_dep = {args['T_deposit']} C, "
                     f"T_plate = {args['T_plate']} C, {args['n_steps']} heat steps / super-layer, "
                     f"anneal = {args['anneal']}, nu = {args['nu']})\n")
        lines.append("| quantity (measurement) | points | measured RMS | RMSE | bias | correlation r |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(f"| ridge-1 deflection after leg cut (measured 1.276 mm) | 1 | | | "
                     f"{r['deflection'][0] - AB.DEFLECTION_RIDGE1:+.3f} mm | predicted {r['deflection'][0]:.3f} mm |")
        lines += AB.metric_lines(r)
        lines.append("\nPredicted ridge deflections 1..11 (mm): " + ", ".join(f"{v:.3f}" for v in r["deflection"]))
        if elastic is not None:
            lines.append("\nFor comparison, elastic inherent strain calibrated on ridge 1: legs neutron eps_xx r = "
                         f"{elastic['neutron_exx_legs']['r']:.2f}, body r = {elastic['neutron_exx_body']['r']:.2f}, "
                         f"contour r = {elastic['contour_sxx']['r']:.2f}.")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "report.md"), "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--first-step", type=float, default=0.05)
    ap.add_argument("--mech-every", type=int, default=1)
    ap.add_argument("--T-deposit", type=float, default=1350.0)
    ap.add_argument("--T-plate", type=float, default=20.0)
    ap.add_argument("--anneal", type=float, default=None)
    ap.add_argument("--direction", default="TD")
    ap.add_argument("--nu", type=float, default=IM.NU)
    ap.add_argument("--h-conv", type=float, default=10.0, help="W/(m^2 K)")
    ap.add_argument("--emissivity", type=float, default=0.4)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    files = AB.fetch()
    print(tf.device_info(), flush=True)
    if not a.report:
        run(a, files)
    else:
        report(files)
