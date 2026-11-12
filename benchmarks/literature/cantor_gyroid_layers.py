"""Cell structure and Mn+Ni segregation in the first, middle and final layers of the LPBF
CoCrFeMnNi gyroid column (phase field at each layer's fusion-boundary conditions).

- First layer: on the plate at 80 C. Conditions of the single track on a solid plate
  (cantor_gyroid_solidification.py, validation run). There is no powder below it.
- Middle layer (4.4 mm) and final layer (8.8 mm): the part-scale build gives the temperature
  the layer meets (cantor_gyroid_multilayer.py). G and R of the thin-wall window are
  interpolated in that temperature between the melt-pool runs at 80 C and 140 C, for wall
  regions over solid and over powder (down-facing).

    python benchmarks/literature/cantor_gyroid_layers.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cantor_gyroid_solidification as CG  # noqa: E402
from common import ResultStore  # noqa: E402

STORE = "benchmarks/literature/results/cantor_gyroid_layers.json"
FIELDS = "benchmarks/literature/results/cantor_gyroid_layers_fields.npz"
Z_MID, Z_END = 4.4, 8.8


def conditions():
    base = json.load(open(CG.STORE))
    pre = json.load(open(CG.STORE.replace(".json", "_preheat.json")))
    ml = json.load(open("benchmarks/literature/results/cantor_gyroid_multilayer.json"))
    recs = (ml.get("build|layer_time=10") or ml["build|layer_time=10|partial"])["records"]
    z = np.array([r["z_top"] for r in recs])
    T = np.array([r["T_top_mean"] for r in recs])
    v = base["validation_meltpool"]
    out = {"first|solid": dict(z=0.03, T0=CG.T_PLATE, G=v["G"]["p50"], R=v["R"]["p50"])}
    w80, w140 = base["window|thin"], pre["window|thin|z9.3"]
    for name, zl in (("middle", Z_MID), ("final", Z_END)):
        T0 = float(np.interp(zl, z, T))
        f = (T0 - CG.T_PLATE) / (w140["T0"] - CG.T_PLATE)
        for c, lab in (("supported", "solid"), ("down-facing", "powder")):
            out[f"{name}|{lab}"] = dict(z=zl, T0=T0, G=(1 - f) * w80[c]["G"]["p50"] + f * w140[c]["G"]["p50"],
                                        R=(1 - f) * w80[c]["R"]["p50"] + f * w140[c]["R"]["p50"])
    return out


def main():
    store = ResultStore(STORE)
    for key, c in conditions().items():
        if store.get(key) is None:
            out, r = CG.run_pf(c["G"], c["R"], 60.0)
            out.update(z=c["z"], T0=c["T0"])
            fields = dict(np.load(FIELDS)) if os.path.exists(FIELDS) else {}
            top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
            lo = max(top - int(1.4e-6 / r.dx_m), 0)
            fields[key + "_c"] = r.c[lo:top + 10].astype(np.float16)
            fields[key + "_phi"] = r.phi[lo:top + 10].astype(np.float16)
            fields[key + "_dx"] = np.array(r.dx_m)
            np.savez_compressed(FIELDS, **fields)
            store.put(key, out)
        o = store.get(key)
        print(f"{key:14s} z = {o['z']:.2f} mm, T0 = {o['T0']:.0f} C, G = {o['G']:.2e} K/m, R = {o['R']:.3f} m/s: "
              f"spacing {o['spacing_um']:.2f} um, cores {o['cs_rel']:.3f}, max between cells {o['intercell_max_rel']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
