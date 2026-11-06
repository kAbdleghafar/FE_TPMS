"""Tables for docs/literature_benchmarks.md from the saved Ahmed et al. results.

Reads results/literature/ahmed_barsoum_2022_{finite,periodic,secant,rs,sensitivity}.json
and prints the markdown blocks used in the report:

* secant study: for each case, the macroscopic strain at which our secant
  modulus has dropped to the ratio (published / our converged initial modulus)
* residual-stress trend study
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ahmed_barsoum_2022 as B  # noqa: E402
from common import ResultStore  # noqa: E402

R = "results/literature"


def converged_finite():
    st = ResultStore(os.path.join(R, "ahmed_barsoum_2022_finite.json"))
    rows = B.summarize(st, "finite", list(B.TOPOLOGIES), [0.1, 0.2, 0.3])
    return {k: (v["rich"]["f_ext"] if v["rich"]["monotone"] else v["fine"]) for k, v in rows.items()}


def published(topo, rho):
    t3 = B.REF["table3_gibson_ashby_E_over_E0"][topo]
    return float(np.mean([t3["tension"][0] * rho ** t3["tension"][1],
                          t3["compression"][0] * rho ** t3["compression"][1]]))


def matching_strain(sec, target):
    """Strain (in %) where secant/initial falls to ``target`` (linear interpolation)."""
    pts = sorted((float(k.rstrip("%")), v) for k, v in sec.items())
    for (e0, r0), (e1, r1) in zip(pts, pts[1:]):
        if r0 >= target >= r1 and r0 != r1:
            return e0 + (r0 - target) / (r0 - r1) * (e1 - e0)
    return float("nan")


def secant_table():
    sec = json.load(open(os.path.join(R, "ahmed_barsoum_2022_secant.json")))
    fin = converged_finite()
    lines = ["| topology | rho | ours (converged, elastic) | published | published / ours | "
             "our secant / initial at 0.5 % / 1 % | strain where they match |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for key in sorted(sec, key=lambda k: (sec[k]["topology"], sec[k]["rho"])):
        v = sec[key]
        topo, rho = v["topology"], v["rho"]
        ours = fin.get((topo, rho))
        pub = published(topo, rho)
        ratio = pub / ours
        s = v["secant_over_initial"]
        lines.append(f"| {topo} | {rho:.2f} | {ours:.4f} | {pub:.4f} | {ratio:.2f} | "
                     f"{s['0.50%']:.3f} / {s['1.00%']:.3f} | {matching_strain(s, ratio):.2f} % |")
    return "\n".join(lines)


def rs_table():
    rs = json.load(open(os.path.join(R, "ahmed_barsoum_2022_rs.json")))
    lines = ["| topology (rho = 30 %) | E/E0 no RS | E/E0 with RS, tension | compression | "
             "reduction ours (tension / compression) | reduction published (tension / compression) | "
             "RS von Mises p50 / p95 | yielded during build |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for topo, v in rs.items():
        t3 = B.REF["table3_gibson_ashby_E_over_E0"][topo]
        rho = v["rho"]
        red = {m: 1 - (t3[m + "_RS"][0] * rho ** t3[m + "_RS"][1]) / (t3[m][0] * rho ** t3[m][1])
               for m in ("tension", "compression")}
        lines.append(f"| {topo} | {v['E_tension']:.4f} | {v['E_tension_RS']:.4f} | {v['E_compression_RS']:.4f} | "
                     f"{100 * v['reduction_tension']:.1f} % / {100 * v['reduction_compression']:.1f} % | "
                     f"{100 * red['tension']:.1f} % / {100 * red['compression']:.1f} % | "
                     f"{v['rs_vm_p50']:.0f} / {v['rs_vm_p95']:.0f} MPa | {100 * v['rs_plastic_fraction']:.0f} % |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("### secant\n")
    print(secant_table())
    if os.path.exists(os.path.join(R, "ahmed_barsoum_2022_rs.json")):
        print("\n### residual stress\n")
        print(rs_table())
