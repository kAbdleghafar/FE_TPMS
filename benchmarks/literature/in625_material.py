"""IN625 (LPBF, AM-Bench) material for the thermo-mechanical AM-Bench 2018 model.

Data (benchmarks/literature/data/in625_constitutive.json):

* hardening of the bridge material: NIST MaCTO compression on specimens cut
  from the AMB2018-01 bridge build, transverse (in-plane) direction, at 25,
  250 and 500 C (doi:10.18434/mds2-2681); plastic strain
  eps_p = eps_true - sigma_true / E(T);
* above 500 C (no AM-Bench data): the 500 C curve scaled by the ratio
  sigma_y(T) / sigma_y(500 C) of the baseline (Denlinger-Michaleris) table,
  held constant above 870 C (no extrapolation towards the liquidus);
* E(T), alpha(T), k(T), cp(T), density, latent heat, solidus, liquidus: the
  baseline table (data guide sec. 2), alpha taken as a mean coefficient from
  20 C; thermal strain frozen above the solidus;
* build plate (wrought IN625): baseline yield(T), perfectly plastic.

Room-temperature tension (AMB2022 E8, doi:10.18434/mds2-2760) is reduced
here as a check of the compression data (tension-compression asymmetry):

    python benchmarks/literature/in625_material.py     # downloads E8 CSVs, writes the figure/table
"""
import json
import os
import urllib.request

import numpy as np

import tpmsfem as tf
from tpmsfem.fem.materials import FlowTable, TemperatureTable, ThermalProperties

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "in625_constitutive.json")))
OUT = "results/literature/in625_flow"
EPS_GRID = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2])
E8_URL = "https://data.nist.gov/od/ds/mds2-2760/SpecimenT{}_stress_strain.csv"
NU = DATA["baseline_table"]["poisson_used"]


def baseline():
    b = DATA["baseline_table"]
    return {k: np.asarray(v, float) if isinstance(v, list) else v for k, v in b.items()}


def E_of_T(T):
    b = baseline()
    return np.interp(T, b["T"], b["E_GPa"] * 1e3)


def macto_curves(direction="TD"):
    """{T: (eps_p, sigma_true)} from the MaCTO true flow stresses."""
    m = DATA["macto"]
    eps = np.asarray(m["true_strain"])
    out = {}
    for T, c in m["curves"][direction].items():
        T = float(T)
        s = np.array([np.nan if v is None else v for v in c["stress"]], float)
        ok = np.isfinite(s)
        ep = eps[ok] - s[ok] / E_of_T(T)
        out[T] = (np.r_[0.0, ep], np.r_[c["yield"], s[ok]])     # 0.2 % offset yield at eps_p = 0
    return out


def part_flow(direction="TD", softening=None):
    """Temperature-dependent hardening table of the as-built bridge material.
    ``softening``: optional {T: factor} extra points relative to the 870 C curve
    (e.g. {1290: 0.05}) -- NOT in the data guide, sensitivity studies only."""
    curves = macto_curves(direction)
    b = baseline()
    base = FlowTable.from_curves(curves, eps_p=EPS_GRID)
    s500 = base.at(500.0)
    ref = np.interp(500.0, b["T"], b["yield_MPa"])
    T, S = list(base.T), list(base.sigma)
    for Tb in (650.0, 760.0, 870.0):
        T.append(Tb)
        S.append(s500 * np.interp(Tb, b["T"], b["yield_MPa"]) / ref)
    s870 = S[-1]
    for Tb, f in sorted((softening or {}).items()):
        if Tb <= 870:
            raise ValueError("softening points must lie above the last table temperature (870 C)")
        T.append(float(Tb))
        S.append(s870 * f)
    return FlowTable(EPS_GRID, np.array(S), np.array(T), 0.0, f"MaCTO-{direction}")


def temperature_table(nu_table=False):
    b = baseline()
    return TemperatureTable(T=b["T"], E=b["E_GPa"] * 1e3, alpha=b["alpha_1e-6_K"] * 1e-6, T_ref=20.0,
                            T_cap=b["solidus_C"])


def thermal_properties():
    b = baseline()
    return ThermalProperties.from_si(b["T"], b["k_W_mK"], b["cp_J_kgK"], b["density_kg_m3"], b["latent_J_kg"],
                                     b["solidus_C"], b["liquidus_C"])


def part_material(direction="TD", anneal=None, softening=None, nu=NU):
    return tf.Material("IN625_LPBF_AMB2018", E=E_of_T(20.0), nu=nu, flow=part_flow(direction, softening),
                       temperature=temperature_table(), thermal=thermal_properties(), anneal_temperature=anneal,
                       density=8.44e-9)


def plate_material(nu=NU):
    b = baseline()
    flow = FlowTable([0.0], b["yield_MPa"][:, None], b["T"], 0.0, "baseline yield, perfectly plastic")
    return tf.Material("IN625_plate", E=E_of_T(20.0), nu=nu, flow=flow, temperature=temperature_table(),
                       thermal=thermal_properties(), density=8.44e-9)


# ---------------------------------------------------------------- E8 check
def reduce_e8():
    raw = os.path.join(OUT, "e8")
    os.makedirs(raw, exist_ok=True)
    rows = []
    for n in (3, 4, 6, 5, 7):
        path = os.path.join(raw, f"SpecimenT{n}_stress_strain.csv")
        if not os.path.exists(path):
            req = urllib.request.Request(E8_URL.format(n), headers={"User-Agent": "tpmsfem-benchmark/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as fh:
                fh.write(r.read())
        d = np.genfromtxt(path, delimiter=",", skip_header=2)
        t, e, s = d[:, 0], d[:, 1], d[:, 2]
        iu = int(np.argmax(s))
        sel = (s > 100) & (s < 400) & (np.arange(len(s)) < iu)
        E = np.polyfit(e[sel], s[sel], 1)[0]
        e, s = e[:iu + 1], s[:iu + 1]                       # uniform deformation only (up to UTS)
        st, et = s * (1 + e), np.log1p(e)
        ep = et - st / E
        y = s[np.flatnonzero(e - s / E > 0.002)[0]]
        rate = float(np.median(np.diff(e) / np.diff(t[:iu + 1])))
        rows.append(dict(specimen=f"T{n}", rate=rate, E=E, yield02=float(y), uts=float(s.max()),
                         eps_p=ep.tolist(), sigma_true=st.tolist()))
    return rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    e8 = reduce_e8()
    fl = part_flow()
    lines = ["| source | condition | 0.2 % yield [MPa] | true stress at eps_p = 0.02 / 0.05 / 0.10 [MPa] |",
             "|---|---|---:|---|"]
    for T in fl.T:
        s = fl.at(T)
        tag = "MaCTO TD (AMB2018-01 build), compression" if T <= 500 else "500 C curve x baseline ratio (assumption)"
        lines.append(f"| {tag} | {T:g} C | {s[0]:.0f} | " + " / ".join(f"{np.interp(e, fl.eps_p, s):.0f}"
                                                                    for e in (0.02, 0.05, 0.1)) + " |")
    for T, c in macto_curves("BD").items():
        lines.append(f"| MaCTO BD, compression | {T:g} C | {c[1][0]:.0f} | " + " / ".join(
            f"{np.interp(e, *c):.0f}" for e in (0.02, 0.05, 0.1)) + " |")
    for r in e8:
        lines.append(f"| E8 tension (AMB2022 build), {r['specimen']}, {r['rate']:.0e}/s | 20 C | {r['yield02']:.0f} | "
                     + " / ".join(f"{np.interp(e, r['eps_p'], r['sigma_true']):.0f}" for e in (0.02, 0.05, 0.1)) + " |")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "flow_table.md"), "w").write(txt + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    cols = {25.0: "#2a78d6", 250.0: "#eb6834", 500.0: "#1f9e74", 650.0: "#8a5cc4", 760.0: "#c4475b", 870.0: "#52514e"}
    for T in fl.T:
        ax.plot(100 * fl.eps_p, fl.at(T), "-o", ms=3, lw=1.6, color=cols.get(T, "k"),
                label=f"model {T:g} C" + (" (MaCTO TD)" if T <= 500 else " (scaled)"))
    for i, r in enumerate(e8):
        ax.plot(100 * np.array(r["eps_p"]), r["sigma_true"], color="#0b0b0b", lw=0.8, alpha=0.6,
                label="E8 tension 20 C (AMB2022 build)" if i == 0 else None)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1400)
    ax.set_xlabel("true plastic strain [%]")
    ax.set_ylabel("true stress [MPa]")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    ax.grid(color="#e4e3df", lw=0.6)
    fig.suptitle("IN625 hardening tables used for the AM-Bench 2018 thermo-mechanical model", fontsize=10,
                 x=0.02, ha="left")
    os.makedirs("docs/img", exist_ok=True)
    fig.savefig("docs/img/in625_flow.png", dpi=200)


if __name__ == "__main__":
    main()
