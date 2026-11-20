"""Keyhole stage 1: keyhole depth, fluctuation and stability maps from the scaling laws of Gan et
al., Nat. Commun. 12 (2021) 2379 (tpmsfem.process.defects), validated on their operando X-ray
keyhole depths and applied to IN625 and the CoCrFeMnNi gyroid.

1. Validation (Ti-6Al-4V first). Supplementary Data 1 of Gan et al. (data/gan2021_keyhole_xray.json):
   87 keyhole depths from synchrotron X-ray imaging of bare plates (71 Ti-6Al-4V from Cunningham
   et al., Science 2019; 9 Al6061; 7 SS316), 100-520 W, 0.3-1.2 m/s, r0 44-70 um.
   * route "published eta": Ke with the absorptivity column of the data set (checks Ke and
     e* = 0.4 (Ke - 1.4) as implemented);
   * route "predictive": the absorptivity from Eq. 6 with the flat-melt absorptivity eta_m of
     their Supplementary Table 1, i.e. the chain used for a new alloy (P, v, r0 -> eta -> Ke -> e);
   * leave-one-material-out: e* = a (Ke - b) fitted to Al6061 + SS316 only, tested on Ti-6Al-4V;
   * the IN625 AM-Bench 2018 tracks (conduction pools, D/W = 0.28-0.29) as a regime check.
2. IN625 maps. Locatelli et al. (2026) machine (30 um layers, 50 um hatch, 80 C plate): reported
   keyhole pores at 250-300 W / 0.5 m/s and dense parts at 150-300 W / 1.0-1.5 m/s. The spot
   radius is not reported; it is scanned (20-60 um) for the radius that reproduces the report
   with the Ke 16 pore threshold.
3. CoCrFeMnNi gyroid. The study's settings (200 W, 0.75 m/s, r0 35 um) and the part
   temperatures of the heat-accumulation study (80-500 C).

Properties: liquid values at the liquidus, as Gan et al. use (IN625: the repo table, which ends
at 870 C and is held constant above; CoCrFeMnNi: cantor_material). eta_m is not measured for
either alloy: 0.33, the SS316 value of Gan et al., with 0.28 and 0.38 as bounds.

    python benchmarks/literature/keyhole_maps.py          # seconds
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpmsfem.process import defects as DF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gan2021_keyhole_xray.json")
STORE = os.path.join(HERE, "results", "keyhole_maps.json")
FIG = os.path.join(HERE, "..", "..", "docs", "img", "keyhole_maps.png")
ETA_M = (0.28, 0.33, 0.38)                    # IN625 and CoCrFeMnNi: SS316 value of Gan et al. +- 0.05
LOCATELLI = dict(T0=80.0, layer=30e-6, hatch=50e-6, pores=[(250, 0.5), (300, 0.5)],
                 dense=[(p, v) for p in (150, 200, 250, 300) for v in (1.0, 1.5)])
AMBENCH = dict(r0=85e-6, T0=25.0, cases={"A": (137.9, 0.4, 42.5, 147.9), "B": (179.2, 0.8, 36.0, 123.5),
                                          "C": (179.2, 1.2, 29.6, 106.0)})     # P, v, depth, width [um]
GYROID = dict(P=200.0, v=0.75, r0=35e-6, layer=30e-6, hatch=85e-6, T0=(80.0, 140.0, 300.0, 500.0),
              r0_alt=45e-6)                 # a larger spot, as inferred for the IN625 machine
POWERS = np.linspace(80.0, 400.0, 81)
SPEEDS = np.linspace(0.3, 2.5, 111)


def liquid(name):
    """rho [kg/m^3], c_p [J/(kg K)], k [W/(m K)], T_liquidus [C] at the liquidus."""
    if name == "IN625":
        import in625_material as IM
        th = IM.thermal_properties()
    else:
        import cantor_material as CM
        th = CM.material().thermal
    Tl = float(th.T_liquidus)
    return dict(rho=float(th.rho) * 1e12, cp=float(np.interp(Tl, th.T, th.cp)) * 1e-6, k=float(th.conductivity(Tl)),
                T_l=Tl)


def state(name, P, v, r0, T0, eta_m=0.33):
    p = liquid(name)
    return DF.keyhole_state(P, v, r0, eta_min=eta_m, rho=p["rho"], cp=p["cp"], alpha=p["k"] / (p["rho"] * p["cp"]),
                            T_l=p["T_l"], T0=T0)


def metrics(meas, pred, r0):
    """Errors of the aspect ratio e* and of the depth e [um]."""
    res = pred - meas
    return dict(n=int(meas.size), rmse_um=float(1e6 * np.sqrt(np.mean((res * r0) ** 2))),
                bias_um=float(1e6 * np.mean(res * r0)), mae_rel=float(np.mean(np.abs(res) / meas)),
                rmse_aspect=float(np.sqrt(np.mean(res ** 2))),
                r2_aspect=float(1.0 - np.sum(res ** 2) / np.sum((meas - meas.mean()) ** 2)))


def validate():
    d = json.load(open(DATA))
    rows, props = d["rows"], d["table_S1"]
    out = dict(source=d["citation"], materials={}, rows=[])
    cols = {}
    for r in rows:
        m = r["material"]
        tl0 = r["Tl_minus_T0_K"]
        args = dict(rho=r["rho_kg_m3"], cp=r["cp_J_kgK"], T_l=tl0, T0=0.0)
        ke_pub = DF.keyhole_number(r["P_W"], r["v_m_s"], r["eta"], r["r0_m"], alpha=r["alpha_m2_s"], **args)
        eta_pred = float(DF.keyhole_absorptivity(r["P_W"], r["v_m_s"], r["r0_m"], props[m]["eta_min"], **args))
        ke_pred = DF.keyhole_number(r["P_W"], r["v_m_s"], eta_pred, r["r0_m"], alpha=r["alpha_m2_s"], **args)
        row = dict(material=m, group=r["group"], P=r["P_W"], v=r["v_m_s"], r0_um=1e6 * r["r0_m"], e_um=1e6 * r["e_m"],
                   aspect=r["e_m"] / r["r0_m"], eta_pub=r["eta"], Ke_pub=float(ke_pub), eta_pred=eta_pred,
                   Ke_pred=float(ke_pred), tan_front=r["tan_front_angle"])
        out["rows"].append(row)
        for k, val in row.items():
            cols.setdefault(k, []).append(val)
    c = {k: np.asarray(v) for k, v in cols.items()}
    for m in ("Ti-6Al-4V", "Al6061", "SS316", "all"):
        s = np.ones(c["material"].size, bool) if m == "all" else c["material"] == m
        r0 = 1e-6 * c["r0_um"][s]
        out["materials"][m] = dict(
            published_eta=metrics(c["aspect"][s], DF.keyhole_aspect_ratio(c["Ke_pub"][s]), r0),
            predictive=metrics(c["aspect"][s], DF.keyhole_aspect_ratio(c["Ke_pred"][s]), r0),
            Ke_range=[float(c["Ke_pub"][s].min()), float(c["Ke_pub"][s].max())],
            e_range_um=[float(c["e_um"][s].min()), float(c["e_um"][s].max())])
    # own fits e* = a (Ke - b): all, Ti-6Al-4V only, and Al6061 + SS316 only (tested on Ti-6Al-4V)
    ti = c["material"] == "Ti-6Al-4V"
    fits = {}
    for name, s in (("all", np.ones_like(ti)), ("Ti-6Al-4V", ti), ("Al6061+SS316", ~ti)):
        a, b0 = np.polyfit(c["Ke_pub"][s], c["aspect"][s], 1)
        fits[name] = dict(slope=float(a), Ke0=float(-b0 / a), n=int(s.sum()))
    f = fits["Al6061+SS316"]
    pred = np.maximum(f["slope"] * (c["Ke_pub"][ti] - f["Ke0"]), 0.0)
    fits["Al6061+SS316 -> Ti-6Al-4V"] = metrics(c["aspect"][ti], pred, 1e-6 * c["r0_um"][ti])
    out["fits"] = fits
    # regime check on the data: Gan et al. place the transition regime at e* <= 2
    s = c["Ke_pub"] <= DF.KE_KEYHOLE
    out["transition_rows"] = dict(n=int(s.sum()), aspect_le_2=int((c["aspect"][s] <= 2.0).sum()))
    s = c["Ke_pub"] > DF.KE_KEYHOLE
    out["keyhole_rows"] = dict(n=int(s.sum()), aspect_gt_2=int((c["aspect"][s] > 2.0).sum()))
    # front-wall angle (stage 2 reference): tan(theta) ~ e*
    s = np.array([t is not None for t in c["tan_front"]])
    a = np.polyfit(c["aspect"][s].astype(float), c["tan_front"][s].astype(float), 1)
    out["front_angle_fit"] = dict(slope=float(a[0]), intercept=float(a[1]), n=int(s.sum()),
                                  r=float(np.corrcoef(c["aspect"][s].astype(float), c["tan_front"][s].astype(float))[0, 1]))
    # AM-Bench 2018 IN625 tracks on bare plate
    amb = {}
    for k, (P, v, D, W) in AMBENCH["cases"].items():
        st = {e: state("IN625", P, v, AMBENCH["r0"], AMBENCH["T0"], e) for e in ETA_M}
        amb[k] = dict(P=P, v=v, depth_um=D, width_um=W, D_over_W=D / W,
                      Ke={str(e): float(st[e]["Ke"]) for e in ETA_M}, eta={str(e): float(st[e]["eta"]) for e in ETA_M},
                      regime={str(e): str(st[e]["regime"]) for e in ETA_M},
                      keyhole_depth_um={str(e): float(1e6 * st[e]["depth"]) for e in ETA_M})
    out["ambench2018_in625"] = amb
    return out


def locatelli_scan():
    """Agreement with the reported pores / dense points against the unknown spot radius."""
    radii = np.arange(20.0, 60.5, 1.0)
    out = {}
    for e in ETA_M:
        good, ke_p, ke_d = [], [], []
        for r in radii:
            kp = [float(state("IN625", P, v, r * 1e-6, LOCATELLI["T0"], e)["Ke"]) for P, v in LOCATELLI["pores"]]
            kd = [float(state("IN625", P, v, r * 1e-6, LOCATELLI["T0"], e)["Ke"]) for P, v in LOCATELLI["dense"]]
            good.append(sum(k > DF.KE_STABLE for k in kp) + sum(k <= DF.KE_STABLE for k in kd))
            ke_p.append(kp)
            ke_d.append(kd)
        good = np.asarray(good)
        n = len(LOCATELLI["pores"]) + len(LOCATELLI["dense"])
        best = radii[good == good.max()]
        rb = 0.5 * (best.min() + best.max()) * 1e-6
        kl_p = [float(state("IN625", P, v, rb, LOCATELLI["T0"], e)["KeL"]) for P, v in LOCATELLI["pores"]]
        kl_d = [float(state("IN625", P, v, rb, LOCATELLI["T0"], e)["KeL"]) for P, v in LOCATELLI["dense"]]
        out[str(e)] = dict(radii_um=radii.tolist(), matched=good.tolist(), n=n, best_matched=int(good.max()),
                           best_radii_um=[float(best.min()), float(best.max())], Ke_pores=ke_p, Ke_dense=ke_d,
                           # Eq. 8 midpoint as the alternative pore criterion, at the middle of the best radii
                           KeL_mid_radius_um=1e6 * rb, KeL_pores=kl_p, KeL_dense=kl_d,
                           KeL_matched=sum(k > 2.32 for k in kl_p) + sum(k <= 2.32 for k in kl_d))
    return out


def gyroid():
    g = GYROID
    rows = []
    for T0 in g["T0"]:
        for e, r0 in [(e, g["r0"]) for e in ETA_M] + [(0.33, g["r0_alt"])]:
            st = state("Cantor", g["P"], g["v"], r0, T0, e)
            p = liquid("Cantor")
            ned = DF.normalized_energy_density(g["P"], g["v"], g["hatch"], g["layer"], e, p["rho"], p["cp"], p["T_l"], T0)
            rows.append(dict(T0=T0, eta_m=e, r0_um=1e6 * r0, eta=float(st["eta"]), Ke=float(st["Ke"]), depth_um=float(1e6 * st["depth"]),
                             depth_min_um=float(1e6 * st["depth_min"]), depth_max_um=float(1e6 * st["depth_max"]),
                             fluctuation=float(st["fluctuation"]), KeL=float(st["KeL"]), NED=float(ned),
                             regime=str(st["regime"]), stability=str(st["stability"])))
    return rows


def run():
    out = dict(validation=validate(), locatelli=locatelli_scan(), gyroid=gyroid(),
               liquid=dict(IN625=liquid("IN625"), CoCrFeMnNi=liquid("Cantor")), eta_m=ETA_M)
    json.dump(out, open(STORE, "w"), indent=1)
    v = out["validation"]
    for m, d in v["materials"].items():
        a, b = d["published_eta"], d["predictive"]
        print(f"{m:10s} n {a['n']:2d}: published eta RMSE {a['rmse_um']:5.1f} um, bias {a['bias_um']:+5.1f}, R2 "
              f"{a['r2_aspect']:.3f} | Eq. 6 eta RMSE {b['rmse_um']:5.1f} um, bias {b['bias_um']:+5.1f}, R2 {b['r2_aspect']:.3f}")
    for k, f in v["fits"].items():
        print(" fit", k, {kk: round(vv, 3) for kk, vv in f.items()})
    print(" transition rows", v["transition_rows"], "keyhole rows", v["keyhole_rows"], "front", v["front_angle_fit"])
    for k, a in v["ambench2018_in625"].items():
        print(f" AM-Bench {k}: Ke {a['Ke']}, regime {a['regime']}, D/W {a['D_over_W']:.2f}")
    for e, d in out["locatelli"].items():
        print(f" Locatelli eta_m {e}: best {d['best_matched']}/{d['n']} at r0 {d['best_radii_um']} um; "
              f"Eq. 8 midpoint at r0 {d['KeL_mid_radius_um']:.0f} um: {d['KeL_matched']}/{d['n']}")
    for r in out["gyroid"]:
        if r["eta_m"] == 0.33:
            print(f" gyroid r0 {r['r0_um']:.0f} T0 {r['T0']:.0f}: eta {r['eta']:.2f} Ke {r['Ke']:.1f} e {r['depth_um']:.0f} "
                  f"({r['depth_min_um']:.0f}-{r['depth_max_um']:.0f}) um, {r['regime']}, {r['stability']}, KeL {r['KeL']:.2f}")
    print("wrote", STORE)
    return out


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    INK, MUTED, GRID, BLUE, ORANGE, GREEN, RED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#2f9e6e", "#c8302c"
    s = json.load(open(STORE))
    v = s["validation"]

    def style(ax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    fig, axs = plt.subplots(2, 3, figsize=(15, 9.4))
    fig.subplots_adjust(left=0.05, right=0.955, top=0.885, bottom=0.085, wspace=0.3, hspace=0.36)
    rows = v["rows"]
    # (a) the collapse
    ax = axs[0, 0]
    ke = np.linspace(0, 42, 200)
    e = DF.keyhole_aspect_ratio(ke)
    de = DF.keyhole_fluctuation(ke)
    ax.fill_between(ke, np.maximum(e - de / 2, 0), e + de / 2, color="#d6d5d0", alpha=0.7, lw=0,
                    label="fluctuation band, e* ± Δe*/2 (Eq. 5)")
    ax.plot(ke, e, color=INK, lw=1.5, label="e* = 0.4 (Ke − 1.4) (Eq. 2)")
    f = v["fits"]["Al6061+SS316"]
    ax.plot(ke, np.maximum(f["slope"] * (ke - f["Ke0"]), 0), color=MUTED, lw=1.0, ls="--",
            label=f"fit to Al6061 + SS316 only: {f['slope']:.2f} (Ke − {f['Ke0']:.1f})")
    for grp, col, mk, lab in (("Ti64, spot size 95", BLUE, "o", "Ti-6Al-4V, r0 47.5 µm"),
                              ("Ti6, spot size 140", BLUE, "^", "Ti-6Al-4V, r0 70 µm"),
                              ("Al6061", GREEN, "s", "Al6061"), ("SS316", ORANGE, "D", "SS316")):
        sel = [r for r in rows if r["group"].startswith(grp)]
        ax.scatter([r["Ke_pub"] for r in sel], [r["aspect"] for r in sel], s=18, marker=mk, color=col, alpha=0.85,
                   edgecolor="white", lw=0.4, label=lab, zorder=3)
    for x in (DF.KE_TRANSITION, DF.KE_KEYHOLE, DF.KE_STABLE, DF.KE_CHAOTIC):
        ax.axvline(x, color=MUTED, lw=0.6, ls=":")
    for x, t in ((6.4, "keyhole,\nstable"), (16.4, "inter-\nmediate"), (34.0, "chaotic")):
        ax.text(x, 13.3, t, fontsize=6.8, color=MUTED, va="top")
    ax.text(3.7, 13.3, "transition", fontsize=6.8, color=MUTED, va="top", ha="center", rotation=90)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 13.5)
    ax.set_xlabel("keyhole number Ke (published absorptivity)", fontsize=8.5)
    ax.set_ylabel("keyhole aspect ratio e* = e / r0", fontsize=8.5)
    ax.set_title("(a) X-ray keyhole depths collapse on Ke (Gan et al. 2021)", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    style(ax)
    # (b) Ti-6Al-4V predicted vs measured depth
    ax = axs[0, 1]
    ti = [r for r in rows if r["material"] == "Ti-6Al-4V"]
    meas = np.array([r["e_um"] for r in ti])
    r0 = np.array([r["r0_um"] for r in ti])
    p_pub = r0 * DF.keyhole_aspect_ratio(np.array([r["Ke_pub"] for r in ti]))
    p_pred = r0 * DF.keyhole_aspect_ratio(np.array([r["Ke_pred"] for r in ti]))
    m = v["materials"]["Ti-6Al-4V"]
    ax.plot([0, 520], [0, 520], color=INK, lw=0.9)
    ax.fill_between([0, 520], [0, 0.8 * 520], [0, 1.2 * 520], color="#ecebe7", lw=0, label="± 20 %")
    ax.scatter(meas, p_pub, s=16, color=BLUE, edgecolor="white", lw=0.4, zorder=3,
               label=f"published absorptivity: RMSE {m['published_eta']['rmse_um']:.0f} µm, "
                     f"R² {m['published_eta']['r2_aspect']:.2f}")
    ax.scatter(meas, p_pred, s=16, marker="x", color=ORANGE, lw=0.9, zorder=3,
               label=f"Eq. 6 with η_m = 0.26: RMSE {m['predictive']['rmse_um']:.0f} µm, "
                     f"bias {m['predictive']['bias_um']:+.0f} µm")
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 520)
    ax.set_xlabel("measured keyhole depth [µm] (X-ray)", fontsize=8.5)
    ax.set_ylabel("predicted keyhole depth [µm]", fontsize=8.5)
    ax.set_title(f"(b) Ti-6Al-4V: {m['predictive']['n']} tracks, 100-520 W, 0.4-1.2 m/s", fontsize=9.5, loc="left")
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    style(ax)
    # (c) AM-Bench IN625: regime check + Locatelli radius scan
    ax = axs[0, 2]
    for e, col in zip(s["eta_m"], (BLUE, INK, ORANGE)):
        d = s["locatelli"][str(e)]
        ax.plot(d["radii_um"], d["matched"], color=col, lw=1.5 if e == 0.33 else 1.0,
                label=f"η_m = {e:.2f}: {d['best_matched']}/{d['n']} at r0 {d['best_radii_um'][0]:.0f}-"
                      f"{d['best_radii_um'][1]:.0f} µm")
    n = s["locatelli"]["0.33"]["n"]
    ax.axhline(n, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(0, n + 0.8)
    ax.set_xlabel("assumed spot radius r0 [µm] (1/e²; not reported)", fontsize=8.5)
    ax.set_ylabel(f"reported outcomes reproduced (of {n})", fontsize=8.5)
    ax.set_title("(c) IN625, Locatelli et al.: pores if Ke > 16", fontsize=9.5, loc="left")
    amb = v["ambench2018_in625"]
    txt = "AM-Bench 2018 IN625 tracks (r0 85 µm, conduction pools):\n" + "\n".join(
        f"  {k}: {a['P']:.0f} W, {a['v']:.1f} m/s, D/W {a['D_over_W']:.2f} -> Ke {a['Ke']['0.33']:.1f}, {a['regime']['0.33']}"
        for k, a in amb.items())
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=6.8, color=MUTED, va="bottom")
    ax.legend(fontsize=6.8, frameon=False, loc="center right")
    style(ax)
    # (d, e) maps
    cmap = ListedColormap(["#f3f2ee", "#d9e6f5", "#9cc2ea", "#f6c39f", "#e48a6a"])

    def pmap(ax, name, r0, T0, title, pts=()):
        PP, VV = np.meshgrid(POWERS, SPEEDS, indexing="ij")
        st = state(name, PP, VV, r0, T0)
        cls = np.select([st["Ke"] <= DF.KE_TRANSITION, st["Ke"] <= DF.KE_KEYHOLE, st["Ke"] <= DF.KE_STABLE,
                         st["Ke"] <= DF.KE_CHAOTIC], [0, 1, 2, 3], 4)
        ax.pcolormesh(VV, PP, cls, cmap=cmap, vmin=-0.5, vmax=4.5, shading="auto")
        cs = ax.contour(VV, PP, 1e6 * st["depth"], [50, 100, 200, 300], colors=[INK], linewidths=0.7)
        ax.clabel(cs, fmt="%d µm", fontsize=6.5)
        ax.contour(VV, PP, st["KeL"], [2.32], colors=[RED], linewidths=1.2, linestyles="--")
        for P, v_, mk, col, lab in pts:
            ax.scatter([v_], [P], marker=mk, s=110 if mk == "*" else 40, color=col, edgecolor="white", lw=0.6, zorder=4, label=lab)
        ax.set_xlabel("scan speed [m/s]", fontsize=8.5)
        ax.set_ylabel("laser power [W]", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlim(SPEEDS[0], SPEEDS[-1])
        ax.tick_params(colors=MUTED, labelsize=7.5)

    r_in = 0.5 * sum(s["locatelli"]["0.33"]["best_radii_um"]) * 1e-6
    pts = [(P, v_, "s", RED, "keyhole pores (reported)" if i == 0 else None) for i, (P, v_) in enumerate(LOCATELLI["pores"])]
    pts += [(P, v_, "o", GREEN, "dense (reported)" if i == 0 else None) for i, (P, v_) in enumerate(LOCATELLI["dense"])]
    pmap(axs[1, 0], "IN625", r_in, LOCATELLI["T0"], f"(d) IN625, r0 {1e6 * r_in:.0f} µm, plate 80 °C", pts)
    axs[1, 0].legend(fontsize=6.8, frameon=True, loc="upper right")
    g = GYROID
    pmap(axs[1, 1], "Cantor", g["r0"], 80.0, "(e) CoCrFeMnNi, r0 35 µm, part at 80 °C",
         [(g["P"], g["v"], "*", INK, "gyroid study, 200 W, 0.75 m/s")])
    axs[1, 1].legend(fontsize=6.8, frameon=True, loc="lower right")
    # legend for classes
    from matplotlib.patches import Patch
    hs = [Patch(color=cmap(i), label=t) for i, t in enumerate(("conduction, Ke ≤ 1.4", "transition, ≤ 6",
                                                                "keyhole, stable, ≤ 16", "intermediate, ≤ 30",
                                                                "chaotic, > 30"))]
    hs.append(plt.Line2D([], [], color=RED, ls="--", lw=1.2, label="Ke_m L_d* = 2.32 (Eq. 8 keyhole-pore midpoint)"))
    hs.append(plt.Line2D([], [], color=INK, lw=0.7, label="keyhole depth e"))
    axs[1, 1].figure.legend(handles=hs, fontsize=7, frameon=False, ncol=7, loc="lower center", bbox_to_anchor=(0.36, 0.0))
    # (f) gyroid vs part temperature
    ax = axs[1, 2]
    rows_g = s["gyroid"]
    T0s = sorted({r["T0"] for r in rows_g})
    r_s, r_a = 1e6 * GYROID["r0"], 1e6 * GYROID["r0_alt"]
    get = lambda e, k, r0=r_s: np.array([[r[k] for r in rows_g if r["T0"] == T and r["eta_m"] == e
                                          and abs(r["r0_um"] - r0) < 0.1][0] for T in T0s])
    ax.fill_between(T0s, get(0.28, "Ke"), get(0.38, "Ke"), color="#d9e6f5", lw=0, label="η_m 0.28-0.38")
    ax.plot(T0s, get(0.33, "Ke"), "o-", color=BLUE, lw=1.6, ms=4, label=f"Ke, η_m 0.33, r0 {r_s:.0f} µm (study)")
    ax.plot(T0s, get(0.33, "Ke", r_a), "o:", color=MUTED, lw=1.2, ms=3, label=f"Ke, η_m 0.33, r0 {r_a:.0f} µm")
    for y, t in ((DF.KE_KEYHOLE, "keyhole above 6"), (DF.KE_STABLE, "pores likely above 16"), (DF.KE_CHAOTIC, "chaotic above 30")):
        ax.axhline(y, color=MUTED, lw=0.7, ls=":")
        ax.text(T0s[-1], y + 0.4, t, fontsize=6.8, color=MUTED, ha="right")
    ax.set_xlabel("part temperature under the layer [°C] (heat accumulation)", fontsize=8.5)
    ax.set_ylabel("keyhole number Ke", fontsize=8.5)
    ax.set_ylim(0, 36)
    ax2 = ax.twinx()
    ax2.errorbar(T0s, get(0.33, "depth_um"), yerr=[get(0.33, "depth_um") - get(0.33, "depth_min_um"),
                                                   get(0.33, "depth_max_um") - get(0.33, "depth_um")],
                 color=ORANGE, fmt="s--", ms=4, lw=1.1, capsize=3, label="keyhole depth ± Δe/2, η_m 0.33")
    ax2.set_ylabel("keyhole depth [µm]", fontsize=8.5, color=ORANGE)
    ax2.tick_params(colors=ORANGE, labelsize=7.5)
    ax2.set_ylim(0, 450)
    ax.set_title("(f) gyroid point (200 W, 0.75 m/s) with build height", fontsize=9.5, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.83))
    style(ax)
    ax2.spines["top"].set_visible(False)
    fig.suptitle("Keyhole stage 1: depth, fluctuation and stability from the keyhole number (Gan et al., Nat. Commun. 12 "
                 "(2021) 2379)\nValidated on 87 operando X-ray keyhole depths (71 Ti-6Al-4V), then applied to IN625 and "
                 "the CoCrFeMnNi gyroid (liquid properties at the liquidus, process absorptivity of Eq. 6)",
                 fontsize=10.5, x=0.02, ha="left")
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    figure()
