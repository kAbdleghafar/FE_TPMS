"""Surface roughness vs build angle: LPBF IN718 flat surfaces (Kasperovich et al. 2021).

Kasperovich, Becker, Artzt, Barriobero-Vila, Requena, Haubrich, "The effect of build
direction and geometric optimization in laser powder bed fusion of Inconel 718 structures
with internal channels", Materials & Design 207 (2021) 109858, doi:10.1016/j.matdes.2021.109858
(gold open access). Concept Laser M2, layer 30 um, powder D10/D50/D90 = 22/33/45 um.
Sa by laser scanning microscopy on 2.44 x 2.44 mm^2, flat outer surfaces at 45 deg (upskin),
90 deg (vertical) and 135 deg (downskin) to the platform (Sec. 3.1.1, centre of the platform):

* contour 180 W, 1600 mm/s (E_L = 0.113 J/mm):  Sa = 9.01 +- 0.33, 11.17 +- 1.44, 52.80 +- 3.41 um
* optimum contour, E_L = 0.18 J/mm (Sec. 4):       Sa = 6.8, 7.2, 47.5 um

Model (tpmsfem.process.roughness): staircase (t = 30 um) + melt-pool rounding + partially
fused powder (the paper's PSD). The particle density on vertical walls and its increase on
downskin / decrease on upskin surfaces (rho_side, k_down, k_up) are fitted to the three Sa
values of the E_L = 0.113 J/mm set -- three parameters for three points, so this is a
calibrated description, not a prediction. The bead radius comes from the melt-pool model
(contour track 180 W, 1600 mm/s, absorptivity calibrated on AM-Bench case B).

    python benchmarks/literature/kasperovich_2021_roughness.py
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpmsfem.process.roughness import PowderPSD, SurfaceModel  # noqa: E402

ANGLES = np.array([45.0, 90.0, 135.0])
SA_0113 = np.array([9.01, 11.17, 52.80])
SD_0113 = np.array([0.33, 1.44, 3.41])
SA_018 = np.array([6.8, 7.2, 47.5])
OUT = "benchmarks/literature/results/kasperovich_2021_roughness.json"
PATCH = dict(size=(1.0, 1.0), dx=0.0025)
SEEDS = (0, 1)


def sa(model, theta):
    return float(np.mean([model.roughness(theta, seed=s, **PATCH)["Sa"] for s in SEEDS]))


def fit(base, target):
    def resid(p):
        m = replace(base, rho_side=p[0], k_down=p[1], k_up=p[2])
        return [(sa(m, t) - y) / y for t, y in zip(ANGLES, target)]
    sol = least_squares(resid, x0=[150.0, 10.0, 0.5], bounds=([1.0, 0.0, 0.0], [1500.0, 200.0, 1.0]),
                        diff_step=0.05, max_nfev=60)
    return replace(base, rho_side=sol.x[0], k_down=sol.x[1], k_up=sol.x[2]), sol


def main(bead_radius):
    base = SurfaceModel(layer=0.030, bead_radius=bead_radius, psd=PowderPSD(0.022, 0.033, 0.045), embed=0.4)
    out = {"bead_radius_mm": bead_radius}
    for tag, target in (("EL0.113", SA_0113), ("EL0.18", SA_018)):
        m, sol = fit(base, target)
        curve = {int(t): sa(m, float(t)) for t in range(0, 181, 15)}
        out[tag] = dict(rho_side=m.rho_side, k_down=m.k_down, k_up=m.k_up,
                        fitted={int(t): sa(m, t) for t in ANGLES}, measured=dict(zip(map(int, ANGLES), target.tolist())),
                        curve=curve, staircase_only={int(t): replace(m, rho_side=0.0).roughness(float(t), seed=0, **PATCH)["Sa"]
                                                      for t in range(0, 181, 15)})
        print(tag, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[tag].items()
                               if k in ("rho_side", "k_down", "k_up", "fitted", "measured")}), flush=True)
    out["tpms"] = tpms_application(replace(base, rho_side=out["EL0.113"]["rho_side"], k_down=out["EL0.113"]["k_down"],
                                           k_up=out["EL0.113"]["k_up"]))
    json.dump(out, open(OUT, "w"), indent=1)
    figure(out)


def tpms_application(model):
    """Roughness of an as-built gyroid sheet cell (3 mm, 30 % dense) from the facet
    inclinations: area-weighted Sa and the share of downskin area."""
    from tpmsfem.geometry import TPMS
    from tpmsfem.mesh.surface import implicit_to_surface
    from tpmsfem.process.roughness import inclination

    lat = TPMS("gyroid", cell_size=3.0, kind="sheet", density=0.3)
    v, f = implicit_to_surface(lat, (0, 0, 0), (3, 3, 3), 0.05)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    inner = np.all((tri > 0.05) & (tri < 2.95), axis=(1, 2))          # drop the clipping faces
    th = inclination(n[inner])
    thetas = np.arange(0, 181, 15)
    table = np.array([sa(model, float(t)) for t in thetas])
    sa_face = np.interp(th, thetas, table)
    w = area[inner]
    hist, edges = np.histogram(th, bins=np.arange(0, 181, 15), weights=w)
    np.savez_compressed("results/literature/kasperovich_2021_tpms.npz", verts=v, faces=f[inner], sa=sa_face)
    return dict(area_mm2=float(w.sum()), Sa_area_mean=float((sa_face * w).sum() / w.sum()),
                downskin_area_fraction_theta_gt_120=float(w[th > 120].sum() / w.sum()),
                area_by_theta=dict(edges=edges.tolist(), area=hist.tolist()), Sa_table=dict(zip(thetas.tolist(), table.tolist())))


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK2, SURF, GRID = "#52514e", "#fcfcfb", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1)
    r = out["EL0.113"]
    th = np.array(sorted(map(int, r["curve"])))
    ax.plot(th, [r["curve"][str(t)] if str(t) in r["curve"] else r["curve"][t] for t in th], "-", color=BLUE, lw=2,
            label="tpmsfem surface model")
    ax.plot(th, [r["staircase_only"][str(t)] if str(t) in r["staircase_only"] else r["staircase_only"][t] for t in th],
            "--", color=BLUE, lw=1, label="staircase + melt-pool rounding only")
    ax.errorbar(ANGLES, SA_0113, yerr=SD_0113, fmt="o", color=ORANGE, ms=5, capsize=2,
                label="measured, Kasperovich et al. 2021")
    ax.set_xlabel("angle between surface normal and build direction [°]")
    ax.set_ylabel("Sa [µm]")
    ax.set_xticks(range(0, 181, 45))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("LPBF IN718, 30 µm layers (3 parameters fitted to 3 points)", fontsize=8.5, loc="left")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    d = np.load("results/literature/kasperovich_2021_tpms.npz")
    v, f, c = d["verts"], d["faces"], d["sa"]
    from matplotlib.colors import Normalize
    cmap = matplotlib.colormaps["Blues"]
    coll = ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=0, antialiased=False, shade=False)
    coll.set_facecolors(cmap(Normalize(5, 55)(c)))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-60)
    t = out["tpms"]
    ax.set_title(f"gyroid sheet cell: area-weighted Sa {t['Sa_area_mean']:.0f} µm,\n"
                 f"{100 * t['downskin_area_fraction_theta_gt_120']:.0f} % of the area is downskin (> 120°)",
                 fontsize=8.5, loc="left")
    sm = matplotlib.cm.ScalarMappable(norm=Normalize(5, 55), cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Sa [µm]")
    fig.savefig("docs/img/roughness_vs_angle.png", dpi=200)


if __name__ == "__main__":
    store = "benchmarks/literature/results/ambench2018_tracks.json"
    r = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if r is None:
        d = json.load(open(store))
        r = d["contour_180W_1600"]["width"] / 2e3 if "contour_180W_1600" in d else 0.05
    main(r)
