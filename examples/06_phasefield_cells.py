"""Cells and Nb segregation in LPBF IN625 with the 2-D quantitative phase field.

The solidification conditions are those of the AM-Bench case-B melt pool 5 um above its bottom
(benchmarks/literature/nist_ambench2018_tracks_pf.py): G = 1.25e7 K/m, R = 0.074 m/s. IN625 is
treated as quasi-binary Ni-4 wt% Nb (``phasefield.IN625_NB``). The run:

1. starts from a planar front at the solidus with the steady solute boundary layer and noise;
2. lets the Mullins-Sekerka instability break it into cells;
3. follows the cell array as it coarsens towards a steady spacing;
4. writes the Nb map and the spacing / tip-composition history.

    python examples/06_phasefield_cells.py       # ~2 min on 4 CPU cores (numba), slower without
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tpmsfem.process import phasefield as PF  # noqa: E402
from tpmsfem.process import solidification as S  # noqa: E402

OUT = "results/process"
G, R = 1.25e7, 0.074
alloy = PF.IN625_NB
pf = PF.DirectionalPF(alloy, G, R, width_m=2.5e-6, height_m=2.0e-6, W0_m=10e-9, seed=0)
print(f"IN625 as Ni-{alloy.c_inf:g}Nb: dT0 = {alloy.freezing_range:.1f} K, d0 = {alloy.d0 * 1e9:.1f} nm; "
      f"V / V_cs = {R / PF.constitutional_velocity(alloy, G):.0f}, V / V_abs = "
      f"{R / PF.absolute_stability_velocity(alloy):.2f}; grid {pf.nx} x {pf.nz}, dx = {pf.dx * pf.W0 * 1e9:.0f} nm, "
      f"kernel = {pf.kernel}")

history = []
for i in range(20):
    r = pf.run(0.4e-6 / R)                              # 0.4 um of growth per chunk
    tips = r.tips()
    history.append(dict(grown_um=R * r.t_s * 1e6, cells=len(tips),
                        spacing_um=r.spacing()[0] * 1e6,
                        cs_wt=float(np.median([t["cs"] for t in tips])) * alloy.c_inf if tips else None))
    print(f"  grown {history[-1]['grown_um']:4.1f} um: {len(tips)} cells, spacing {history[-1]['spacing_um']:.2f} um "
          f"[{r.seconds:.0f} s]", flush=True)

lam_kf = float(S.primary_spacing(G, R, S.SolidificationConstants(freezing_range=alloy.freezing_range), "kurz-fisher"))
lam_hunt = float(S.primary_spacing(G, R, S.SolidificationConstants(freezing_range=alloy.freezing_range), "hunt"))
print(f"phase field: {history[-1]['spacing_um']:.2f} um; Hunt {lam_hunt:.2f} um, Kurz-Fisher {lam_kf:.2f} um "
      f"(spacing laws; note R = {R} m/s is above the ~0.05 m/s where W0 = 10 nm is converged)")

os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
top = int(np.flatnonzero((r.phi > 0).any(axis=1)).max())
c = r.c[max(top - 150, 0):top + 10] * alloy.c_inf
ext = [0, c.shape[1] * r.dx_m * 1e6, 0, c.shape[0] * r.dx_m * 1e6]
im = ax[0].imshow(c, origin="lower", cmap="magma", extent=ext, vmin=2.0, vmax=8.0)
ax[0].contour(r.phi[max(top - 150, 0):top + 10], [0.0], colors="w", linewidths=0.5, extent=ext)
fig.colorbar(im, ax=ax[0], label="Nb [wt%]")
ax[0].set_xlabel("x [um]")
ax[0].set_ylabel("z [um]")
ax[0].set_title(f"Nb, G = {G:.2e} K/m, R = {R} m/s", fontsize=9)
ax[1].plot([h["grown_um"] for h in history], [h["spacing_um"] for h in history], "o-")
ax[1].axhline(lam_hunt, color="C2", ls="--", label="Hunt")
ax[1].set_xlabel("growth distance [um]")
ax[1].set_ylabel("primary spacing [um]")
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "phasefield_cells.png"), dpi=130)
json.dump(dict(G=G, R=R, alloy=alloy.__dict__, history=history, hunt_um=lam_hunt, kurz_fisher_um=lam_kf,
               segregation=r.segregation()), open(os.path.join(OUT, "phasefield_cells.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, "phasefield_cells.png"))
