"""TPMSsuite thumbnail: 1280 x 640 px (GitHub social preview, 2:1) and README banner.

Left: name and tagline over four result tiles (all computed by TPMSsuite):
mechanics (von Mises in a gyroid cube), melt pool (surface temperature of a laser track),
cells (Nb map from the phase field) and grains (IPF map from the 3-D grain CA).
Right: a gyroid sheet lattice, coloured by the von Mises stress of a compression test.

    python tools/branding/make_social_preview.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import tpmsfem as tf  # noqa: E402
from tpmsfem.fem.materials import TI6AL4V_LPBF  # noqa: E402
from tpmsfem.geometry import TPMS, box, lattice_cube  # noqa: E402
from tpmsfem.mesh import implicit_to_surface, voxelize  # noqa: E402
from tpmsfem.post import results as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "img", "tpmssuite_social_preview.png")
BG, INK, MUTED, ACCENT = "#0f1b2d", "#f4f6fa", "#9fb0c8", "#4cc3ff"


def gyroid_stress():
    """Compression of a 2 x 2 x 2-cell gyroid; von Mises mapped to the surface triangles."""
    L, size = 4.0, 8.0
    lat = TPMS("gyroid", cell_size=L, kind="sheet", density=0.3)
    mesh = voxelize(lattice_cube(lat, size, plate=0.4), (0, 0, 0), (size,) * 3, 0.2)
    md = tf.SolidModel(mesh, TI6AL4V_LPBF.elastic())
    z = mesh.points[:, 2]
    top = np.flatnonzero(z > size - 1e-9)
    md.fix(np.flatnonzero(z < 1e-9))
    md.fix(top, (0, 1), 0.0)
    md.fix(top, (2,), -0.008)
    md.solve()
    vm = R.result_fields(md)["vonMises_nodal"]
    v, f = implicit_to_surface(box((0, 0, 0.4), (size, size, size - 0.4)) & lat, (0, 0, 0), (size,) * 3, 0.1)
    from scipy.spatial import cKDTree
    d, j = cKDTree(mesh.points).query(v, k=8)                     # inverse-distance average of nearby nodes
    w = 1.0 / np.maximum(d, 1e-6)
    sv = (vm[j] * w).sum(axis=1) / w.sum(axis=1)
    return v, f, sv, md, mesh


def tile_mechanics(ax, md, mesh):
    vm = R.result_fields(md)["vonMises_elem"]
    c = mesh.centroids()
    h = mesh.meta["voxel_size"]
    sel = np.abs(c[:, 1] - 4.0) < 0.5 * h
    img = np.full((40, 40), np.nan)
    i = np.clip((c[sel, 0] / h).astype(int), 0, 39)
    k = np.clip((c[sel, 2] / h).astype(int), 0, 39)
    img[k, i] = vm[sel]
    cm = plt.get_cmap("turbo").copy()
    cm.set_bad(BG)
    ax.imshow(np.ma.masked_invalid(img), origin="lower", cmap=cm, interpolation="nearest", aspect="auto",
              vmin=np.nanpercentile(img, 5), vmax=np.nanpercentile(img, 98))


def tile_meltpool(ax):
    from tpmsfem.fem.materials import Material, ThermalProperties
    from tpmsfem.process import Evaporation, Laser, MeltPoolModel, ScanVector
    th = ThermalProperties.from_si([20.0, 1290.0, 1350.0, 3000.0], [9.8, 26.0, 30.0, 30.0], [410.0, 650.0, 720.0, 720.0],
                                   8440.0, 2.9e5, 1290.0, 1350.0)
    mp = MeltPoolModel(Material("IN625", 208e3, 0.28, thermal=th), 0.01, 0.6, 0.15, 0.08, T0=25.0,
                       evaporation=Evaporation())
    t_snap = 0.4 / 800.0
    res = mp.run(Laser(180.0, 0.35, 0.05, depth=0.015), [ScanVector(0.05, 0.0, 0.55, 0.0, 800.0)],
                 snapshot_times=[t_snap])
    T = res.snapshots[0][1][:, :, -1]
    top = np.concatenate([T[:, ::-1], T], axis=1).T
    ax.imshow(top[:, 15:58], origin="lower", cmap="inferno", vmin=25, vmax=2600, aspect="auto", interpolation="bicubic")
    ax.contour(top[:, 15:58], [1290.0], colors=[INK], linewidths=0.8)


def tile_cells(ax):
    c = np.load(os.path.join(ROOT, "benchmarks/literature/results/pf_path_trapping_fields_z-32p5_trap.npz"))["c"]
    ax.imshow(c.astype(float), origin="lower", cmap="magma", vmin=2.5, vmax=7.0, aspect="auto")


def tile_grains(ax):
    p = np.load(os.path.join(ROOT, "benchmarks/literature/results/ca_multilayer_rot67_side480_dx1.npz"))["plan_bd"]
    ax.imshow(np.transpose(p.astype(float), (1, 0, 2))[:300, :180], origin="lower", interpolation="nearest",
              aspect="auto")


def main():
    v, f, s, md, mesh = gyroid_stress()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=BG)
    # title block
    fig.text(0.045, 0.80, "TPMSsuite", fontsize=58, fontweight="bold", color=INK, family="DejaVu Sans")
    fig.text(0.048, 0.725, "Open-source, GPU-ready simulation of TPMS structures", fontsize=17, color=ACCENT)
    fig.text(0.048, 0.665, "design  ·  meshing  ·  mechanics  ·  heat  ·  AM process  ·  microstructure",
             fontsize=13, color=MUTED)
    # tiles
    tiles = [("mechanics", tile_mechanics, (md, mesh)), ("melt pool", tile_meltpool, ()),
             ("cells & segregation", tile_cells, ()), ("grains & texture", tile_grains, ())]
    w, gap, x0, y0, hgt = 0.118, 0.012, 0.048, 0.17, 0.36
    for k, (name, fn, args) in enumerate(tiles):
        ax = fig.add_axes([x0 + k * (w + gap), y0, w, hgt])
        fn(ax, *args)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2b3d57")
        fig.text(x0 + k * (w + gap), y0 - 0.055, name, fontsize=11.5, color=INK)
    fig.text(0.048, 0.045, "verified against analytical solutions, CalculiX / Abaqus and published experiments",
             fontsize=10.5, color=MUTED)
    # gyroid render
    ax = fig.add_axes([0.56, -0.02, 0.46, 1.04], projection="3d", facecolor=BG)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(v)
    for a in range(3):
        np.add.at(vn, f[:, a], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    ls = LightSource(azdeg=315, altdeg=35)
    shade_v = np.clip(vn @ ls.direction, 0.0, 1.0) * 0.65 + 0.35
    shade = shade_v[f].mean(axis=1)
    cmap = plt.get_cmap("turbo")
    sf = s[f].mean(axis=1)
    cols = cmap(Normalize(np.percentile(sf, 3), np.percentile(sf, 99), clip=True)(sf))
    cols[:, :3] *= shade[:, None]
    pc = Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=0.25)
    ax.add_collection3d(pc)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 8)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor=BG)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
