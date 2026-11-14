"""Animation of one LPBF track over a powder layer with the free-surface lattice Boltzmann
melt pool: (left) the metal surface in 3-D, coloured by temperature; (right) the longitudinal
section through the track centre (metal and temperature).

    python tools/lbm/freesurface_animation.py run  <out.npz>     # simulate, ~10-15 min on a CPU
    python tools/lbm/freesurface_animation.py movie <out.npz> <out.mp4> [smooth|raw]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmarks", "literature"))

H = 5e-6
LX, LY, PLATE, LAYER, GAS = 0.40e-3, 0.24e-3, 60e-6, 40e-6, 50e-6
P, V, RADIUS, A, DELTA = 70.0, 0.8, 50e-6, 0.40, 5e-6
X0, X1 = 0.06e-3, 0.34e-3


def simulate(out, every=50, seed=5):
    import in625_material as IM

    import tpmsfem as tf
    from tpmsfem.process import Evaporation, Laser, ScanVector
    from tpmsfem.process.freesurface import FreeSurfaceMeltPool
    from tpmsfem.process.powder import rain_powder, voxelise

    nx, ny = int(round(LX / H)), int(round(LY / H))
    npl, nla, nga = int(round(PLATE / H)), int(round(LAYER / H)), int(round(GAS / H))
    shape = (nx + 2, ny + 2, npl + nla + nga + 2)
    fill = np.zeros(shape)
    fill[1:-1, 1:-1, 1:npl + 1] = 1.0
    bed = rain_powder((H, H + LX, H, H + LY), (npl + 1) * H, LAYER, seed=seed)
    fill = np.maximum(fill, voxelise(bed, H, shape))
    mat = tf.Material("IN625", 208_000.0, 0.278, thermal=IM.thermal_properties())
    md = FreeSurfaceMeltPool(mat, H, fill, dt=1e-7, viscosity=5e-3, sigma=1.8, T0=25.0, T_max=8000.0,
                             dsigma_dT=-1.1e-4, evaporation=Evaporation(), emissivity=0.4, energy_transport="lbm")
    y = H + 0.5 * LY
    res = md.run(Laser(P, A, RADIUS, depth=DELTA), [ScanVector(H + X0, y, H + X1, y, V)], t_extra=40e-6,
                 verbose=True, snapshot_every=every)
    t = np.array([s[0] for s in res.snapshots])
    T = np.stack([s[1] for s in res.snapshots]).astype(np.float32)
    F = np.stack([s[2] for s in res.snapshots]).astype(np.float16)
    np.savez_compressed(out, t=t, T=T, fill=F, h=H, plate_top=(npl + 1) * H, x0=X0, v=V, t_scan=(X1 - X0) / V,
                        energy_balance=res.energy_balance, mass_error=res.mass_error, limited=md.lbm.limited,
                        P=P, radius=RADIUS, A=A)
    print(f"saved {len(t)} frames; mass {res.mass_error:.1e}, energy {res.energy_balance:+.2e}, "
          f"limiter {md.lbm.limited}")


def surface(fill, h):
    """Height of the topmost metal per column [m] and the index of that cell."""
    nz = fill.shape[2]
    has = fill > 0.02
    k = nz - 1 - np.argmax(has[:, :, ::-1], axis=2)
    any_ = has.any(axis=2)
    f = np.take_along_axis(fill, k[:, :, None], axis=2)[:, :, 0]
    return np.where(any_, (k + f) * h, 0.0), k


def iso_mesh(fill, T, h_um, z0_um, sigma=0.8):
    """Smoothed isosurface of the metal (fill level 0.5 after a Gaussian filter of ``sigma``
    cells): triangle vertices [um] and the temperature at each triangle. Display only: the
    smoothing does not touch the simulated fields."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(gaussian_filter(fill, sigma), 0.5)
    Tf = map_coordinates(T, verts.T, order=1)[faces].mean(axis=1)
    P = verts * h_um
    P[:, 2] -= z0_um
    return P[faces], Tf


def movie(npz, out, fps=20, style="smooth"):
    import matplotlib
    matplotlib.use("Agg")
    import subprocess
    import tempfile

    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, Normalize

    d = np.load(npz)
    t, T, F = d["t"], d["T"], d["fill"].astype(np.float32)
    h = float(d["h"])
    pt = float(d["plate_top"])
    norm = Normalize(25, 2800)
    cmap = plt.get_cmap("RdYlBu_r")              # cold metal blue, melt orange to red
    light = LightSource(azdeg=315, altdeg=40)
    sun = np.array([-0.4, -0.5, 0.75])
    sun /= np.linalg.norm(sun)
    tmp = tempfile.mkdtemp()
    x = (np.arange(T.shape[1] - 2) + 0.5) * h * 1e6
    y = (np.arange(T.shape[2] - 2) + 0.5) * h * 1e6
    X, Y = np.meshgrid(x, y, indexing="ij")
    j = T.shape[2] // 2
    for n in range(len(t)):
        fill, Tn = F[n], T[n]
        zs, k = surface(fill, h)
        Ts = np.take_along_axis(Tn, k[:, :, None], axis=2)[:, :, 0]
        Z = (zs[1:-1, 1:-1] - pt) * 1e6
        Ts = Ts[1:-1, 1:-1]
        fig = plt.figure(figsize=(14, 5.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.05, left=0.0, right=0.95)
        ax = fig.add_subplot(gs[0], projection="3d")
        xl = (float(d["x0"]) + float(d["v"]) * min(t[n], float(d["t_scan"]))) * 1e6
        on = t[n] <= float(d["t_scan"])
        if style == "smooth":
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            # interior cells only (the wall layer is dropped), cell (0, 0, 0) at the origin
            tri, Tf = iso_mesh(fill[1:-1, 1:-1, 1:-1], Tn[1:-1, 1:-1, 1:-1], h * 1e6, pt * 1e6 - h * 1e6)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.clip(-(nrm @ sun), 0.0, 1.0)
            col = cmap(norm(Tf))[:, :3] * shade[:, None]
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors=col, linewidths=0.2))
            if on:                                  # the beam, drawn last and on top
                ax.plot([xl, xl], [y[-1] / 2, y[-1] / 2], [40, 75], color="#1baf7a", lw=4, zorder=100,
                        solid_capstyle="round")
                ax.scatter([xl], [y[-1] / 2], [40], color="#1baf7a", s=40, zorder=101, depthshade=False)
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, y[-1])
            ax.set_zlim(-60, 75)
            ax.set_box_aspect((x[-1], y[-1], 135), zoom=0.92)
            ax.view_init(elev=32, azim=-60)
            ax.set_axis_off()
        else:
            rgb = light.shade_rgb(cmap(norm(Ts))[..., :3], Z, blend_mode="soft", vert_exag=0.4)
            ax.plot_surface(X, Y, Z, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False,
                            shade=False)
            ax.set_zlim(-40, 80)
            ax.set_box_aspect((x[-1], y[-1], 120 * 1.6))
            ax.view_init(elev=38, azim=-62)
            ax.set_xlabel("x [um]", fontsize=8)
            ax.set_ylabel("y [um]", fontsize=8)
            ax.set_zlabel("height [um]", fontsize=8)
            ax.tick_params(labelsize=7)
        what = "smoothed metal surface (display only), true scale" if style == "smooth" else "metal surface"
        ax.set_title(f"{what}, coloured by temperature\nlaser {'at x = %.0f um' % xl if on else 'off'}",
                     fontsize=9.5)
        # longitudinal section
        ax2 = fig.add_subplot(gs[1])
        sec_f = fill[1:-1, j, 1:-1].T
        sec_T = np.where(sec_f > 0.5, Tn[1:-1, j, 1:-1].T, np.nan)
        zz = (np.arange(fill.shape[2] - 2) + 0.5) * h * 1e6 - (pt * 1e6 - h * 1e6)
        im = ax2.pcolormesh(x, zz, sec_T, cmap=cmap, norm=norm, shading="nearest")
        ax2.contour(x, zz, sec_f, [0.5], colors=["#333333"], linewidths=0.6)
        melt = np.where(sec_f > 0.5, Tn[1:-1, j, 1:-1].T, 0.0)
        ax2.contour(x, zz, melt, [1290.0], colors=["#000000"], linewidths=1.0, linestyles="--")
        ax2.axhline(0.0, color="#888888", lw=0.6, ls=":")
        if on:
            ax2.axvline(xl, color="#1baf7a", lw=1.2)
        ax2.set_ylim(-45, 85)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x along the track [um]", fontsize=8.5)
        ax2.set_ylabel("z above the plate [um]", fontsize=8.5)
        ax2.set_title("section through the track centre\nblack dashed: melt boundary (solidus), green: laser",
                      fontsize=9.5)
        ax2.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        cb.set_label("temperature [C]", fontsize=8)
        fig.suptitle(f"One LPBF track over a 40 um powder layer (IN625, {float(d['P']):.0f} W, "
                     f"{float(d['v']):.1f} m/s): free-surface lattice Boltzmann, 5 um cells    t = {t[n] * 1e6:5.1f} us",
                     fontsize=10.5, x=0.02, y=0.99, ha="left")
        fig.savefig(os.path.join(tmp, f"f{n:04d}.png"), dpi=110)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20",
                    out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        simulate(sys.argv[2])
    else:
        movie(sys.argv[2], sys.argv[3], style=sys.argv[4] if len(sys.argv) > 4 else "smooth")
