"""Compare an OpenLB total-enthalpy LBM run of the AM-Bench case-B track (tools/lbm/lpbfTrack3d.cpp)
with tpmsfem's MeltPoolModel on the same problem: conduction + melting, constant properties,
no surface losses or evaporation, the same moving Gaussian source with absorption depth.

    python tools/lbm/compare_openlb_track.py <path to enthalpy.bin> [cell size in um]
"""
import json
import sys
import time

import numpy as np

import tpmsfem as tf
from tpmsfem.fem.materials import ThermalProperties
from tpmsfem.process import Laser, MeltPoolModel, ScanVector

RHO, CP, K, LAT, TS, TL, T0 = 8440.0, 650.0, 25.0, 2.72e5, 1290.0, 1350.0, 25.0
P, A, W, DELTA, V = 179.2, 0.3234, 0.085, 0.0153, 800.0          # W, -, mm, mm, mm/s
LX, LY, LZ, X0, X1 = 1.0, 0.3, 0.15, 0.15, 0.85                    # mm


def temperature_from_enthalpy(H):
    Hs, Hl = TS, TL + LAT / CP
    return np.where(H <= Hs, H, np.where(H >= Hl, H - LAT / CP, TS + (H - Hs) / (Hl - Hs) * (TL - TS)))


def pool(Tf, h, level=TS):
    """Length, width, depth [um] of the region at or above ``level`` in a (nx, ny, nz) field
    whose top cell layer is the surface."""
    m = Tf >= level
    if not m.any():
        return dict(length=0.0, width=0.0, depth=0.0)
    i, j, k = np.nonzero(m)
    nz = Tf.shape[2]
    return dict(length=float((i.max() - i.min() + 1) * h * 1e3), width=float((j.max() - j.min() + 1) * h * 1e3),
                depth=float((nz - k.min()) * h * 1e3))


def main(path, res_um=5.0):
    h = res_um * 1e-3
    nx, ny, nz = int(round(LX / h)), int(round(LY / h)), int(round(LZ / h))
    H = np.fromfile(path, dtype=np.float64).reshape(nz, ny, nx).transpose(2, 1, 0)
    T_lbm = temperature_from_enthalpy(H)
    th = ThermalProperties.from_si([0.0, 6000.0], [K, K], [CP, CP], RHO, LAT, TS, TL)
    mat = tf.Material("const", 200_000.0, 0.3, thermal=th)
    md = MeltPoolModel(mat, h, length=LX, width=LY, depth=LZ, symmetric_y=False, T0=T0, emissivity=0.0, h_conv=0.0,
                       evaporation=None, T_max=20000.0)
    t_end = (X1 - X0) / V
    t0 = time.perf_counter()
    res = md.run(Laser(P, A, W, depth=DELTA), [ScanVector(X0, 0.0, X1, 0.0, V)], t_extra=0.0)
    sec = time.perf_counter() - t0
    T_fd = np.asarray(res.T)
    p_lbm, p_fd = pool(T_lbm, h), pool(T_fd, h)
    hot = (T_fd > 500.0) | (T_lbm > 500.0)
    diff = T_lbm - T_fd
    out = dict(cell_um=res_um, cells=int(T_fd.size), t_end_ms=t_end * 1e3,
               tpmsfem=dict(pool_um=p_fd, T_max=float(T_fd.max()), steps=res.steps, seconds=sec,
                            energy_balance=float(res.energy_balance)),
               openlb=dict(pool_um=p_lbm, T_max=float(T_lbm.max())),
               difference=dict(rms_K_where_hot=float(np.sqrt(np.mean(diff[hot] ** 2))),
                               max_abs_K=float(np.abs(diff).max()),
                               rel_rms_where_hot=float(np.sqrt(np.mean(diff[hot] ** 2)) / np.mean(T_fd[hot] - T0)),
                               energy_ratio=float((H - T0).sum() / (np.asarray(md.H).sum() / (RHO * 1e-12 * CP * 1e6)
                                                                    - T0 * T_fd.size))))
    print(json.dumps(out, indent=1))
    return out, T_lbm, T_fd


if __name__ == "__main__":
    o, T_lbm, T_fd = main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0)
    np.savez_compressed(sys.argv[3] if len(sys.argv) > 3 else "lbm_compare.npz", T_lbm=T_lbm.astype(np.float32),
                        T_fd=T_fd.astype(np.float32))


def figure(npz, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz)
    T_lbm, T_fd = d["T_lbm"], d["T_fd"]
    h = LX / T_fd.shape[0]
    nx, ny, nz = T_fd.shape
    jc = ny // 2
    ext_top = [0, LX, -LY / 2, LY / 2]
    ext_side = [0, LX, -LZ, 0]
    fig, ax = plt.subplots(3, 2, figsize=(12, 7.2), gridspec_kw=dict(height_ratios=[1, 1, 1]))
    kw = dict(origin="lower", cmap="inferno", vmin=25, vmax=2400, aspect="equal")
    for c, (T, name) in enumerate(((T_fd, "tpmsfem (finite volume, NumPy)"), (T_lbm, "OpenLB (total-enthalpy LBM, C++)"))):
        im = ax[0, c].imshow(T[:, :, -1].T, extent=ext_top, **kw)
        ax[0, c].contour((np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h - LY / 2, T[:, :, -1].T, [TS], colors="w",
                         linewidths=0.8)
        ax[0, c].set_title(f"{name}: top surface", fontsize=9, loc="left")
        ax[1, c].imshow(T[:, jc, :].T, extent=ext_side, **kw)
        ax[1, c].contour((np.arange(nx) + 0.5) * h, (np.arange(nz) + 0.5) * h - LZ, T[:, jc, :].T, [TS], colors="w",
                         linewidths=0.8)
        ax[1, c].set_title("centre-line section (white: solidus)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax[:2, :].ravel().tolist(), fraction=0.02, pad=0.01).set_label("temperature [C]", fontsize=8.5)
    diff = T_lbm - T_fd
    im2 = ax[2, 0].imshow(diff[:, jc, :].T, extent=ext_side, origin="lower", cmap="RdBu_r", vmin=-100, vmax=100,
                          aspect="equal")
    ax[2, 0].set_title("OpenLB minus tpmsfem, centre-line section [K]", fontsize=9, loc="left")
    fig.colorbar(im2, ax=ax[2, 0], fraction=0.03, pad=0.02)
    x = (np.arange(nx) + 0.5) * h
    ax[2, 1].plot(x, T_fd[:, jc, -1], color="#2a78d6", lw=1.5, label="tpmsfem")
    ax[2, 1].plot(x, T_lbm[:, jc, -1], "--", color="#eb6834", lw=1.5, label="OpenLB")
    ax[2, 1].axhline(TS, color="#52514e", lw=0.8, ls=":")
    ax[2, 1].set_xlabel("x [mm]", fontsize=8.5)
    ax[2, 1].set_ylabel("surface temperature on the track [C]", fontsize=8.5)
    ax[2, 1].legend(fontsize=8, frameon=False)
    ax[2, 1].grid(True, color="#e4e3df", lw=0.6)
    for a in ax.ravel():
        a.tick_params(labelsize=7.5, colors="#52514e")
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    fig.suptitle("AM-Bench case-B track, conduction + melting, identical physics: tpmsfem vs OpenLB (5 um cells, t = 0.875 ms)",
                 fontsize=10, x=0.02, ha="left")
    fig.savefig(out, dpi=150)
    print("wrote", out)
