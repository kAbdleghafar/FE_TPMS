"""Minimal EBSD toolkit for comparing simulated and measured grain structures (cubic phases).

* :func:`read_ctf` -- Oxford HKL ``.ctf`` maps (Bunge Euler angles in degrees);
* :func:`euler_to_matrix` -- orientation matrices whose *rows are the crystal axes in the
  sample frame* (the convention of :mod:`tpmsfem.process.solidification`);
* :func:`misorientation` -- disorientation angle with the 24 proper cubic symmetry operators;
* :func:`segment_grains` -- grains as connected regions without boundaries above a threshold;
* :func:`kam` -- kernel average misorientation (local lattice curvature, e.g. of as-solidified
  material against an annealed substrate).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np


@dataclass
class EBSDMap:
    phase: np.ndarray            # (ny, nx) int, 0 = not indexed
    euler: np.ndarray            # (ny, nx, 3) Bunge [rad]
    step: tuple                  # (dx, dy) [um]
    extra: dict

    @property
    def shape(self):
        return self.phase.shape

    def orientations(self):
        return euler_to_matrix(self.euler)


def read_ctf(path) -> EBSDMap:
    """Read an Oxford HKL Channel Text File."""
    hdr = {}
    skip = None
    with open(path, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if parts[0] in ("XCells", "YCells", "XStep", "YStep"):
                hdr[parts[0]] = float(parts[1])
            if line.startswith("Phase\tX\tY"):
                cols = parts
                skip = i + 1
                break
    if skip is None:
        raise ValueError("no data header found")
    d = np.loadtxt(path, skiprows=skip, encoding="latin-1", dtype=np.float32)
    nx, ny = int(hdr["XCells"]), int(hdr["YCells"])
    d = d[: nx * ny]
    col = {c: k for k, c in enumerate(cols)}
    ph = d[:, col["Phase"]].reshape(ny, nx).astype(np.int16)
    eul = np.radians(d[:, [col["Euler1"], col["Euler2"], col["Euler3"]]].astype(np.float64)).reshape(ny, nx, 3)
    extra = {k: d[:, col[k]].reshape(ny, nx) for k in ("MAD", "BC", "BS", "Bands") if k in col}
    return EBSDMap(ph, eul, (hdr["XStep"], hdr["YStep"]), extra)


def euler_to_matrix(e):
    """Bunge (phi1, Phi, phi2) [rad] -> g with rows = crystal axes in sample coordinates
    (g maps sample to crystal components: v_c = g v_s)."""
    p1, P, p2 = e[..., 0], e[..., 1], e[..., 2]
    c1, s1, c, s, c2, s2 = np.cos(p1), np.sin(p1), np.cos(P), np.sin(P), np.cos(p2), np.sin(p2)
    g = np.empty(e.shape[:-1] + (3, 3))
    g[..., 0, 0] = c1 * c2 - s1 * s2 * c
    g[..., 0, 1] = s1 * c2 + c1 * s2 * c
    g[..., 0, 2] = s2 * s
    g[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    g[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    g[..., 1, 2] = c2 * s
    g[..., 2, 0] = s1 * s
    g[..., 2, 1] = -c1 * s
    g[..., 2, 2] = c
    return g


def cubic_symmetry():
    """The 24 proper rotations of the cubic point group."""
    ops = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, j in enumerate(perm):
                m[i, j] = signs[i]
            if np.isclose(np.linalg.det(m), 1.0):
                ops.append(m)
    return np.array(ops)


SYM = cubic_symmetry()


def misorientation(ga, gb):
    """Disorientation angle [deg] between orientation arrays (..., 3, 3) (cubic)."""
    dg = np.einsum("...ij,...kj->...ik", ga, gb)                 # ga gb^T
    tr = np.einsum("sij,...ji->...s", SYM, dg)                    # trace(S dg) for all S
    c = np.clip((tr.max(axis=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def neighbour_misorientations(g, valid):
    """Misorientation [deg] to the right and lower neighbours (nan where either is invalid)."""
    right = np.full(valid.shape, np.nan)
    down = np.full(valid.shape, np.nan)
    r = misorientation(g[:, :-1], g[:, 1:])
    d = misorientation(g[:-1], g[1:])
    right[:, :-1] = np.where(valid[:, :-1] & valid[:, 1:], r, np.nan)
    down[:-1] = np.where(valid[:-1] & valid[1:], d, np.nan)
    return right, down


def segment_grains(g, valid, threshold=10.0, min_size=5):
    """Grain labels (ny, nx) (-1 = invalid or in grains smaller than ``min_size`` points)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    ny, nx = valid.shape
    right, down = neighbour_misorientations(g, valid)
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = [], []
    m = right < threshold
    a.append(idx[m]); b.append(idx[:, 1:][m[:, :-1]] if False else (idx + 1)[m])
    m = down < threshold
    a.append(idx[m]); b.append((idx + nx)[m])
    a, b = np.concatenate(a), np.concatenate(b)
    adj = coo_matrix((np.ones(len(a)), (a, b)), shape=(ny * nx, ny * nx))
    _, lab = connected_components(adj, directed=False)
    lab = lab.reshape(ny, nx)
    lab[~valid] = -1
    u, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    small = cnt[inv.reshape(lab.shape)] < min_size
    lab[small] = -1
    u2, lab2 = np.unique(lab, return_inverse=True)
    lab2 = lab2.reshape(lab.shape) - (1 if u2[0] == -1 else 0)
    lab2[lab < 0] = -1
    return lab2


def kam(g, valid, max_angle=5.0):
    """Kernel average misorientation [deg] over the 4 nearest neighbours (below max_angle)."""
    right, down = neighbour_misorientations(g, valid)
    acc = np.zeros(valid.shape)
    n = np.zeros(valid.shape)
    for arr, sl_a, sl_b in ((right, (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                            (down, (slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        v = arr[sl_a]
        ok = np.isfinite(v) & (v < max_angle)
        for sl in (sl_a, sl_b):
            acc[sl] += np.where(ok, v, 0.0)
            n[sl] += ok
    out = np.where(n > 0, acc / np.maximum(n, 1), np.nan)
    out[~valid] = np.nan
    return out


def grain_shapes(labels, step):
    """Per-grain area [um^2], equivalent diameter, centroid, and the principal axes of the
    second moments (length, width [um], major-axis angle from the map's vertical [deg])."""
    ny, nx = labels.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    ok = labels >= 0
    lab, x, y = labels[ok], X[ok] * step[0], Y[ok] * step[1]
    n = lab.max() + 1
    cnt = np.bincount(lab, minlength=n).astype(float)
    mx, my = np.bincount(lab, x, n) / cnt, np.bincount(lab, y, n) / cnt
    dx, dy = x - mx[lab], y - my[lab]
    sxx, syy, sxy = (np.bincount(lab, v, n) / cnt for v in (dx * dx, dy * dy, dx * dy))
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    l1 = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    l2 = tr / 2 - np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    ang = np.degrees(0.5 * np.arctan2(2 * sxy, syy - sxx))          # 0 = vertical major axis
    area = cnt * step[0] * step[1]
    return dict(area=area, d_eq=2 * np.sqrt(area / np.pi), cx=mx, cy=my,
                length=4 * np.sqrt(np.maximum(l1, 0)), width=4 * np.sqrt(np.maximum(l2, 0)), angle=ang)
