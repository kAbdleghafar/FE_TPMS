"""LPBF scan paths for one layer of an implicit part (e.g. a TPMS lattice slice).

The part is a signed-distance-like field ``f(x, y, z)`` (negative inside), as
:class:`tpmsfem.geometry.Field`. At build height ``z`` the layer is ``f(x, y, z) < 0``:
- contours follow the slice boundary, offset inward by ``contour_offset``
  (``f = -contour_offset``);
- hatch vectors fill the region further inside (``f < -hatch_offset``), on parallel lines
  ``hatch`` apart at ``angle_deg`` to x, alternating in direction;
- the laser jumps between vectors with the power off.

Units follow the melt-pool model: mm, mm/s, W.
"""
from __future__ import annotations

import numpy as np

from .meltpool import ScanVector


def slice_mask(field, z, x, y):
    """Solid cells of the layer at height ``z`` on the grid ``x`` x ``y`` (indexing ij)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.asarray(field(X, Y, np.full_like(X, z))) < 0.0


def _polyline_vectors(pts, speed, power, seg):
    """Resample a polyline to segments of about ``seg`` and return scan vectors."""
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < seg:
        return []
    s = np.linspace(0.0, d[-1], max(int(np.ceil(d[-1] / seg)), 1) + 1)
    px, py = np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])
    return [ScanVector(px[i], py[i], px[i + 1], py[i + 1], speed, power) for i in range(len(s) - 1)]


def slice_scan_path(field, z, bounds, *, hatch=0.08, speed=1000.0, power=None, angle_deg=0.0, contour=True,
                    contour_offset=0.04, hatch_offset=None, contour_speed=None, contour_power=None,
                    jump_speed=5000.0, h=0.0025, min_vector=0.01):
    """Scan vectors (contours, then hatching) for the layer ``f(x, y, z) < 0`` inside
    ``bounds = (x0, x1, y0, y1)``. ``hatch_offset`` (default ``contour_offset + hatch / 2``)
    keeps the hatch inside the contour; ``h`` is the sampling step; vectors shorter than
    ``min_vector`` are dropped. Returns the list of :class:`ScanVector` including jumps
    (``power = 0``) and a dict of statistics."""
    x0, x1, y0, y1 = bounds
    hatch_offset = contour_offset + 0.5 * hatch if hatch_offset is None else hatch_offset
    vecs, n_cont, n_hatch, L_cont, L_hatch = [], 0, 0, 0.0, 0.0
    if contour:
        from skimage.measure import find_contours

        xs = np.arange(x0, x1 + 0.5 * h, h)
        ys = np.arange(y0, y1 + 0.5 * h, h)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        F = np.asarray(field(X, Y, np.full_like(X, z)), float)
        for c in find_contours(F, -contour_offset):
            pts = np.c_[x0 + c[:, 0] * h, y0 + c[:, 1] * h]
            v = _polyline_vectors(pts, contour_speed or speed, contour_power, seg=4 * h)
            if v:
                vecs.extend(v)
                n_cont += 1
                L_cont += sum(np.hypot(a.x1 - a.x0, a.y1 - a.y0) for a in v)
    # hatching in a frame rotated by angle_deg
    th = np.radians(angle_deg)
    c_, s_ = np.cos(th), np.sin(th)
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    u = corners @ np.array([c_, s_])                      # along the hatch lines
    w = corners @ np.array([-s_, c_])                     # across them
    us = np.arange(u.min(), u.max() + 0.5 * h, h)
    k = 0
    for wl in np.arange(w.min() + 0.5 * hatch, w.max(), hatch):
        px, py = c_ * us - s_ * wl, s_ * us + c_ * wl
        inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        f = np.where(inside, np.asarray(field(px, py, np.full_like(px, z)), float), 1.0)
        m = f < -hatch_offset
        edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
        segs = [(a, b - 1) for a, b in zip(edges[::2], edges[1::2]) if (b - 1 - a) * h >= min_vector]
        if k % 2:
            segs = [(b, a) for a, b in segs[::-1]]
        for a, b in segs:
            vecs.append(ScanVector(px[a], py[a], px[b], py[b], speed, power))
            n_hatch += 1
            L_hatch += abs(us[b] - us[a])
        if segs:
            k += 1
    # jumps with the laser off
    path = []
    for v in vecs:
        if path and np.hypot(v.x0 - path[-1].x1, v.y0 - path[-1].y1) > 1e-9:
            path.append(ScanVector(path[-1].x1, path[-1].y1, v.x0, v.y0, jump_speed, 0.0))
        path.append(v)
    stats = dict(contours=n_cont, contour_length_mm=float(L_cont), hatch_vectors=n_hatch,
                 hatch_length_mm=float(L_hatch),
                 scan_time_s=float(sum(np.hypot(v.x1 - v.x0, v.y1 - v.y0) / v.speed for v in path)))
    return path, stats
