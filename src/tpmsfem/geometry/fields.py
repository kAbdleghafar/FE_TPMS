"""Implicit geometry: composable scalar fields (signed-distance-like).

Convention: ``phi(x, y, z) < 0`` inside the solid, ``> 0`` outside. Fields
evaluate on NumPy *or* CuPy arrays, so a voxel grid of a whole implant can be
classified on the GPU.

Boolean algebra uses min/max, which only needs the sign to be right; offsets and
shells assume the field is close to a true distance (in mm) near the surface,
which holds for the primitives here and approximately for the TPMS fields
(they are rescaled by L/2pi).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..backend import array_module


class Field:
    """A scalar field with boolean / offset operators.

    ``a | b`` union, ``a & b`` intersection, ``a - b`` difference,
    ``~a`` complement, ``a.shell(t)`` hollow shell of thickness ``t``.
    """

    def __init__(self, fn: Callable, name: str = "field"):
        self.fn = fn
        self.name = name

    def __call__(self, x, y, z):
        return self.fn(x, y, z)

    # boolean algebra -------------------------------------------------------
    def __or__(self, other: "Field") -> "Field":
        return Field(lambda x, y, z: array_module(x).minimum(self(x, y, z), other(x, y, z)),
                     f"({self.name}|{other.name})")

    def __and__(self, other: "Field") -> "Field":
        return Field(lambda x, y, z: array_module(x).maximum(self(x, y, z), other(x, y, z)),
                     f"({self.name}&{other.name})")

    def __sub__(self, other: "Field") -> "Field":
        return Field(lambda x, y, z: array_module(x).maximum(self(x, y, z), -other(x, y, z)),
                     f"({self.name}-{other.name})")

    def __invert__(self) -> "Field":
        return Field(lambda x, y, z: -self(x, y, z), f"~{self.name}")

    # offsets -----------------------------------------------------------------
    def offset(self, d: float) -> "Field":
        """Grow (d > 0) or shrink (d < 0) the solid by ``d``."""
        return Field(lambda x, y, z: self(x, y, z) - d, f"offset({self.name},{d})")

    def shell(self, t: float) -> "Field":
        """Hollow shell of thickness ``t`` centred on the surface."""
        return Field(lambda x, y, z: array_module(x).abs(self(x, y, z)) - 0.5 * t,
                     f"shell({self.name},{t})")

    def inner_shell(self, t: float) -> "Field":
        """Shell of thickness ``t`` lying just inside the surface."""
        return self - self.offset(-t)

    # transforms --------------------------------------------------------------
    def translate(self, d: Sequence[float]) -> "Field":
        dx, dy, dz = d
        return Field(lambda x, y, z: self(x - dx, y - dy, z - dz), self.name)

    def rotate(self, R) -> "Field":
        """Rotate the geometry by the 3x3 rotation matrix ``R`` (about the origin)."""
        R = np.asarray(R, dtype=float)
        Rt = R.T  # inverse rotation applied to query points

        def fn(x, y, z):
            xr = Rt[0, 0] * x + Rt[0, 1] * y + Rt[0, 2] * z
            yr = Rt[1, 0] * x + Rt[1, 1] * y + Rt[1, 2] * z
            zr = Rt[2, 0] * x + Rt[2, 1] * y + Rt[2, 2] * z
            return self(xr, yr, zr)

        return Field(fn, self.name)

    def __repr__(self):
        return f"Field<{self.name}>"


# --------------------------------------------------------------------------
# primitives (exact or near-exact signed distances, in model units)
# --------------------------------------------------------------------------
def sphere(center=(0.0, 0.0, 0.0), radius=1.0) -> Field:
    cx, cy, cz = center

    def fn(x, y, z):
        xp = array_module(x)
        return xp.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - radius

    return Field(fn, "sphere")


def box(lo, hi) -> Field:
    """Axis-aligned box between corners ``lo`` and ``hi``."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)

    def fn(x, y, z):
        xp = array_module(x)
        qx = xp.abs(x - c[0]) - h[0]
        qy = xp.abs(y - c[1]) - h[1]
        qz = xp.abs(z - c[2]) - h[2]
        outside = xp.sqrt(xp.maximum(qx, 0) ** 2 + xp.maximum(qy, 0) ** 2 + xp.maximum(qz, 0) ** 2)
        inside = xp.minimum(xp.maximum(qx, xp.maximum(qy, qz)), 0)
        return outside + inside

    return Field(fn, "box")


def half_space(point, normal) -> Field:
    """Solid on the side opposite to ``normal`` (i.e. ``(x-p).n < 0``)."""
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)

    def fn(x, y, z):
        return (x - p[0]) * n[0] + (y - p[1]) * n[1] + (z - p[2]) * n[2]

    return Field(fn, "halfspace")


def round_cone(p0, p1, r0, r1) -> Field:
    """Tapered capsule from ``p0`` (radius ``r0``) to ``p1`` (radius ``r1``).

    Distance is ``|x - proj| - r(s)`` with ``s`` the clamped projection
    parameter: exact for a cylinder/capsule and accurate for mild tapers.
    """
    a = np.asarray(p0, float)
    b = np.asarray(p1, float)
    ab = b - a
    L2 = float(ab @ ab)

    def fn(x, y, z):
        xp = array_module(x)
        px, py, pz = x - a[0], y - a[1], z - a[2]
        s = xp.clip((px * ab[0] + py * ab[1] + pz * ab[2]) / L2, 0.0, 1.0)
        dx, dy, dz = px - s * ab[0], py - s * ab[1], pz - s * ab[2]
        return xp.sqrt(dx * dx + dy * dy + dz * dz) - (r0 + s * (r1 - r0))

    return Field(fn, "round_cone")


def capsule(p0, p1, r) -> Field:
    return round_cone(p0, p1, r, r)


def cylinder(p0, p1, r) -> Field:
    """Flat-capped cylinder between ``p0`` and ``p1``."""
    a = np.asarray(p0, float)
    b = np.asarray(p1, float)
    ab = b - a
    L = float(np.linalg.norm(ab))
    e = ab / L

    def fn(x, y, z):
        xp = array_module(x)
        px, py, pz = x - a[0], y - a[1], z - a[2]
        t = px * e[0] + py * e[1] + pz * e[2]
        rx, ry, rz = px - t * e[0], py - t * e[1], pz - t * e[2]
        d_rad = xp.sqrt(rx * rx + ry * ry + rz * rz) - r
        d_ax = xp.abs(t - 0.5 * L) - 0.5 * L
        outside = xp.sqrt(xp.maximum(d_rad, 0) ** 2 + xp.maximum(d_ax, 0) ** 2)
        return outside + xp.minimum(xp.maximum(d_rad, d_ax), 0)

    return Field(fn, "cylinder")


def zone_z(z0: float, z1: float) -> Field:
    """Slab ``z0 <= z <= z1`` (useful to restrict a lattice to a region)."""
    return Field(lambda x, y, z: array_module(x).maximum(z0 - z, z - z1), "zone_z")


def from_function(fn: Callable, name="custom") -> Field:
    return Field(fn, name)


def rotation_matrix(axis, angle_deg) -> np.ndarray:
    """Rodrigues rotation matrix about ``axis`` by ``angle_deg`` degrees."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    a = np.deg2rad(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
