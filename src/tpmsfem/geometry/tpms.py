"""Triply periodic minimal surface (TPMS) lattices as implicit fields.

Each surface is the zero set of a periodic function ``f(X, Y, Z)`` with
``X = 2*pi*x/L``. Two solid variants are supported:

* ``"sheet"``   (matrix / shell lattice): solid where ``|f| < c``
* ``"network"`` (skeletal / ligament):    solid where ``f < c`` (``side="lower"``)
  or ``f > c`` (``side="upper"``). For the asymmetric I-WP the two sides are
  different lattices: ``upper`` is the body-centred "I" labyrinth (ligaments
  along the cube diagonals), ``lower`` the "WP" labyrinth.

The level ``c`` is set directly or, more usefully, from a target *relative
density* (scalar or a spatial function for functionally graded lattices); a
density->level lookup table is computed once per surface by sampling a unit
cell.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from ..backend import array_module
from .fields import Field

TWO_PI = 2.0 * np.pi


def gyroid(X, Y, Z):
    xp = array_module(X)
    return xp.sin(X) * xp.cos(Y) + xp.sin(Y) * xp.cos(Z) + xp.sin(Z) * xp.cos(X)


def schwarz_p(X, Y, Z):
    xp = array_module(X)
    return xp.cos(X) + xp.cos(Y) + xp.cos(Z)


def diamond(X, Y, Z):
    xp = array_module(X)
    sx, sy, sz = xp.sin(X), xp.sin(Y), xp.sin(Z)
    cx, cy, cz = xp.cos(X), xp.cos(Y), xp.cos(Z)
    return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz


def iwp(X, Y, Z):
    xp = array_module(X)
    cx, cy, cz = xp.cos(X), xp.cos(Y), xp.cos(Z)
    return 2.0 * (cx * cy + cy * cz + cz * cx) - (xp.cos(2 * X) + xp.cos(2 * Y) + xp.cos(2 * Z))


def neovius(X, Y, Z):
    xp = array_module(X)
    cx, cy, cz = xp.cos(X), xp.cos(Y), xp.cos(Z)
    return 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz


def fischer_koch_s(X, Y, Z):
    xp = array_module(X)
    return (xp.cos(2 * X) * xp.sin(Y) * xp.cos(Z) + xp.cos(X) * xp.cos(2 * Y) * xp.sin(Z)
            + xp.sin(X) * xp.cos(Y) * xp.cos(2 * Z))


SURFACES = {
    "gyroid": gyroid,
    "schwarz_p": schwarz_p,
    "primitive": schwarz_p,
    "diamond": diamond,
    "iwp": iwp,
    "neovius": neovius,
    "fks": fischer_koch_s,
}

_TABLE_CACHE: dict = {}


def _unit_cell_samples(surface: str, n: int = 48):
    f = SURFACES[surface]
    s = (np.arange(n) + 0.5) / n * TWO_PI
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    vals = f(X, Y, Z)
    gx, gy, gz = np.gradient(vals, s[1] - s[0])
    return vals.ravel(), np.sqrt(gx**2 + gy**2 + gz**2).ravel()


def density_table(surface: str, kind: str, n_levels: int = 201):
    """Relative density as a function of level ``c`` (monotone increasing)."""
    key = (surface, kind)
    if key not in _TABLE_CACHE:
        vals, grad = _unit_cell_samples(surface)
        fmax = np.abs(vals).max()
        if kind == "sheet":
            levels = np.linspace(0.0, fmax, n_levels)
            rho = np.array([(np.abs(vals) < c).mean() for c in levels])
        elif kind == "network":
            levels = np.linspace(-fmax, fmax, n_levels)
            rho = np.array([(vals < c).mean() for c in levels])
        elif kind == "network_upper":          # solid where f > c: store as a function of -c
            levels = np.linspace(-fmax, fmax, n_levels)
            rho = np.array([(vals > -c).mean() for c in levels])
        else:
            raise ValueError("kind must be 'sheet', 'network' or 'network_upper'")
        # typical |grad f| near the surface, used to turn f into ~distance
        near = np.abs(vals) < 0.15 * fmax
        g0 = float(np.mean(grad[near])) if near.any() else 1.0
        _TABLE_CACHE[key] = (levels, np.maximum.accumulate(rho), g0)
    return _TABLE_CACHE[key]


def level_for_density(surface: str, kind: str, rho):
    """Level ``c`` giving relative density ``rho`` (array-friendly, host side).

    For ``kind="network_upper"`` (solid where f > c) the returned value is the
    actual level ``c``."""
    levels, dens, _ = density_table(surface, kind)
    rho = np.clip(np.asarray(rho, float), dens[0] + 1e-6, dens[-1] - 1e-6)
    c = np.interp(rho, dens, levels)
    return -c if kind == "network_upper" else c


def relative_density(surface: str, kind: str, level: float) -> float:
    levels, dens, _ = density_table(surface, kind)
    x = -level if kind == "network_upper" else level
    return float(np.interp(x, levels, dens))


class TPMS(Field):
    """A TPMS lattice field.

    Parameters
    ----------
    surface : one of ``SURFACES`` (``"gyroid"``, ``"diamond"``, ``"schwarz_p"`` ...)
    cell_size : unit-cell edge length (scalar or (Lx, Ly, Lz)), model units (mm)
    kind : ``"sheet"`` or ``"network"``
    density : target relative density in (0, 1) -- scalar, or a callable
        ``rho(x, y, z)`` for a functionally graded lattice. Ignored if ``level``
        is given.
    level : explicit iso-level ``c`` (scalar or callable).
    origin, rotation : lattice placement/orientation (rotation is a 3x3 matrix).
    """

    def __init__(self, surface: str = "gyroid", cell_size=3.0, kind: str = "sheet",
                 density: Optional[Union[float, Callable]] = 0.3,
                 level: Optional[Union[float, Callable]] = None,
                 origin=(0.0, 0.0, 0.0), rotation=None, side: str = "lower"):
        if surface not in SURFACES:
            raise KeyError(f"unknown TPMS {surface!r}; choose from {sorted(SURFACES)}")
        if kind == "network" and side == "upper":
            kind = "network_upper"
        elif side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        self.surface = surface
        self.kind = kind
        L = np.broadcast_to(np.asarray(cell_size, float), (3,)).copy()
        self.cell_size = L
        self.origin = np.asarray(origin, float)
        self.R = None if rotation is None else np.asarray(rotation, float).T
        levels, dens, g0 = density_table(surface, kind)
        self._levels, self._dens = levels, dens
        # f -> approximately signed distance in model units
        self._scale = float(np.mean(L)) / (TWO_PI * g0)
        self.density = density
        self.level = level
        super().__init__(self._eval, f"{kind}-{surface}")

    def _level_at(self, x, y, z):
        xp = array_module(x)
        if self.level is not None:
            return self.level(x, y, z) if callable(self.level) else self.level
        if callable(self.density):
            rho = self.density(x, y, z)
            # np.interp equivalent on the active backend
            lv = xp.asarray(self._levels)
            dn = xp.asarray(self._dens)
            rho = xp.clip(rho, float(self._dens[0]) + 1e-6, float(self._dens[-1]) - 1e-6)
            c = xp.interp(rho, dn, lv)
            return -c if self.kind == "network_upper" else c
        return float(level_for_density(self.surface, self.kind, self.density))

    def _eval(self, x, y, z):
        xp = array_module(x)
        px, py, pz = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        if self.R is not None:
            R = self.R
            px, py, pz = (R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz,
                          R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz,
                          R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz)
        L = self.cell_size
        f = SURFACES[self.surface](TWO_PI * px / L[0], TWO_PI * py / L[1], TWO_PI * pz / L[2])
        c = self._level_at(x, y, z)
        if self.kind == "sheet":
            phi = xp.abs(f) - c
        elif self.kind == "network_upper":
            phi = c - f
        else:
            phi = f - c
        return phi * self._scale

    def nominal_relative_density(self) -> float:
        if self.level is not None and not callable(self.level):
            return relative_density(self.surface, self.kind, self.level)
        if callable(self.density):
            raise ValueError("graded lattice has no single nominal density")
        return float(self.density)
