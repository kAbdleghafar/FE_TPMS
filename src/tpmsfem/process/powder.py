"""Powder beds for the free-surface melt-pool model: sphere packing and voxelisation.

``rain_powder`` drops spheres one by one onto a floor (the plate or the part built so far)
and the spheres already placed. Each sphere tries several nearby landing points and keeps the
lowest resting position; that imitates rolling into pockets and gives a packing density of
about 0.5-0.6, like a spread powder layer. A recoater then removes every particle that
sticks out above the layer height.

``voxelise`` turns the spheres into cell fill levels (sub-sampled), for
:class:`tpmsfem.process.freesurface.FreeSurfaceMeltPool`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PowderBed:
    centres: np.ndarray        # (n, 3) [m]
    radii: np.ndarray          # (n,) [m]
    floor: float               # z of the floor [m]
    layer: float               # nominal layer thickness [m]
    extent: tuple              # (x0, x1, y0, y1) [m]

    @property
    def packing_density(self):
        """Particle volume / layer volume (floor to floor + layer)."""
        x0, x1, y0, y1 = self.extent
        return float((4.0 / 3.0 * np.pi * self.radii ** 3).sum() / ((x1 - x0) * (y1 - y0) * self.layer))

    def __len__(self):
        return len(self.radii)


def particle_sizes(n, d50=30e-6, sigma_ln=0.25, d_min=15e-6, d_max=45e-6, rng=None):
    """Diameters [m] from a log-normal distribution (median ``d50``, log standard deviation
    ``sigma_ln``), clipped to [d_min, d_max] by redrawing."""
    rng = np.random.default_rng(rng)
    out = np.empty(0)
    while out.size < n:
        d = d50 * np.exp(sigma_ln * rng.standard_normal(2 * n))
        out = np.r_[out, d[(d >= d_min) & (d <= d_max)]]
    return out[:n]


def rain_powder(extent, floor, layer, d50=30e-6, sigma_ln=0.25, d_min=15e-6, d_max=45e-6, seed=0,
                n_try=4, reach=1.0, roll=True, floor_height=None, max_particles=200000, patience=400):
    """Spread one powder layer over ``extent = (x0, x1, y0, y1)`` [m] on a floor at height
    ``floor`` [m] (or a floor-height function ``floor_height(x, y)``), with a recoater at
    ``floor + layer``. Returns a :class:`PowderBed`."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = extent
    top = floor + layer
    cell = d_max
    nbx, nby = int(np.ceil((x1 - x0) / cell)) + 1, int(np.ceil((y1 - y0) / cell)) + 1
    grid = [[[] for _ in range(nby)] for _ in range(nbx)]
    C = np.zeros((max_particles, 3))
    R = np.zeros(max_particles)
    m = 0

    def rest_height(x, y, r):
        z = (floor_height(x, y) if floor_height is not None else floor) + r
        ix, iy = int((x - x0) / cell), int((y - y0) / cell)
        for jx in range(max(ix - 1, 0), min(ix + 2, nbx)):
            for jy in range(max(iy - 1, 0), min(iy + 2, nby)):
                for k in grid[jx][jy]:
                    dh2 = (C[k, 0] - x) ** 2 + (C[k, 1] - y) ** 2
                    s = R[k] + r
                    if dh2 < s * s:
                        z = max(z, C[k, 2] + np.sqrt(s * s - dh2))
        return z

    # particles land one at a time; one that would stick out above the recoater is swept
    # away (not placed), and spreading stops after ``patience`` such particles in a row
    dirs = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    sizes = iter(())
    misses = 0
    while m < max_particles and misses < patience:
        try:
            r = 0.5 * next(sizes)
        except StopIteration:
            sizes = iter(particle_sizes(1000, d50, sigma_ln, d_min, d_max, rng))
            continue
        xs = rng.uniform(x0 + r, x1 - r)
        ys = rng.uniform(y0 + r, y1 - r)
        best = None
        for _ in range(n_try):
            x = float(np.clip(xs + rng.uniform(-reach, reach) * r, x0 + r, x1 - r))
            y = float(np.clip(ys + rng.uniform(-reach, reach) * r, y0 + r, y1 - r))
            z = rest_height(x, y, r)
            if best is None or z < best[2]:
                best = (x, y, z)
        x, y, z = best
        if roll:                                       # roll downhill into a pocket
            step = 0.3 * r
            while step > 0.02 * r and z + r > (floor if floor_height is None else floor_height(x, y)) + r + 1e-12:
                moved = False
                for ang in dirs:
                    xn = float(np.clip(x + step * np.cos(ang), x0 + r, x1 - r))
                    yn = float(np.clip(y + step * np.sin(ang), y0 + r, y1 - r))
                    zn = rest_height(xn, yn, r)
                    if zn < z - 1e-12:
                        x, y, z, moved = xn, yn, zn, True
                        break
                if not moved:
                    step *= 0.5
        if z + r > top + 1e-12:
            misses += 1
            continue
        misses = 0
        C[m] = (x, y, z)
        R[m] = r
        grid[int((x - x0) / cell)][int((y - y0) / cell)].append(m)
        m += 1
    return PowderBed(C[:m].copy(), R[:m].copy(), floor, layer, extent)


def voxelise(bed: PowderBed, dx, shape, origin=(0.0, 0.0, 0.0), sub: int = 3):
    """Fill level of each cell (``shape``, cell size ``dx``, cell (0,0,0) starting at
    ``origin``) covered by the particles, from ``sub**3`` sample points per cell."""
    fill = np.zeros(shape)
    o = np.asarray(origin, float)
    s = (np.arange(sub) + 0.5) / sub
    for c, r in zip(bed.centres, bed.radii):
        lo = np.maximum(np.floor((c - r - o) / dx).astype(int), 0)
        hi = np.minimum(np.ceil((c + r - o) / dx).astype(int) + 1, shape)
        if np.any(hi <= lo):
            continue
        ax = [o[a] + (np.arange(lo[a], hi[a])[:, None] + s[None, :]).ravel() * dx for a in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        inside = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= r * r
        n = [hi[a] - lo[a] for a in range(3)]
        frac = inside.reshape(n[0], sub, n[1], sub, n[2], sub).mean(axis=(1, 3, 5))
        fill[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += frac
    return np.minimum(fill, 1.0)
