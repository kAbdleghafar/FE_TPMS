"""As-built surface topography of LPBF parts: staircase, melt-pool rounding, adhered powder.

A surface element is characterised by the angle ``theta`` between its outward normal and the
build direction (+z): 0 = up-facing (upskin) flat, 90 = vertical wall, 180 = down-facing
(downskin) flat. Its as-built topography is the superposition of

1. **the layer staircase** -- layers of thickness ``t`` leave a sawtooth of peak-to-valley
   ``t |cos(theta)|`` perpendicular to the mean surface, with period ``t / |sin(theta)|``
   along it (Ra = t |cos theta| / 4 for a sharp sawtooth);
2. **melt-pool rounding** -- each layer edge is the solidified melt-pool boundary, so the
   convex step corners are rounded with the bead radius (default: half the melt-pool width,
   e.g. from :func:`tpmsfem.process.meltpool.fusion_zone`); implemented as a morphological
   closing of the profile with a disk;
3. **partially fused powder** -- spheres from the powder size distribution stuck to the
   surface as a Poisson process, embedded to a fraction of their diameter, with an areal
   density that grows on down-facing surfaces (melt pool resting on powder) and drops on
   up-facing ones (``density(theta) = rho_side * (1 + k_down max(0, -cos) - k_up max(0, cos))``).

Roughness parameters follow ISO 4287 / 25178 definitions on the raw (form-removed,
unfiltered) profile or height map: Ra, Rq, Rz (mean of five peak-to-valley sections), Sa,
Sq, Sz. The particle parameters are material/machine specific and must be calibrated on
measured roughness; the defaults are only placeholders.

Units: mm (heights reported in um).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PowderPSD:
    d10: float = 0.015
    d50: float = 0.030
    d90: float = 0.045

    def sample(self, n, rng):
        s = np.log(self.d90 / self.d10) / (2.0 * 1.2815516)
        return np.exp(rng.normal(np.log(self.d50), s, size=n))


@dataclass(frozen=True)
class SurfaceModel:
    layer: float = 0.02                 # layer thickness [mm]
    bead_radius: float = 0.06           # melt-pool half width [mm]
    psd: PowderPSD = PowderPSD()
    rho_side: float = 0.0               # adhered particles per mm^2 on a vertical wall (calibrate!)
    k_down: float = 2.0
    k_up: float = 1.0
    embed: float = 0.4                  # embedded fraction of the particle diameter
    rho_ref: float = 2000.0             # candidate density for Poisson thinning [1/mm^2]

    def particle_density(self, theta_deg):
        c = np.cos(np.radians(theta_deg))
        return self.rho_side * np.maximum(1.0 + self.k_down * np.maximum(-c, 0.0) - self.k_up * np.maximum(c, 0.0), 0.0)

    # --------------------------------------------------------------
    def staircase(self, theta_deg, s):
        """Sawtooth height [mm] (perpendicular to the mean surface, zero mean) at positions s."""
        th = np.radians(theta_deg)
        sn, cs = abs(np.sin(th)), abs(np.cos(th))
        if cs < 1e-9:
            return np.zeros_like(s)
        if sn < 1e-9:
            return np.zeros_like(s)            # flat: one layer, no staircase
        period = self.layer / sn
        amp = self.layer * cs
        frac = np.mod(s, period) / period
        return amp * (frac - 0.5)

    def surface(self, theta_deg, size=(1.0, 1.0), dx=0.002, seed=0, rounding=True, particles=True):
        """Height map z(s, u) [mm] of a ``size`` patch (s: across layers, u: along them)."""
        rng = np.random.default_rng(seed)
        ns, nu = int(round(size[0] / dx)), int(round(size[1] / dx))
        s = (np.arange(ns) + 0.5) * dx
        u = (np.arange(nu) + 0.5) * dx
        prof = self.staircase(theta_deg, s)
        if rounding and self.bead_radius > 0:
            prof = _close_with_disk(prof, dx, self.bead_radius)
        Z = np.repeat(prof[:, None], nu, axis=1)
        if particles:
            # Poisson thinning of one candidate set at density rho_ref: for a fixed seed a higher
            # density only adds particles, so roughness grows monotonically (calibration)
            rho = self.particle_density(theta_deg)
            rho_ref = max(self.rho_ref, rho)
            n = rng.poisson(rho_ref * size[0] * size[1])
            d = self.psd.sample(n, rng)
            cs_, cu_ = rng.uniform(0, size[0], n), rng.uniform(0, size[1], n)
            keep = rng.uniform(0.0, 1.0, n) < rho / rho_ref
            if keep.any():
                for di, si, ui in zip(d[keep], cs_[keep], cu_[keep]):
                        r = di / 2.0
                        i0, i1 = max(0, int((si - r) / dx)), min(ns, int((si + r) / dx) + 2)
                        j0, j1 = max(0, int((ui - r) / dx)), min(nu, int((ui + r) / dx) + 2)
                        if i0 >= i1 or j0 >= j1:
                            continue
                        ss, uu = np.meshgrid(s[i0:i1], u[j0:j1], indexing="ij")
                        q = r * r - (ss - si) ** 2 - (uu - ui) ** 2
                        base = Z[i0:i1, j0:j1].max()
                        top = base - self.embed * di + r + np.sqrt(np.maximum(q, 0.0))
                        Z[i0:i1, j0:j1] = np.where(q > 0, np.maximum(Z[i0:i1, j0:j1], top), Z[i0:i1, j0:j1])
        return s, u, Z

    def roughness(self, theta_deg, size=(1.0, 1.0), dx=0.002, seed=0, **kw):
        s, u, Z = self.surface(theta_deg, size, dx, seed, **kw)
        return surface_parameters(Z, s)


def _close_with_disk(prof, dx, r):
    """Grey-scale closing (dilation then erosion) of a 1-D profile with a disk of radius r
    from above: fills the re-entrant corners of the sawtooth, rounding them with radius r."""
    k = int(np.ceil(r / dx))
    off = np.arange(-k, k + 1) * dx
    cap = np.sqrt(np.maximum(r * r - off ** 2, 0.0)) - r            # <= 0, disk profile
    n = len(prof)
    pad = np.pad(prof, k, mode="edge")
    # dilation: max over the disk of (profile + cap)
    dil = np.max(np.stack([pad[k + o: k + o + n] + cap[i] for i, o in enumerate(range(-k, k + 1))]), axis=0)
    padd = np.pad(dil, k, mode="edge")
    ero = np.min(np.stack([padd[k + o: k + o + n] - cap[i] for i, o in enumerate(range(-k, k + 1))]), axis=0)
    return ero


def profile_parameters(z):
    """Ra, Rq, Rz [um] of a 1-D profile [mm] after removing its least-squares line."""
    z = np.asarray(z, float)
    x = np.arange(len(z))
    z = z - np.polyval(np.polyfit(x, z, 1), x)
    seg = np.array_split(z, 5)
    rz = np.mean([q.max() - q.min() for q in seg])
    return dict(Ra=1e3 * np.mean(np.abs(z)), Rq=1e3 * np.sqrt(np.mean(z * z)), Rz=1e3 * rz)


def surface_parameters(Z, s=None):
    """Sa, Sq, Sz [um] of a height map (least-squares plane removed) and the mean Ra/Rq/Rz
    of its profiles across the layers (axis 0)."""
    ns, nu = Z.shape
    I, J = np.meshgrid(np.arange(ns), np.arange(nu), indexing="ij")
    A = np.column_stack([I.ravel(), J.ravel(), np.ones(I.size)])
    coef, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    R = Z - (A @ coef).reshape(Z.shape)
    prof = [profile_parameters(Z[:, j]) for j in range(0, nu, max(1, nu // 25))]
    return dict(Sa=1e3 * np.mean(np.abs(R)), Sq=1e3 * np.sqrt(np.mean(R * R)), Sz=1e3 * (R.max() - R.min()),
                Ra=float(np.mean([p["Ra"] for p in prof])), Rq=float(np.mean([p["Rq"] for p in prof])),
                Rz=float(np.mean([p["Rz"] for p in prof])))


def calibrate_particle_density(model: SurfaceModel, theta_deg: float, Ra_target: float, **kw):
    """Vertical-wall particle density giving a measured Ra [um] at ``theta_deg`` (bisection,
    Ra increases monotonically with the density)."""
    from dataclasses import replace

    lo, hi = 0.0, model.rho_ref

    def ra(rho):
        return replace(model, rho_side=rho).roughness(theta_deg, **kw)["Ra"]
    if ra(hi) < Ra_target:
        raise ValueError(f"target roughness not reachable with <= {model.rho_ref} particles/mm^2")
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ra(mid) < Ra_target else (lo, mid)
    return replace(model, rho_side=0.5 * (lo + hi))


def inclination(normals, build_axis=(0.0, 0.0, 1.0)):
    """theta [deg] between unit normals and the build axis."""
    n = np.asarray(normals, float)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-300)
    return np.degrees(np.arccos(np.clip(n @ np.asarray(build_axis, float), -1.0, 1.0)))


def roughness_map(normals, model: SurfaceModel, thetas=np.arange(0, 181, 10), key="Sa", **kw):
    """Per-facet roughness [um] for surface normals (e.g. of a TPMS triangulation), by
    interpolating the model over a table of inclinations."""
    table = np.array([model.roughness(t, **kw)[key] for t in thetas])
    return np.interp(inclination(normals), thetas, table), (np.asarray(thetas), table)
