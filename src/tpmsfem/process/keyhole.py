"""Keyhole physics for the free-surface melt pool (:class:`tpmsfem.process.freesurface.FreeSurfaceMeltPool`).

* **Laser ray tracing** (:class:`RayTracer`). A Gaussian beam is cut into rays that come
  straight down. Each ray marches through the smoothed fill-level field until it meets the
  metal surface (fill = 1/2, located by bisection). There it deposits the Fresnel-absorbed
  share of its power in the metal cell behind the surface, and reflects specularly about the
  surface normal (the fill gradient). It bounces until it leaves the domain or its power
  drops below a threshold. In a deep keyhole the rays bounce several times, so the absorbed
  fraction rises from the flat-melt value towards ~0.7: the effect Gan et al. (2021) express
  by their Eq. 6. Polarisation is averaged (fibre lasers).
* **Fresnel absorption** (:func:`fresnel_absorptivity`). For a metal of complex refractive
  index n + ik. Optical constants of liquid alloys at 1070 nm are poorly known, so
  :func:`index_from_absorptivity` picks n and k that give a chosen normal-incidence
  absorptivity A0 (the flat-melt value) for a given k/n.
* **Bubbles** (:class:`BubbleModel`). Gas regions are labelled (18-neighbour connectivity, as
  the D3Q19 links). The region touching the top of the domain is the ambient gas: the open
  keyhole and the atmosphere, at ambient pressure. Every other region is a bubble with its own
  gas content, fixed when it pinches off (ambient pressure times its volume then), and an
  ideal-gas pressure p = p_amb (V0 / V)^gamma. Bubbles are followed between updates by their
  overlap: merging bubbles add their gas, a splitting bubble shares it by volume, a bubble
  that reopens to the ambient gas releases it. The pressure difference to ambient acts on the
  interface cells around each bubble, so a collapsing keyhole tip leaves a gas pore that
  resists closing and can be frozen in. Each bubble keeps the hottest wall temperature it has
  seen (``T_wall_hist``); ``pore`` marks a frozen bubble whose walls were molten at some time:
  a pore made by the melt, not a void sealed between cold powder particles.
* **Diagnostics** (:func:`keyhole_metrics`): depression depth below the original surface,
  front-wall angle and opening on the beam axis, and the bubbles (pores) with their sizes.

Not modelled: the vapour plume itself (only its recoil pressure on the wall), plasma
absorption, beam divergence (rays are vertical; the Rayleigh length of LPBF beams is mm, far
longer than a keyhole) and the condensation of metal vapour in a closed bubble (its gas is
taken as ambient-pressure gas at pinch-off).
"""
from __future__ import annotations

import math

import numpy as np

from ..backend import asnumpy, get_xp

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:                                  # pragma: no cover
    HAVE_NUMBA = False


# ====================================================================== Fresnel
def fresnel_absorptivity(cos_theta, n, k):
    """Absorptivity 1 - (R_s + R_p) / 2 of unpolarised light on a metal of complex refractive
    index N = n + ik, at incidence cos(theta) (theta from the surface normal)."""
    c = np.clip(np.asarray(cos_theta, float), 0.0, 1.0)
    N2 = complex(n, k) ** 2
    w = np.sqrt(N2 - (1.0 - c * c) + 0j)
    rs = (c - w) / (c + w)
    rp = (N2 * c - w) / (N2 * c + w)
    return 1.0 - 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


def index_from_absorptivity(A0, k_over_n=1.0):
    """(n, k) of a metal whose normal-incidence absorptivity 4n / ((n+1)^2 + k^2) is ``A0``,
    with k = ``k_over_n`` n (the metallic, larger root). k = n is the good-conductor limit
    behind Bramson's formula, the usual choice in keyhole models; a smaller k/n raises the
    grazing-incidence maximum (A0 = 0.26: 0.43 at 83 deg for k = n, 0.49 for k/n = 0.5, 0.34
    for 2). A0 is the flat-melt absorptivity (Gan et al.: Ti-6Al-4V 0.26, SS316 0.33)."""
    r2 = 1.0 + k_over_n ** 2
    a, b, c = A0 * r2, 2.0 * A0 - 4.0, A0
    n = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return n, k_over_n * n


# ====================================================================== ray tracing kernels
if HAVE_NUMBA:
    @nb.njit(cache=True, inline="always")
    def _sample(f, x, y, z):
        """Trilinear value of a cell-centred field at (x, y, z) in index units (cell i spans
        [i, i + 1))."""
        nx, ny, nz = f.shape
        qx, qy, qz = x - 0.5, y - 0.5, z - 0.5
        i = min(max(int(math.floor(qx)), 0), nx - 2)
        j = min(max(int(math.floor(qy)), 0), ny - 2)
        k = min(max(int(math.floor(qz)), 0), nz - 2)
        tx = min(max(qx - i, 0.0), 1.0)
        ty = min(max(qy - j, 0.0), 1.0)
        tz = min(max(qz - k, 0.0), 1.0)
        c00 = f[i, j, k] * (1 - tx) + f[i + 1, j, k] * tx
        c10 = f[i, j + 1, k] * (1 - tx) + f[i + 1, j + 1, k] * tx
        c01 = f[i, j, k + 1] * (1 - tx) + f[i + 1, j, k + 1] * tx
        c11 = f[i, j + 1, k + 1] * (1 - tx) + f[i + 1, j + 1, k + 1] * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @nb.njit(cache=True)
    def _deposit_cell(f, metal, x, y, z, nix, niy, niz):
        """Flat index of the metal cell that takes the absorbed power of a hit at (x, y, z):
        half a cell into the metal along the normal, else the cell of the hit point, else its
        fullest metal neighbour; -1 if there is none."""
        nx, ny, nz = f.shape
        for s in (0.5, 0.0):
            i, j, k = int(math.floor(x + s * nix)), int(math.floor(y + s * niy)), int(math.floor(z + s * niz))
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and metal[i, j, k]:
                return (i * ny + j) * nz + k
        i0, j0, k0 = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        best, bi = -1.0, -1
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and metal[i, j, k] and f[i, j, k] > best:
                        best = f[i, j, k]
                        bi = (i * ny + j) * nz + k
        return bi

    @nb.njit(parallel=True, cache=True)
    def _trace(f, metal, xs, ys, ps, z0, n_re, n_im, step, max_b, min_frac, hit_cell, hit_e, hit_cos, fate, left):
        """March every ray from height z0 (above all metal) down; fate: 0 escaped upwards
        (above z0), 1 through a side, 2 power threshold or bounce limit reached, 3 hit the
        bottom."""
        nx, ny, nz = f.shape
        N2 = complex(n_re, n_im) ** 2
        max_steps = int(4.0 * (nx + ny + nz) / step) + 10
        for r in nb.prange(xs.shape[0]):
            x, y, z = xs[r], ys[r], z0
            dx, dy, dz = 0.0, 0.0, -1.0
            P = ps[r]
            P0 = P
            nb_ = 0
            fate[r] = 2
            while True:
                hit = False
                px, py, pz = x, y, z
                out = -1
                for _ in range(max_steps):
                    x += dx * step
                    y += dy * step
                    z += dz * step
                    if z > z0:
                        out = 0
                        break
                    if x < 1.0 or x >= nx - 1.0 or y < 1.0 or y >= ny - 1.0:
                        out = 1
                        break
                    if z < 1.0:
                        out = 3
                        break
                    if _sample(f, x, y, z) >= 0.5:
                        hit = True
                        break
                    px, py, pz = x, y, z
                if not hit:
                    fate[r] = out if out >= 0 else 1
                    left[r] = P
                    break
                # bisection between the last gas point (p) and the first metal point
                ax, ay, az = px, py, pz
                bx, by, bz = x, y, z
                for _ in range(8):
                    mx, my, mz = 0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)
                    if _sample(f, mx, my, mz) >= 0.5:
                        bx, by, bz = mx, my, mz
                    else:
                        ax, ay, az = mx, my, mz
                hx, hy, hz = 0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz)
                # normal into the metal: the gradient of the fill level
                h = 0.5
                gx = _sample(f, hx + h, hy, hz) - _sample(f, hx - h, hy, hz)
                gy = _sample(f, hx, hy + h, hz) - _sample(f, hx, hy - h, hz)
                gz = _sample(f, hx, hy, hz + h) - _sample(f, hx, hy, hz - h)
                gn = math.sqrt(gx * gx + gy * gy + gz * gz)
                if gn < 1e-12:
                    gx, gy, gz, gn = dx, dy, dz, 1.0
                nix, niy, niz = gx / gn, gy / gn, gz / gn
                c = dx * nix + dy * niy + dz * niz             # > 0: the ray runs into the metal
                if c < 1e-3:                                   # grazing, or a noisy normal
                    c = 1e-3
                c = min(c, 1.0)
                w = (N2 - (1.0 - c * c)) ** 0.5
                rs = (c - w) / (c + w)
                rp = (N2 * c - w) / (N2 * c + w)
                A = 1.0 - 0.5 * (abs(rs) ** 2 + abs(rp) ** 2)
                E = P * A
                hit_cell[r, nb_] = _deposit_cell(f, metal, hx, hy, hz, nix, niy, niz)
                hit_e[r, nb_] = E
                hit_cos[r, nb_] = c
                P -= E
                nb_ += 1
                if P < min_frac * P0 or nb_ >= max_b:
                    left[r] = P
                    fate[r] = 2
                    break
                # specular reflection, restarting from the gas side
                dn = dx * nix + dy * niy + dz * niz
                dx, dy, dz = dx - 2.0 * dn * nix, dy - 2.0 * dn * niy, dz - 2.0 * dn * niz
                dl = math.sqrt(dx * dx + dy * dy + dz * dz)
                dx, dy, dz = dx / dl, dy / dl, dz / dl
                x, y, z = ax - 0.05 * nix, ay - 0.05 * niy, az - 0.05 * niz


def smooth_fill(fill, passes=1):
    """[1 2 1] smoothing per axis (mirror at the array ends), as the solver applies before
    tracing (one pass puts the surface normal within ~1 deg on an inclined plane)."""
    e = np.asarray(fill, float)
    for _ in range(passes):
        for ax in range(3):
            n = e.shape[ax]
            lo = np.take(e, np.r_[0, np.arange(n - 1)], axis=ax)
            hi = np.take(e, np.r_[np.arange(1, n), n - 1], axis=ax)
            e = 0.25 * (lo + 2.0 * e + hi)
    return e


class RayTracer:
    """Laser absorption by ray tracing with Fresnel absorption and multiple reflections.

    ``A0`` (normal-incidence absorptivity of the flat melt) with ``k_over_n``, or the optical
    constants ``n``, ``k`` directly. ``spacing``: distance between rays in the beam plane
    [cells]; 0.25 converges the absorbed fraction of a sharp-tipped cavity to ~1 % (0.5: up
    to 0.06 off); ``cutoff``: beam radius traced, in 1/e^2 radii (2: 99.97 % of the power, kept
    exact by renormalising). ``max_bounces`` and ``min_power`` (fraction of a ray's power)
    end a ray; ``step``: marching step [cells]. ``box``: half-width of the box traced around
    the beam, in 1/e^2 radii (rays leaving it sideways are lost and counted)."""

    def __init__(self, A0=None, n=None, k=None, k_over_n=1.0, spacing=0.25, cutoff=2.0, max_bounces=40,
                 min_power=1e-3, step=0.25, box=4.0):
        if not HAVE_NUMBA:                         # pragma: no cover
            raise RuntimeError("RayTracer needs numba")
        if n is None or k is None:
            if A0 is None:
                raise ValueError("give A0 (flat-melt absorptivity) or the optical constants n, k")
            n, k = index_from_absorptivity(A0, k_over_n)
        self.n, self.k = float(n), float(k)
        self.spacing, self.cutoff = float(spacing), float(cutoff)
        self.max_bounces, self.min_power, self.step = int(max_bounces), float(min_power), float(step)
        self.box = float(box)

    @property
    def A0(self):
        return float(fresnel_absorptivity(1.0, self.n, self.k))

    def rays(self, xc, yc, power, radius):
        """Ray positions [cells] and powers [W] of a Gaussian beam (1/e^2 radius ``radius``
        in cells) centred at (xc, yc)."""
        s = self.spacing
        m = int(np.ceil(self.cutoff * radius / s))
        g = (np.arange(-m, m) + 0.5) * s           # symmetric about the axis, off the cell faces
        X, Y = np.meshgrid(xc + g, yc + g, indexing="ij")
        r2 = (X - xc) ** 2 + (Y - yc) ** 2
        keep = r2 <= (self.cutoff * radius) ** 2
        w = np.exp(-2.0 * r2[keep] / radius ** 2)
        return X[keep].ravel(), Y[keep].ravel(), power * w / w.sum()

    def trace(self, fill, metal, xc, yc, power, radius):
        """Trace the beam over a host array of (smoothed) fill levels ``fill`` with metal mask
        ``metal`` (both nx, ny, nz; index units: cell i spans [i, i + 1)). Returns the power
        absorbed per cell [W] and statistics: absorbed fraction, fractions escaped through
        the top and the sides, left when rays were stopped, not deposited (no metal cell
        found), mean number of bounces, and the power-weighted mean cosine of the first hit."""
        fill = np.ascontiguousarray(fill, dtype=np.float64)
        metal = np.ascontiguousarray(metal, dtype=np.bool_)
        xs, ys, ps = self.rays(xc, yc, power, radius)
        nr, mb = xs.size, self.max_bounces
        hit_cell = np.full((nr, mb), -1, np.int64)
        hit_e = np.zeros((nr, mb))
        hit_cos = np.zeros((nr, mb))
        fate = np.zeros(nr, np.int64)
        left = np.zeros(nr)
        any_m = np.nonzero((fill > 1e-6).any(axis=(0, 1)))[0]
        z0 = min(float(any_m.max()) + 2.0 if any_m.size else 1.0, float(fill.shape[2]) - 1.0 - 1e-6)
        _trace(fill, metal, xs.astype(np.float64), ys.astype(np.float64), ps.astype(np.float64),
               z0, self.n, self.k, self.step, mb, self.min_power, hit_cell, hit_e, hit_cos, fate, left)
        dep = np.zeros(fill.size)
        ok = hit_cell >= 0
        np.add.at(dep, hit_cell[ok], hit_e[ok])
        P = float(ps.sum())
        bounces = (hit_e > 0).sum(axis=1)
        first = hit_e[:, 0] > 0
        stats = dict(absorbed=float(dep.sum()) / P,
                     escaped_top=float(left[fate == 0].sum()) / P,
                     escaped_side=float(left[fate == 1].sum()) / P,
                     stopped=float(left[fate == 2].sum()) / P,
                     not_deposited=float(hit_e[~ok].sum()) / P,
                     bottom=float(left[fate == 3].sum()) / P,
                     mean_bounces=float((bounces * ps).sum() / P),
                     first_cos=float((hit_cos[first, 0] * ps[first]).sum() / max(ps[first].sum(), 1e-30)),
                     rays=int(nr))
        return dep.reshape(fill.shape), stats


# ====================================================================== bubbles
def _ndimage(xp):
    if xp is np:
        import scipy.ndimage as ndi
    else:                                          # pragma: no cover - GPU
        import cupyx.scipy.ndimage as ndi
    return ndi


class BubbleModel:
    """Ideal-gas pressure in enclosed gas regions (bubbles, pores) of a free-surface lattice.

    Call :meth:`update` with the lattice flags and fill levels every few steps; it returns the
    pressure difference to ambient [Pa] on every interface cell next to a bubble (zero
    elsewhere) and keeps the list of bubbles in ``self.bubbles``. ``gamma``: 1 isothermal, 1.4
    adiabatic (diatomic gas). ``min_cells``: smaller gas regions are ignored (dp = 0): single
    empty cells in the melt are lattice noise, not resolvable pores. ``open_sides``: gas that
    reaches the side faces of the domain is ambient too, e.g. the gas in a powder bed, which
    connects to the chamber through the bed around the domain."""

    def __init__(self, p_ambient=101325.0, gamma=1.0, min_cells=2, open_sides=False):
        self.p_amb = float(p_ambient)
        self.open_sides = bool(open_sides)     # gas reaching the side faces is ambient too (a powder bed)
        self.gamma = float(gamma)
        self.min_cells = int(min_cells)
        self.ids = None               # per-cell bubble id of the last update (0: none / ambient)
        self.content = {}             # id -> gas content, as the volume [cells] it has at ambient pressure
        self.next_id = 1
        self.bubbles = []             # list of dicts: id, volume [cells], V0, pressure [Pa], centroid [cells]
        self.released = 0             # bubbles that reopened to the ambient gas
        self.formed = 0
        self.hot = {}                 # id -> hottest metal ever seen around it (inherited through merges/splits)

    def update(self, flag, fill, T=None, T_solidus=None):
        """Relabel the gas, follow the bubbles and return the pressure difference to ambient on
        the interface cells around them. With the temperature ``T`` and ``T_solidus``, every
        bubble also gets the hottest temperature of the metal around it (``T_wall_max``) and
        ``frozen`` = all of it below the solidus: a pore trapped by solidification."""
        from .freesurface import C, GAS, INTERFACE
        xp = get_xp()
        ndi = _ndimage(xp)
        flag, fill = xp.asarray(flag), xp.asarray(fill)
        if T is not None:
            T = xp.asarray(T)
        gas = flag == GAS
        g = np.indices((3, 3, 3)) - 1
        struct = xp.asarray((np.abs(g).sum(axis=0) <= 2).astype(np.int32))     # the 18 D3Q19 neighbours
        lab, nlab = ndi.label(gas, structure=struct)
        nlab = int(nlab)
        if nlab == 0:
            self.ids, self.bubbles = xp.zeros(flag.shape, np.int64), []
            return xp.zeros(flag.shape, dtype=fill.dtype)
        lab = lab.astype(np.int64)
        # the ambient gas: every region that reaches the top gas layer (below the top wall)
        amb = xp.zeros(nlab + 1, dtype=bool)
        top = lab[:, :, -2].ravel()
        amb[top] = True
        if self.open_sides:                    # the cells next to the side walls
            for face in (lab[1], lab[-2], lab[:, 1], lab[:, -2]):
                amb[face.ravel()] = True
        amb[0] = True
        # volume: gas cells plus the empty part of the interface cells next to them
        itf = flag == INTERFACE
        nb_lab = xp.zeros(flag.shape, np.int64)
        for c in C[1:]:
            nb_lab = xp.maximum(nb_lab, xp.roll(lab, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2)))
        nb_lab = xp.where(itf, nb_lab, 0)
        vol = xp.bincount(lab.ravel(), minlength=nlab + 1).astype(float)
        vol = vol + xp.bincount(nb_lab.ravel(), weights=xp.where(itf, 1.0 - fill, 0.0).ravel(), minlength=nlab + 1)
        ncell = xp.bincount(lab.ravel(), minlength=nlab + 1)
        # follow the bubbles: overlap of the new regions with the old bubble ids
        content_new = np.zeros(nlab + 1)
        amb_h = asnumpy(amb)
        vol_h = asnumpy(vol)
        ncell_h = asnumpy(ncell)
        if self.ids is not None:
            sel = gas & (self.ids > 0)
            key = lab[sel] * (self.next_id + 1) + self.ids[sel]
            uk, cnt = xp.unique(key, return_counts=True)
            uk, cnt = asnumpy(uk), asnumpy(cnt)
            L, B = uk // (self.next_id + 1), uk % (self.next_id + 1)
            tot = {}
            for b, n in zip(B, cnt):
                tot[b] = tot.get(b, 0) + n
            seen = set()
            hot_new = np.full(nlab + 1, -np.inf)
            for l_, b, n in zip(L, B, cnt):
                if b in self.content:
                    content_new[l_] += self.content[b] * n / tot[b]
                    hot_new[l_] = max(hot_new[l_], self.hot.get(int(b), -np.inf))
                    seen.add(b)
                    if amb_h[l_]:
                        self.released += 1
        else:
            hot_new = np.full(nlab + 1, -np.inf)
        # new bubbles (no gas from an old one): born at ambient pressure
        enclosed = ~amb_h & (ncell_h >= self.min_cells)
        born = enclosed & (content_new <= 0.0)
        content_new[born] = vol_h[born]
        self.formed += int(born.sum())
        # pressure per region
        p = np.full(nlab + 1, self.p_amb)
        V = np.maximum(vol_h, 1e-9)
        p[enclosed] = self.p_amb * (content_new[enclosed] / V[enclosed]) ** self.gamma
        dp_reg = np.where(enclosed, p - self.p_amb, 0.0)
        # ids for the next update
        ids_reg = np.zeros(nlab + 1, np.int64)
        idx = np.nonzero(enclosed)[0]
        ids_reg[idx] = self.next_id + np.arange(idx.size)
        self.content = {int(ids_reg[i]): float(content_new[i]) for i in idx}
        self.next_id += int(idx.size)
        self.ids = xp.asarray(ids_reg)[lab]
        # hottest metal around each bubble (interface cells next to it): frozen in or not
        T_wall = np.full(nlab + 1, np.nan)
        if T is not None and idx.size:
            sel = xp.nonzero(nb_lab > 0)
            lab_w, T_w = asnumpy(nb_lab[sel]), asnumpy(T[sel])
            T_wall = np.full(nlab + 1, -np.inf)
            np.maximum.at(T_wall, lab_w, T_w)
            hot_new = np.maximum(hot_new, np.where(np.isfinite(T_wall), T_wall, -np.inf))
        self.hot = {int(ids_reg[i]): float(hot_new[i]) for i in idx}
        # bubble list with centroids
        self.bubbles = []
        if idx.size:
            coords = xp.indices(flag.shape)
            lr = lab.ravel()
            cen = [asnumpy(xp.bincount(lr, weights=coords[a].ravel().astype(float), minlength=nlab + 1)) for a in range(3)]
            for i in idx:
                tw = float(T_wall[i])
                th = float(hot_new[i])
                frozen = bool(T_solidus is not None and np.isfinite(tw) and tw < T_solidus)
                self.bubbles.append(dict(id=int(ids_reg[i]), volume=float(vol_h[i]), V0=float(content_new[i]),
                                         pressure=float(p[i]), cells=int(ncell_h[i]),
                                         centroid=[float(cen[a][i] / max(ncell_h[i], 1)) + 0.5 for a in range(3)],
                                         T_wall_max=tw, T_wall_hist=th, frozen=frozen,
                                         # a pore: frozen in after its walls were molten (not a sealed void
                                         # between cold powder particles)
                                         pore=bool(frozen and T_solidus is not None and th >= T_solidus)))
        # pressure difference on the interface cells around each bubble
        dpr = xp.asarray(dp_reg)
        dp_gas = dpr[lab]
        hi = xp.zeros(flag.shape, dtype=fill.dtype)
        lo = xp.zeros(flag.shape, dtype=fill.dtype)
        for c in C[1:]:
            v = xp.roll(dp_gas, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2))
            hi = xp.maximum(hi, v)
            lo = xp.minimum(lo, v)
        dp = xp.where(-lo > hi, lo, hi)
        return xp.where(itf, dp, 0.0)


# ====================================================================== diagnostics
def surface_heights(fill, dx):
    """Height [m] of the metal surface seen from above, per column: where the fill level (as
    given, e.g. smoothed once) first reaches 1/2 coming down, interpolated between cells.
    Low-fill specks above the surface (below 1/2) are not counted. Columns without metal: 0."""
    e = np.asarray(fill, float)
    nz = e.shape[2]
    has = e >= 0.5
    any_m = has.any(axis=2)
    k = nz - 1 - np.argmax(has[:, :, ::-1], axis=2)                  # topmost cell at or above 1/2
    lo = np.take_along_axis(e, k[:, :, None], axis=2)[:, :, 0]
    hi = np.take_along_axis(e, np.minimum(k + 1, nz - 1)[:, :, None], axis=2)[:, :, 0]
    frac = np.clip((lo - 0.5) / np.maximum(lo - hi, 1e-12), 0.0, 1.0)
    return np.where(any_m, (k + 0.5 + frac) * dx, 0.0)


def keyhole_metrics(md, x_l, y_l, radius, z_surface, window=3.0):
    """Keyhole depth and shape from the free-surface melt pool ``md`` with the beam at
    (x_l, y_l) [m], 1/e^2 radius ``radius`` [m] and the original surface at ``z_surface`` [m].
    The surface is the fill = 1/2 level of the once-smoothed fill field (the surface the rays
    see; isolated specks of metal in the keyhole do not count).

    * ``depth``: deepest point of the open depression within ``window`` radii of the beam
      axis, below the original surface [m]; ``centre_depth``: the same on the track centreline.
    * ``x_bottom``: where the deepest point is, relative to the beam axis [m] (negative: behind).
    * ``front_tan``: tan of the front-wall angle from the horizontal on the track centreline,
      a straight-line fit to the front wall between 20 and 80 % of the centreline depth.
    * ``opening``: length of the depression along the centreline at 10 % of its depth [m].
    * ``bubbles``: from ``md.bubbles`` if a :class:`BubbleModel` is attached: count, volume
      and equivalent diameters [m]."""
    dx = md.dx
    s = surface_heights(asnumpy(md.lbm.smoothed_fill(1)), dx)
    x = (np.arange(s.shape[0]) + 0.5) * dx
    y = (np.arange(s.shape[1]) + 0.5) * dx
    ix = (np.abs(x - x_l) <= window * radius)
    iy = (np.abs(y - y_l) <= window * radius)
    sub = s[np.ix_(ix, iy)]
    d = np.where(sub > 0, z_surface - sub, -np.inf)            # columns without metal: walls
    k = np.unravel_index(np.argmax(d), d.shape)
    depth = float(d[k])
    xb = float(x[ix][k[0]])
    out = dict(depth=depth, x_bottom=xb - x_l, front_tan=float("nan"), opening=float("nan"), centre_depth=float("nan"))
    j = int(np.argmin(np.abs(y - y_l)))
    prof = np.where(s[:, j] > 0, z_surface - s[:, j], 0.0)
    win = np.nonzero(ix)[0]
    ib = int(win[np.argmax(prof[win])])
    dc = float(prof[ib])
    out["centre_depth"] = dc
    if dc > 2 * dx:
        fr = ib
        while fr + 1 < len(prof) and prof[fr + 1] > 0.1 * dc:
            fr += 1
        bk = ib
        while bk - 1 >= 0 and prof[bk - 1] > 0.1 * dc:
            bk -= 1
        out["opening"] = float((fr - bk + 1) * dx)
        sel = np.arange(ib, fr + 1)
        sel = sel[(prof[sel] >= 0.2 * dc) & (prof[sel] <= 0.8 * dc)]
        if sel.size >= 2:
            slope = np.polyfit(x[sel], prof[sel], 1)[0]
            out["front_tan"] = float(abs(slope))
    bm = getattr(md, "bubbles", None)
    if bm is not None:
        vols = np.array([b["volume"] for b in bm.bubbles]) * dx ** 3
        out["n_bubbles"] = int(vols.size)
        out["bubble_volume"] = float(vols.sum())
        out["bubble_d_eq"] = [float((6.0 * v / np.pi) ** (1.0 / 3.0)) for v in vols]
    return out


def open_gas_reaches(flag, k_below):
    """True where the gas connected to the top of the domain (the open keyhole and the
    atmosphere) reaches below the cell layer ``k_below``: a keyhole broken through into what
    lies under a slab. Returns (reaches, number of such cells below)."""
    from .freesurface import GAS
    xp = get_xp()
    ndi = _ndimage(xp)
    g = np.indices((3, 3, 3)) - 1
    struct = xp.asarray((np.abs(g).sum(axis=0) <= 2).astype(np.int32))
    flag = xp.asarray(flag)
    lab, n = ndi.label(flag == GAS, structure=struct)
    if int(n) == 0:
        return False, 0
    top = xp.zeros(int(n) + 1, dtype=bool)
    top[lab[:, :, -2].ravel()] = True
    top[0] = False
    below = top[lab[:, :, 1:k_below]]
    cnt = int(below.sum())
    return cnt > 0, cnt
