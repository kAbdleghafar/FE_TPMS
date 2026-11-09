"""Melt-pool-scale model of laser powder-bed fusion: moving laser, melting, solidification.

Explicit finite-volume enthalpy method on a structured voxel grid (backend
agnostic: NumPy or CuPy, the same kernel runs on a GPU):

    dH/dt = div(k(T, phase) grad T) + q_laser - q_losses,    T = T(H)

* ``H`` is the volumetric enthalpy (sensible + latent, :class:`ThermalProperties`),
  inverted to ``T`` by table look-up, so latent heat is conserved exactly;
* conductivity at faces is the harmonic mean of the two cells; the liquid can be
  given an (anisotropic) enhancement factor, the standard conduction-model proxy
  for Marangoni convection in the melt pool;
* powder cells conduct with a reduced factor until they melt, then become dense;
* the laser is a Gaussian (or top-hat) surface flux of absorbed power
  ``A P`` (optionally spread over a penetration depth), moving along scan
  vectors; the discrete absorbed power is normalised to ``A P`` exactly;
* radiation, convection and (optionally) evaporation from the top surface; the other faces are
  adiabatic (symmetry ``y = 0`` for a single track on the centreline);
* stability: ``dt = cfl * h^2 / (6 alpha_max)`` from the property tables.

Recorded per cell (for solidification models and for ExaCA): peak temperature,
number of melting events, last melting and last solidification time (liquidus
crossings), cooling rate, thermal gradient ``G`` and solidification velocity
``R = cooling rate / G`` at the last solidification. Probes record surface
temperature histories (cooling rates as measured by thermography).

Units: mm, s, t, N, mJ (power in mW = W x 1e3), degC.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..backend import asnumpy, get_xp
from ..fem.materials import Material

SIGMA_SB = 5.670374419e-11          # mW / (mm^2 K^4)
R_GAS = 8.314462618                 # J / (mol K)


@dataclass
class Evaporation:
    """Surface evaporation (Clausius-Clapeyron vapour pressure + Hertz-Knudsen-Langmuir
    mass flux with back-condensation ``beta_r``, Anisimov). Caps the melt-pool surface near
    the boiling point. Defaults: nickel (IN625 major element)."""
    T_boil: float = 2913.0          # degC at 1 atm
    L_v: float = 6.38e6             # J/kg
    M: float = 0.0587               # kg/mol
    beta_r: float = 0.18
    p0: float = 101325.0            # Pa

    def flux(self, T, xp=np):
        """Evaporative heat flux [mW/mm^2] at surface temperature T [degC]."""
        Tk = xp.maximum(T + 273.15, 300.0)
        expo = xp.minimum(self.L_v * self.M / R_GAS * (1.0 / (self.T_boil + 273.15) - 1.0 / Tk), 50.0)
        p_sat = self.p0 * xp.exp(expo)
        mdot = (1.0 - self.beta_r) * p_sat * xp.sqrt(self.M / (2.0 * np.pi * R_GAS * Tk))   # kg/(m^2 s)
        return mdot * self.L_v * 1e-3                                                          # W/m^2 -> mW/mm^2


@dataclass
class Laser:
    power: float                 # W (nominal; per-vector power overrides it)
    absorptivity: float
    radius: float                # 1/e^2 radius [mm]  (= D4sigma / 2 for a Gaussian)
    profile: str = "gaussian"    # "gaussian" | "tophat"
    depth: float = 0.0           # exponential absorption depth [mm]; 0 = into the top cell


@dataclass
class ScanVector:
    x0: float
    y0: float
    x1: float
    y1: float
    speed: float                 # mm/s
    power: Optional[float] = None    # W, None = laser.power; 0 = jump (laser off)


def raster(x0, x1, y0, y1, hatch, speed, power=None, bidirectional=True, along="x", jump_speed=None):
    """Hatch vectors filling a rectangle (alternating direction if bidirectional)."""
    vecs = []
    if along == "x":
        ys = np.arange(y0, y1 + 1e-12, hatch)
        for i, y in enumerate(ys):
            a, b = (x0, x1) if (i % 2 == 0 or not bidirectional) else (x1, x0)
            if vecs and jump_speed:
                vecs.append(ScanVector(vecs[-1].x1, vecs[-1].y1, a, y, jump_speed, 0.0))
            vecs.append(ScanVector(a, y, b, y, speed, power))
    else:
        xs = np.arange(x0, x1 + 1e-12, hatch)
        for i, x in enumerate(xs):
            a, b = (y0, y1) if (i % 2 == 0 or not bidirectional) else (y1, y0)
            if vecs and jump_speed:
                vecs.append(ScanVector(vecs[-1].x1, vecs[-1].y1, x, a, jump_speed, 0.0))
            vecs.append(ScanVector(x, a, x, b, speed, power))
    return vecs


@dataclass
class MeltPoolResult:
    h: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray                # cell-centre coordinates
    T: np.ndarray                # final temperature (nx, ny, nz)
    T_peak: np.ndarray
    n_melt: np.ndarray
    t_melt: np.ndarray           # last time above liquidus began (nan = never melted)
    t_solid: np.ndarray          # last liquidus crossing on cooling
    cooling_rate: np.ndarray     # at t_solid [K/s]
    G: np.ndarray                # |grad T| at t_solid [K/mm]
    R: np.ndarray                # cooling_rate / G [mm/s]
    grad_dir: np.ndarray         # unit grad T at t_solid (nx, ny, nz, 3): local growth direction
    probes: dict
    snapshots: list
    dt: float
    steps: int
    seconds: float
    symmetric_y: bool
    T_solidus: float
    T_liquidus: float
    absorbed_energy: float       # mJ (whole body, incl. mirror half if symmetric)
    energy_balance: float        # relative error of enthalpy gain vs absorbed - lost energy


class MeltPoolModel:
    """Structured-grid melt-pool model.

    The substrate occupies ``z in [-depth, 0]``; an optional powder layer
    ``z in [0, powder]`` sits on top. ``x in [0, length]``; ``y in [0, width]``
    with a symmetry plane at ``y = 0`` (``symmetric_y=True``, single track on
    ``y = 0``) or ``y in [-width/2, width/2]``.
    """

    def __init__(self, material: Material, h: float, length: float, width: float, depth: float, *,
                 powder: float = 0.0, powder_k_factor: float = 0.02, T0: float = 20.0,
                 symmetric_y: bool = True, liquid_k_factor=(1.0, 1.0, 1.0), emissivity: float = 0.4,
                 h_conv: float = 1e-5, cfl: float = 0.9, dtype=np.float64, T_max: float = 6000.0,
                 evaporation: Optional[Evaporation] = None):
        th = material.thermal
        if th is None:
            raise ValueError("material needs ThermalProperties")
        self.xp = xp = get_xp()
        self.material, self.h, self.T0 = material, float(h), float(T0)
        self.dtype = dtype
        self.symmetric_y = symmetric_y
        nx, ny = int(round(length / h)), int(round(width / h))
        nzs, nzp = int(round(depth / h)), int(round(powder / h))
        self.shape = (nx, ny, nzs + nzp)
        self.nz_sub, self.nz_powder = nzs, nzp
        self.x = (np.arange(nx) + 0.5) * h
        self.y = (np.arange(ny) + 0.5) * h if symmetric_y else (np.arange(ny) + 0.5) * h - width / 2
        self.z = -depth + (np.arange(nzs + nzp) + 0.5) * h
        self.Ts, self.Tl = float(th.T_solidus), float(th.T_liquidus)
        self.powder_k_factor = float(powder_k_factor)
        self.fl = np.broadcast_to(np.asarray(liquid_k_factor, float), (3,)).copy()
        self.emissivity, self.h_conv = float(emissivity), float(h_conv)
        self.evaporation = evaporation
        # property tables (1 K) on the backend
        tt, HH = th.enthalpy_table(T_lo=T0 - 50.0, T_hi=T_max, dT=1.0)
        self._tt, self._HH = xp.asarray(tt, dtype=dtype), xp.asarray(HH, dtype=dtype)
        self._kT = xp.asarray(th.conductivity(tt), dtype=dtype)
        rc = th.rho * np.interp(tt, th.T, th.cp)
        alpha = th.conductivity(tt) * self.fl.max() / rc
        self.dt = cfl * h * h / (6.0 * alpha.max())
        # state
        self.H = xp.full(self.shape, float(np.interp(T0, tt, HH)), dtype=dtype)
        self.T = xp.full(self.shape, T0, dtype=dtype)
        self.powder_mask = xp.zeros(self.shape, dtype=bool)
        if nzp:
            self.powder_mask[:, :, nzs:] = True
        self.t = 0.0

    # ------------------------------------------------------------------
    def _T_of_H(self, H):
        return self.xp.interp(H.ravel(), self._HH, self._tt).reshape(H.shape)

    def _conductivity(self, T):
        """Base conductivity k(T) and liquid fraction."""
        xp = self.xp
        k = xp.interp(T.ravel(), self._tt, self._kT).reshape(T.shape)
        k = xp.where(self.powder_mask, k * self.powder_k_factor, k)
        phi = xp.clip((T - self.Ts) / max(self.Tl - self.Ts, 1e-9), 0.0, 1.0)
        return k, phi

    def _source(self, laser, x_l, y_l, P_w):
        """Absorbed power density [mW/mm^3] (nx, ny, nz), normalised to A*P exactly."""
        xp, h = self.xp, self.h
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        r2 = (X - x_l) ** 2 + (Y - y_l) ** 2
        w = laser.radius
        if laser.profile == "gaussian":
            f = np.exp(-2.0 * r2 / (w * w))
        else:
            f = (r2 <= w * w).astype(float)
        area_w = f.sum() * h * h * (2.0 if self.symmetric_y else 1.0)
        if area_w <= 0:
            return None
        qs = laser.absorptivity * P_w * 1e3 * f / area_w             # mW/mm^2, integrates to A P
        q = np.zeros(self.shape)
        top = self.shape[2] - 1
        if laser.depth > 0:
            depth_from_top = (top - np.arange(self.shape[2]) + 0.5) * h
            wz = np.exp(-depth_from_top / laser.depth)
            wz[depth_from_top > 6 * laser.depth] = 0.0
            wz /= wz.sum() * h
            q = qs[:, :, None] * wz[None, None, :]
        else:
            q[:, :, top] = qs / h
        return xp.asarray(q, dtype=self.dtype)

    # ------------------------------------------------------------------
    def run(self, laser: Laser, path: Sequence[ScanVector], t_extra: float = 0.0, probes: Optional[dict] = None,
            snapshot_times: Sequence[float] = (), verbose: bool = False,
            snapshot_reduce=None) -> MeltPoolResult:
        """Scan ``path`` (then let it cool for ``t_extra``). ``probes``: {name: (x, y)} surface
        points whose temperature history is recorded (top-cell value extrapolated to the surface).
        ``snapshot_reduce``: optional function applied to each temperature snapshot (a host
        array, shape (nx, ny, nz)) before it is stored, e.g. ``lambda T: T[:, :, -1]`` to keep
        only the surface when many frames are recorded for an animation."""
        xp, h, dt = self.xp, self.h, self.dt
        nx, ny, nz = self.shape
        t_start = time.perf_counter()
        # path schedule
        seg_t = []
        tcur = 0.0
        for v in path:
            L = float(np.hypot(v.x1 - v.x0, v.y1 - v.y0))
            seg_t.append((tcur, tcur + L / v.speed, v))
            tcur += L / v.speed
        t_end = tcur + t_extra
        T_peak = self.T.copy()
        n_melt = xp.zeros(self.shape, dtype=np.int32)
        nan = xp.full(self.shape, np.nan, dtype=self.dtype)
        t_melt, t_sol, cr, G = nan.copy(), nan.copy(), nan.copy(), nan.copy()
        gdir = xp.full(self.shape + (3,), np.nan, dtype=self.dtype)
        probe_ij = {}
        for name, (px, py) in (probes or {}).items():
            probe_ij[name] = (int(np.clip(np.floor(px / h), 0, nx - 1)),
                              int(np.clip(np.floor((py - self.y[0] + h / 2) / h), 0, ny - 1)))
        hist = {name: [] for name in probe_ij}
        times = []
        snaps, snap_q = [], sorted(snapshot_times)
        absorbed = 0.0
        lost = 0.0
        H0 = float(self.H.sum()) * h ** 3
        Ta4 = (self.T0 + 273.15) ** 4
        fl = self.fl
        step = 0
        q_cache = {}
        while self.t < t_end - 1e-15:
            dts = min(dt, t_end - self.t)
            tm = self.t + 0.5 * dts
            q = None
            for (a, b, v) in seg_t:
                if a <= tm < b:
                    P = laser.power if v.power is None else v.power
                    if P > 0:
                        s = (tm - a) / (b - a)
                        x_l, y_l = v.x0 + s * (v.x1 - v.x0), v.y0 + s * (v.y1 - v.y0)
                        key = (round(x_l / (0.25 * h)), round(y_l / (0.25 * h)), P)
                        q = q_cache.get(key)
                        if q is None:
                            q = self._source(laser, x_l, y_l, P)
                            if len(q_cache) > 64:
                                q_cache.clear()
                            q_cache[key] = q
                    break
            T = self.T
            k, phi = self._conductivity(T)
            div = xp.zeros(self.shape, dtype=self.dtype)
            for ax in range(3):
                kf = k * (1.0 + (fl[ax] - 1.0) * phi) if fl[ax] != 1.0 else k
                a_ = [slice(None)] * 3
                b_ = [slice(None)] * 3
                a_[ax], b_[ax] = slice(None, -1), slice(1, None)
                ka, kb = kf[tuple(a_)], kf[tuple(b_)]
                kface = 2.0 * ka * kb / (ka + kb)
                flux = kface * (T[tuple(b_)] - T[tuple(a_)])
                div[tuple(a_)] += flux
                div[tuple(b_)] -= flux
            dH = div * (1.0 / (h * h))
            # top-surface losses (radiation + convection)
            Tt = T[:, :, -1]
            ql = self.emissivity * SIGMA_SB * ((Tt + 273.15) ** 4 - Ta4) + self.h_conv * (Tt - self.T0)
            if self.evaporation is not None:
                ql = ql + self.evaporation.flux(Tt, xp)
            dH[:, :, -1] -= ql / h
            mirror = 2.0 if self.symmetric_y else 1.0
            lost += float(ql.sum()) * h * h * dts * mirror
            if q is not None:
                dH += q
                absorbed += float(q.sum()) * h ** 3 * dts * mirror
            self.H += dts * dH
            T_new = self._T_of_H(self.H)
            # liquidus crossings
            up = (T < self.Tl) & (T_new >= self.Tl)
            down = (T >= self.Tl) & (T_new < self.Tl)
            if bool(up.any()):
                frac = (self.Tl - T[up]) / xp.maximum(T_new[up] - T[up], 1e-12)
                t_melt[up] = self.t + frac * dts
                n_melt[up] += 1
                self.powder_mask[up] = False             # consolidated
            if bool(down.any()):
                frac = (T[down] - self.Tl) / xp.maximum(T[down] - T_new[down], 1e-12)
                t_sol[down] = self.t + frac * dts
                rate = (T[down] - T_new[down]) / dts
                cr[down] = rate
                gv = self._grad(T_new, down)
                g = xp.sqrt(gv[0] ** 2 + gv[1] ** 2 + gv[2] ** 2)
                G[down] = g
                gdir[down] = xp.stack(gv, axis=-1) / xp.maximum(g, 1e-30)[:, None]
            T_peak = xp.maximum(T_peak, T_new)
            self.T = T_new
            self.t += dts
            step += 1
            if probe_ij:
                times.append(self.t)
                for name, (i, j) in probe_ij.items():
                    hist[name].append(float(1.5 * T_new[i, j, -1] - 0.5 * T_new[i, j, -2]))
            while snap_q and self.t >= snap_q[0] - 1e-15:
                Ts = asnumpy(self.T).copy()
                snaps.append((self.t, Ts if snapshot_reduce is None else snapshot_reduce(Ts)))
                snap_q.pop(0)
            if verbose and step % 1000 == 0:
                print(f"  t = {self.t * 1e3:.3f} ms  T_max = {float(T_new.max()):.0f} C  "
                      f"({time.perf_counter() - t_start:.0f}s)", flush=True)
        dHtot = float(self.H.sum()) * h ** 3 * (2.0 if self.symmetric_y else 1.0) - H0 * (2.0 if self.symmetric_y else 1.0)
        bal = (dHtot - (absorbed - lost)) / max(abs(absorbed), 1e-30)
        Gn = asnumpy(G)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = asnumpy(cr) / Gn
        return MeltPoolResult(h, self.x, self.y, self.z, asnumpy(self.T), asnumpy(T_peak), asnumpy(n_melt),
                              asnumpy(t_melt), asnumpy(t_sol), asnumpy(cr), Gn, R, asnumpy(gdir),
                              {"t": np.array(times), **{k_: np.array(v_) for k_, v_ in hist.items()}},
                              snaps, dt, step, time.perf_counter() - t_start, self.symmetric_y,
                              self.Ts, self.Tl, absorbed, bal)

    def _grad(self, T, mask):
        """grad T components [K/mm] at the cells of ``mask`` (central differences, one-sided
        at faces; zero normal gradient on the symmetry plane)."""
        xp, h = self.xp, self.h
        idx = xp.nonzero(mask)
        comps = []
        for ax in range(3):
            n = T.shape[ax]
            i = idx[ax]
            ip, im = xp.minimum(i + 1, n - 1), xp.maximum(i - 1, 0)
            if ax == 1 and self.symmetric_y:
                im = xp.where(i == 0, 0, im)
            sel_p, sel_m = list(idx), list(idx)
            sel_p[ax], sel_m[ax] = ip, im
            d = (T[tuple(sel_p)] - T[tuple(sel_m)]) / (xp.maximum(ip - im, 1) * h)
            comps.append(d)
        return comps


# ---------------------------------------------------------------- measurements
def _cross(vals, coords, level):
    """Last coordinate where ``vals`` (monotone-ish profile from the inside out) >= level,
    linearly interpolated to the crossing."""
    above = np.flatnonzero(vals >= level)
    if len(above) == 0:
        return None
    i = above[-1]
    if i == len(vals) - 1:
        return coords[i]
    v0, v1 = vals[i], vals[i + 1]
    return coords[i] + (coords[i + 1] - coords[i]) * (v0 - level) / (v0 - v1)


def fusion_zone(res: MeltPoolResult, x_section: float, level: Optional[float] = None):
    """Width and depth [mm] of the melted zone (peak temperature >= ``level``, default the
    solidus) in the cross-section ``x = x_section``. The top-cell values are extrapolated
    to the surface; the symmetry plane doubles the half width."""
    level = res.T_solidus if level is None else level
    i = int(np.argmin(np.abs(res.x - x_section)))
    Tp = res.T_peak[i]                                   # (ny, nz)
    surf = 1.5 * Tp[:, -1] - 0.5 * Tp[:, -2]
    if res.symmetric_y:
        yy = np.r_[0.0, res.y]
        prof = np.r_[1.5 * surf[0] - 0.5 * surf[1] if len(surf) > 1 else surf[0], surf]
        ym = _cross(prof, yy, level)
        width = None if ym is None else 2.0 * ym
        col = Tp[0]
    else:
        j0 = int(np.argmin(np.abs(res.y)))
        right = _cross(surf[j0:], res.y[j0:], level)
        left = _cross(surf[:j0 + 1][::-1], res.y[:j0 + 1][::-1], level)
        width = None if right is None or left is None else right - left
        col = Tp[j0]
    zs = np.r_[0.0, res.z[::-1]]
    colz = np.r_[1.5 * col[-1] - 0.5 * col[-2], col[::-1]]
    d = _cross(colz, -zs, level)
    depth = None if d is None else d
    return width, depth


def pool_length(T_field: np.ndarray, res: MeltPoolResult, level: Optional[float] = None):
    """Length [mm] of the region at or above ``level`` (default solidus) along the
    surface centreline of a temperature snapshot."""
    level = res.T_solidus if level is None else level
    j0 = 0 if res.symmetric_y else int(np.argmin(np.abs(res.y)))
    line = 1.5 * T_field[:, j0, -1] - 0.5 * T_field[:, j0, -2]
    above = np.flatnonzero(line >= level)
    if len(above) == 0:
        return 0.0
    a, b = above[0], above[-1]
    x = res.x
    lo = x[a] - (x[a] - x[a - 1]) * (line[a] - level) / (line[a] - line[a - 1]) if a > 0 else x[a]
    hi = x[b] + (x[b + 1] - x[b]) * (line[b] - level) / (line[b] - line[b + 1]) if b < len(x) - 1 else x[b]
    return float(hi - lo)


def cooling_rate_history(t, Tp, T_from=1290.0, T_to=1190.0):
    """Mean cooling rate [K/s] of a probe history between two temperatures on cooling
    (as defined for the AM-Bench thermography: time to cool from the solidus to T_to)."""
    t, Tp = np.asarray(t), np.asarray(Tp)
    ipk = int(np.argmax(Tp))
    tt, TT = t[ipk:], Tp[ipk:]
    if TT[0] < T_from or TT.min() > T_to:
        return np.nan

    def when(level):
        i = int(np.flatnonzero(TT <= level)[0])
        return tt[i - 1] + (tt[i] - tt[i - 1]) * (TT[i - 1] - level) / (TT[i - 1] - TT[i])
    return (T_from - T_to) / (when(T_to) - when(T_from))


def moving_gaussian_analytic(x, y, z, P_abs, v, w, k, rho_c, T0=20.0):
    """Quasi-steady temperature [degC] of a Gaussian surface source (absorbed power
    ``P_abs`` [mW], 1/e^2 radius ``w``) moving at ``v`` along +x on a semi-infinite body
    with constant properties (Eagar & Tsai 1983); ``x`` is relative to the beam centre,
    ``z <= 0`` the depth. Used to verify the numerical solver."""
    from scipy.integrate import quad

    a = k / rho_c
    s2 = (w / 2.0) ** 2                          # Gaussian variance

    def integrand(s):                            # tau = exp(s): resolves the short-time peak
        tau = np.exp(s)
        lat = 2.0 * (s2 + 2.0 * a * tau)
        return tau * (np.exp(-((x + v * tau) ** 2 + y ** 2) / lat - z * z / (4.0 * a * tau))
                      / (np.pi * lat) / np.sqrt(4.0 * np.pi * a * tau))

    val, _ = quad(integrand, np.log(1e-14), np.log(1e3), limit=1000, epsabs=0.0, epsrel=1e-9)
    return T0 + 2.0 * P_abs / rho_c * val
