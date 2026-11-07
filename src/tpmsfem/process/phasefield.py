"""2-D quantitative phase-field model of dilute-alloy directional solidification.

Resolves cells and dendrites and the solute segregation between them, at the melt-pool
conditions (thermal gradient G, solidification velocity R) computed by
:mod:`tpmsfem.process.meltpool`.

Model (Karma, PRL 87 (2001) 115701; Echebarria, Folch, Karma & Plapp, PRE 70 (2004) 061604):
dimensionless phase field ``phi`` (+1 solid, -1 liquid) and supersaturation

    U = [2 c / c_l0 / (1 + k - (1 - k) phi) - 1] / (1 - k),     c_l0 = c_inf / k,

in the frozen-temperature approximation ``theta = (z - z0 - V t) / l_T``, ``l_T = dT0 / G``:

    tau0 a(n)^2 [1 - (1 - k) theta] d_t phi = div(W(n)^2 grad phi) + anisotropic terms
                                               + phi - phi^3 - lambda (1 - phi^2)^2 (U + theta)
    (1 + k - (1 - k) phi)/2 d_t U = div( D q(phi) grad U + a_t W [1 + (1 - k) U] d_t phi n )
                                    + [1 + (1 - k) U]/2 d_t phi

with ``q = (1 - phi)/2`` (no diffusion in the solid), the anti-trapping current
``a_t = 1/(2 sqrt 2)``, ``W(n) = W0 (1 + eps4 cos 4(theta_n - theta_0))`` (``theta_0``: tilt of
the crystal's <10> axis from the growth direction), ``lambda = a1 W0/d0`` and
``tau0 = a2 lambda W0^2 / D`` (``a1 = 0.8839``, ``a2 = 0.6267``): the thin-interface limit
with vanishing interface kinetics and no spurious solute trapping.

Numerics: explicit finite differences on a uniform grid (``dx = 0.8 W0``), conservative face
fluxes, periodic in x, no-flux in z; lab frame with a window that shifts whenever the front
passes 60 % of the height, while the rows leaving the window are stored as the solidified
microstructure (``c / c_inf``). NumPy/CuPy backend.

Validity. The model needs ``W0`` well below the diffusion length ``D/V`` and the tip radius.
At LPBF speeds (``D/V`` ~ 10-100 nm) this is marginal, so check convergence in ``W0``.
Non-equilibrium solute trapping and kinetic undercooling at very high V are not included.
The run is 2-D: 2-D spacings are larger than 3-D ones.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..backend import asnumpy, get_xp

A1, A2, A_AT = 0.8839, 0.6267, 1.0 / (2.0 * np.sqrt(2.0))


@dataclass(frozen=True)
class DiluteAlloy:
    """Effective dilute binary alloy with a linear phase diagram.

    Defaults: IN625 as quasi-binary Ni-Nb (Ghosh et al., JOM 70 (2018) 1011: 4 wt% Nb,
    k = 0.48, eps4 = 0.02) with the Ni-Nb liquidus slope, liquid diffusivity and Gibbs-Thomson
    coefficient of Ghosh et al., MSMSE 25 (2017) 065002 (m = -10.5 K/wt%, D = 3e-9 m^2/s,
    Gamma = 3.65e-7 K m), so dT0 = |m| c_inf (1 - k)/k = 45.5 K. Use :meth:`from_slope`
    for other alloys."""
    c_inf: float = 4.0               # wt% Nb
    k: float = 0.48
    freezing_range: float = 45.5     # dT0 = |m| c_inf (1 - k) / k  [K]
    D: float = 3.0e-9                # liquid diffusivity [m^2/s]
    gibbs_thomson: float = 3.65e-7   # Gamma [K m]
    eps4: float = 0.02

    @classmethod
    def from_slope(cls, c_inf: float, k: float, m: float, **kw) -> "DiluteAlloy":
        """From the liquidus slope ``m`` [K/wt%] (dT0 = |m| c_inf (1 - k)/k)."""
        return cls(c_inf=c_inf, k=k, freezing_range=abs(m) * c_inf * (1.0 - k) / k, **kw)

    @property
    def d0(self):
        """Chemical capillary length Gamma / dT0 [m]."""
        return self.gibbs_thomson / self.freezing_range

    @property
    def m(self):
        """Liquidus slope [K/wt%]."""
        return -self.freezing_range * self.k / (self.c_inf * (1.0 - self.k))


#: IN625 (Ghosh et al. 2018) and the IN718-like Ni-5Nb alloy of Ghosh et al. (2017).
IN625_NB = DiluteAlloy()
NI5NB_GHOSH2017 = DiluteAlloy.from_slope(5.0, 0.48, -10.5, eps4=0.03)


@dataclass
class PFResult:
    phi: np.ndarray
    c: np.ndarray                    # c / c_inf in the final window
    solid_c: np.ndarray              # c / c_inf of the rows that left the window (oldest first)
    dx_m: float
    t_s: float
    steps: int
    seconds: float
    front_history: list = field(default_factory=list)   # (t [s], highest solid row above the start [m])
    params: dict = field(default_factory=dict)

    def tips(self, prominence_m: Optional[float] = None, active_dT: float = 0.02, phi_solid: float = 0.9):
        """Active cell / dendrite tips: peaks of the phi = 0 front profile (periodic in x)
        with a prominence of at least ``prominence_m`` (default 2 W0) that lie within
        ``active_dT * dT0`` (as a height, ``active_dT * l_T``) of the leading tip; lagging cells
        being overgrown are not counted. Each tip: column, height [m], the solid composition
        c_s* / c_inf behind it (the first cell with ``phi > phi_solid`` in the solid connected
        to the tip, else its most solid cell), the peak liquid composition c_l* / c_inf ahead
        of it and the effective partition c_s* / c_l*."""
        from scipy.signal import find_peaks

        nz, nx = self.phi.shape
        if not (self.phi > 0.0).any():
            return []
        h = front_profile(self.phi, self.dx_m)
        prom = prominence_m if prominence_m is not None else 2.0 * self.params.get("W0_m", self.dx_m / 0.8)
        if h.max() - h.min() < prom:
            return []                                             # planar front
        ext = np.concatenate([h, h, h])
        pk, _ = find_peaks(ext, prominence=prom)
        pk = pk[(pk >= nx) & (pk < 2 * nx)] - nx
        band = active_dT * self.params.get("lT_m", np.inf)
        out = []
        for j in pk:
            if h[j] < h.max() - band:
                continue
            i0 = int(np.flatnonzero(self.phi[:, j] > 0.0).max())    # last solid cell (phi > 0) of the tip
            col = self.phi[:i0 + 1, j]
            liquid = np.flatnonzero(col <= 0.0)
            seg = np.arange(liquid.max() + 1 if len(liquid) else 0, i0 + 1)   # solid connected to the tip
            inside = seg[col[seg] > phi_solid]
            i_s = inside.max() if len(inside) else seg[np.argmax(col[seg])]
            cs = float(self.c[i_s, j])
            cl = float(self.c[i0:min(i0 + 20, nz), j].max())   # liquid peak just ahead of the tip
            out.append(dict(column=int(j), height_m=float(h[j]), cs=cs, cl=cl, k_eff=cs / cl))
        return out

    def boundaries(self, depth=(0.25, 0.5), rel_prominence: float = 0.3, n_rows: int = 9):
        """Mean number of Nb-rich cell boundaries crossed by horizontal lines below the leading
        tip, as counted on a micrograph: peaks of c(x) (periodic) with a prominence of at least
        ``rel_prominence`` of the line's composition range, on ``n_rows`` lines spread over
        ``depth`` (fractions of the tip height above the window bottom, counted down from the
        tip). Lagging cells that are still growing count; cells already cut off do not."""
        from scipy.signal import find_peaks

        nz, nx = self.phi.shape
        rows = np.flatnonzero((self.phi > 0.0).any(axis=1))
        if not len(rows):
            return 0.0
        tip = int(rows.max())
        counts = []
        for f in np.linspace(depth[0], depth[1], n_rows):
            row = self.c[max(int(round(tip * (1.0 - f))), 0)]
            rng = float(row.max() - row.min())
            if rng < 1e-3:
                counts.append(0)                                  # planar
                continue
            ext = np.concatenate([row, row, row])
            pk, _ = find_peaks(ext, prominence=rel_prominence * rng)
            counts.append(int(np.sum((pk >= nx) & (pk < 2 * nx))))
        return float(np.mean(counts))

    def spacing(self, method: str = "boundaries", **kw):
        """(primary spacing [m], number of cells); ``inf`` for a planar front.
        ``method="boundaries"`` (default) counts cell boundaries below the tips
        (:meth:`boundaries`), ``"tips"`` counts active tips (:meth:`tips`); keyword arguments
        go to that method."""
        n = self.boundaries(**kw) if method == "boundaries" else len(self.tips(**kw))
        if n == 0:
            return np.inf, 0
        return self.phi.shape[1] * self.dx_m / n, n

    def segregation(self):
        """Statistics of c / c_inf in the solidified rows (beyond the initial transient)."""
        s = self.solid_c
        if s.size == 0:
            s = self.c[self.phi > 0.5]
        s = s[s.shape[0] // 4:] if s.ndim == 2 and s.shape[0] > 8 else s
        v = np.asarray(s).ravel()
        return dict(min=float(v.min()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)),
                    p99=float(np.percentile(v, 99)), max=float(v.max()))


class DirectionalPF:
    """Directional solidification at thermal gradient ``G`` [K/m] and velocity ``V`` [m/s]."""

    def __init__(self, alloy: DiluteAlloy, G: float, V: float, width_m: float, height_m: float,
                 W_over_d0: float = 2.5, dx_over_W: float = 0.8, tilt_deg: float = 0.0, noise: float = 0.02,
                 seed: int = 0, dtype=np.float64, kernel: str = "auto", W0_m: Optional[float] = None):
        self.xp = xp = get_xp()
        if kernel == "auto":
            kernel = "numba" if (xp is np and numba_available()) else "array"
        if kernel == "numba" and xp is not np:
            raise ValueError("the numba kernel runs on the NumPy backend only")
        self.kernel = kernel
        self._nb = None
        self.alloy, self.G, self.V = alloy, float(G), float(V)
        W_over_d0 = W0_m / alloy.d0 if W0_m is not None else W_over_d0
        self.W0 = W_over_d0 * alloy.d0                                    # [m]
        self.lam = A1 * W_over_d0
        self.tau0 = A2 * self.lam * self.W0 ** 2 / alloy.D                # [s]
        self.Dt = A2 * self.lam                                           # D tau0 / W0^2
        self.dx = dx_over_W                                               # in W0
        self.lT = alloy.freezing_range / self.G / self.W0                 # in W0
        self.Vt = self.V * self.tau0 / self.W0                            # dimensionless velocity
        k = alloy.k
        self.dt = 0.5 * min(self.dx ** 2 * k / 4.0, self.dx ** 2 / (4.0 * self.Dt))
        self.nx = int(round(width_m / (self.dx * self.W0)))
        self.nz = int(round(height_m / (self.dx * self.W0)))
        self.theta0 = np.radians(tilt_deg)
        self._c0, self._s0 = float(np.cos(self.theta0)), float(np.sin(self.theta0))
        self._bufs = None
        rng = np.random.default_rng(seed)
        z = (np.arange(self.nz) + 0.5) * self.dx
        z_front = 0.1 * self.nz * self.dx                                 # initial planar front
        prof = -np.tanh((z - z_front) / np.sqrt(2.0))
        phi = np.repeat(prof[:, None], self.nx, axis=1)
        phi += noise * rng.standard_normal(phi.shape) * (1.0 - phi ** 2)
        self.phi = xp.asarray(np.clip(phi, -1.0, 1.0), dtype=dtype)
        # steady planar front at T0 (theta = 0): solid at c_inf (U = 0), liquid boundary layer
        # c_l = c_inf [1 + (1 - k)/k exp(-V z / D)]  <=>  U = exp(-z / l_D) - 1
        lD = self.alloy.D / self.V / self.W0
        U_liq = np.expm1(-np.maximum(z - z_front, 0.0) / lD)
        h = 0.5 * (1.0 + prof)                                            # 1 solid, 0 liquid
        self.U = xp.asarray(np.repeat((h * 0.0 + (1.0 - h) * U_liq)[:, None], self.nx, axis=1), dtype=dtype)
        self.z0 = z_front                                                 # T0 (solidus of c_inf) at the start
        self.offset = 0.0                                                 # window shift [W0]
        self.t = 0.0
        self.z = xp.asarray(z, dtype=dtype)[:, None]
        self._zc = np.ascontiguousarray(z, dtype=np.float64)
        self.solid_rows = []
        self.front = []

    # ------------------------------------------------------------------
    def composition(self, phi=None, U=None):
        """c / c_inf."""
        k = self.alloy.k
        phi = self.phi if phi is None else phi
        U = self.U if U is None else U
        return (1.0 + k - (1.0 - k) * phi) / (2.0 * k) * (1.0 + (1.0 - k) * U)

    def _aniso(self, px, pz):
        """a(n) = 1 + eps4 cos 4(psi - theta0) and da/dpsi, psi = angle of grad(phi), from the
        gradient components (no trigonometric calls)."""
        eps, c0, s0 = self.alloy.eps4, self._c0, self._s0
        u, w = (px, pz) if s0 == 0.0 else (c0 * px + s0 * pz, c0 * pz - s0 * px)
        u2, w2 = u * u, w * w
        g2 = u2 + w2
        inv = 1.0 / (g2 * g2 + 1e-24)                                     # regularised in the bulk
        a = 1.0 + eps * (u2 * u2 - 6.0 * u2 * w2 + w2 * w2) * inv
        ap = -16.0 * eps * u * w * (u2 - w2) * inv
        return a, ap, g2

    def _pad(self, f, buf):
        """Ghost layer: periodic in x (axis 1), mirror (no-flux) in z (axis 0)."""
        buf[1:-1, 1:-1] = f
        buf[1:-1, 0] = f[:, -1]
        buf[1:-1, -1] = f[:, 0]
        buf[0] = buf[1]
        buf[-1] = buf[-2]
        return buf

    def step(self):
        if self.kernel == "numba":
            return self._step_numba()
        xp, dx, k, dt = self.xp, self.dx, self.alloy.k, self.dt
        if self._bufs is None or self._bufs[0].shape != (self.nz + 2, self.nx + 2):
            self._bufs = [xp.empty((self.nz + 2, self.nx + 2), dtype=self.phi.dtype) for _ in range(3)]
        P = self._pad(self.phi, self._bufs[0])
        Q = self._pad(self.U, self._bufs[1])
        phi, U = P[1:-1, 1:-1], Q[1:-1, 1:-1]
        # x-faces j-1/2 .. nx-1/2 (nx+1 faces) on rows 1..nz
        PL, PR = P[1:-1, :-1], P[1:-1, 1:]
        px_f = (PR - PL) / dx
        pz_f = (P[2:, :-1] + P[2:, 1:] - P[:-2, :-1] - P[:-2, 1:]) / (4.0 * dx)
        a, ap, gx2 = self._aniso(px_f, pz_f)
        Fx = a * a * px_f - a * ap * pz_f
        # z-faces i-1/2 .. nz-1/2 (nz+1 faces); no flux through the bottom and top
        PB, PT = P[:-1, 1:-1], P[1:, 1:-1]
        pz_g = (PT - PB) / dx
        px_g = (P[:-1, 2:] + P[1:, 2:] - P[:-1, :-2] - P[1:, :-2]) / (4.0 * dx)
        b, bp, gz2 = self._aniso(px_g, pz_g)
        Fz = b * b * pz_g + b * bp * px_g
        Fz[0] = 0.0
        Fz[-1] = 0.0
        div = (Fx[:, 1:] - Fx[:, :-1] + Fz[1:] - Fz[:-1]) / dx
        ac, _, _ = self._aniso((P[1:-1, 2:] - P[1:-1, :-2]) / (2.0 * dx), (P[2:, 1:-1] - P[:-2, 1:-1]) / (2.0 * dx))
        theta = (self.z + self.offset - self.z0 - self.Vt * self.t) / self.lT
        tau = ac * ac * xp.maximum(1.0 - (1.0 - k) * theta, k)
        dtphi = (div + phi - phi ** 3 - self.lam * (1.0 - phi * phi) ** 2 * (U + theta)) / tau
        # solute: D q(phi) grad U + anti-trapping, conservative face fluxes
        R = self._pad(dtphi, self._bufs[2])
        F = 1.0 + (1.0 - k) * Q                                           # [1 + (1 - k) U] incl. ghosts
        QL, QR = Q[1:-1, :-1], Q[1:-1, 1:]
        Jx = (self.Dt * 0.25 * (2.0 - PL - PR) * (QR - QL) / dx
              + A_AT * 0.25 * (F[1:-1, :-1] + F[1:-1, 1:]) * (R[1:-1, :-1] + R[1:-1, 1:]) * px_f / xp.sqrt(gx2 + 1e-24))
        QB, QT = Q[:-1, 1:-1], Q[1:, 1:-1]
        Jz = (self.Dt * 0.25 * (2.0 - PB - PT) * (QT - QB) / dx
              + A_AT * 0.25 * (F[:-1, 1:-1] + F[1:, 1:-1]) * (R[:-1, 1:-1] + R[1:, 1:-1]) * pz_g / xp.sqrt(gz2 + 1e-24))
        Jz[0] = 0.0
        Jz[-1] = 0.0
        divJ = (Jx[:, 1:] - Jx[:, :-1] + Jz[1:] - Jz[:-1]) / dx
        phin = phi + dt * dtphi
        coef = (1.0 + k - (1.0 - k) * phin) / 2.0
        self.U = U + dt * (divJ + 0.5 * F[1:-1, 1:-1] * dtphi) / coef
        self.phi = phin
        self.t += dt

    def _step_numba(self):
        if self._nb is None or self._nb[0].shape != self.phi.shape:
            self._nb = [np.empty_like(self.phi) for _ in range(3)]
        R, phin, Un = self._nb
        _numba_kernels()(self.phi, self.U, R, phin, Un, self._zc, self.offset, self.z0, self.Vt, self.t, self.lT,
                         self.alloy.k, self.lam, self.Dt, A_AT, self.dx, self.dt, self.alloy.eps4, self._c0, self._s0)
        self._nb = [R, self.phi, self.U]
        self.phi, self.U = phin, Un
        self.t += self.dt

    def _shift(self, rows):
        """Move the window up by ``rows``: store the leaving solid, add far-field liquid."""
        xp = self.xp
        self.solid_rows.append(asnumpy(self.composition()[:rows]))
        liq_phi = xp.full((rows, self.nx), -1.0, dtype=self.phi.dtype)
        liq_U = xp.full((rows, self.nx), -1.0, dtype=self.U.dtype)
        self.phi = xp.concatenate([self.phi[rows:], liq_phi], axis=0)
        self.U = xp.concatenate([self.U[rows:], liq_U], axis=0)
        self.offset += rows * self.dx

    def solute(self):
        """Total solute in the window, sum of c / c_inf over the cells."""
        return float(self.composition().sum())

    def run(self, t_end_s: float, shift_at: Optional[float] = 0.6, record_every: int = 200,
            verbose: bool = False) -> PFResult:
        """Advance to ``t_end_s`` (physical time). ``shift_at=None`` keeps the window fixed
        (closed, solute-conserving box)."""
        t0 = time.perf_counter()
        n = int(np.ceil(t_end_s / (self.dt * self.tau0)))
        for s in range(n):
            self.step()
            if s % record_every == 0 or s == n - 1:
                phi_h = asnumpy(self.phi)
                rows = np.flatnonzero((phi_h > 0.0).any(axis=1))
                tip = int(rows.max()) if len(rows) else 0
                self.front.append((self.t * self.tau0, ((tip + 0.5) * self.dx + self.offset - self.z0) * self.W0))
                if shift_at is not None and tip > shift_at * self.nz:
                    self._shift(int(0.25 * self.nz))
                if verbose and s % (record_every * 20) == 0:
                    print(f"  step {s}/{n}: t = {self.t * self.tau0 * 1e6:.2f} us, tip at "
                          f"{self.front[-1][1] * 1e6:.2f} um", flush=True)
        solid_c = np.concatenate(self.solid_rows, axis=0) if self.solid_rows else np.zeros((0, self.nx))
        return PFResult(asnumpy(self.phi), asnumpy(self.composition()), solid_c, self.dx * self.W0, self.t * self.tau0, n,
                        time.perf_counter() - t0, self.front,
                        dict(G=self.G, V=self.V, W0_m=self.W0, tau0_s=self.tau0, lambda_=self.lam, dt=self.dt,
                             nx=self.nx, nz=self.nz, lT_m=self.lT * self.W0, d0_m=self.alloy.d0,
                             diffusion_length_m=self.alloy.D / self.V, tilt_deg=float(np.degrees(self.theta0))))


# ----------------------------------------------------------------------------------------
# Optional fused CPU kernel (numba). Same discretisation as DirectionalPF.step (NumPy/CuPy),
# one pass for d_t phi and one for the solute update, threaded over rows.
_KERNELS = {}


def _numba_kernels():
    if "step" in _KERNELS:
        return _KERNELS["step"]
    import numba

    @numba.njit(inline="always", fastmath=False)
    def aniso(px, pz, eps, c0, s0):
        u = c0 * px + s0 * pz
        w = c0 * pz - s0 * px
        u2 = u * u
        w2 = w * w
        g2 = u2 + w2
        inv = 1.0 / (g2 * g2 + 1e-24)
        return 1.0 + eps * (u2 * u2 - 6.0 * u2 * w2 + w2 * w2) * inv, -16.0 * eps * u * w * (u2 - w2) * inv, g2

    @numba.njit(parallel=True, cache=True)
    def step(phi, U, R, phin, Un, zc, offset, z0, Vt, t, lT, k, lam, Dt, at, dx, dt, eps, c0, s0):
        nz, nx = phi.shape
        for i in numba.prange(nz):
            ip = min(i + 1, nz - 1)
            im = max(i - 1, 0)
            theta = (zc[i] + offset - z0 - Vt * t) / lT
            tf = max(1.0 - (1.0 - k) * theta, k)
            for j in range(nx):
                jr = j + 1 if j + 1 < nx else 0
                jl = j - 1 if j > 0 else nx - 1
                # east and west x-faces
                px = (phi[i, jr] - phi[i, j]) / dx
                pz = (phi[ip, j] + phi[ip, jr] - phi[im, j] - phi[im, jr]) / (4.0 * dx)
                a, ap, g2 = aniso(px, pz, eps, c0, s0)
                fe = a * a * px - a * ap * pz
                px = (phi[i, j] - phi[i, jl]) / dx
                pz = (phi[ip, jl] + phi[ip, j] - phi[im, jl] - phi[im, j]) / (4.0 * dx)
                a, ap, g2 = aniso(px, pz, eps, c0, s0)
                fw = a * a * px - a * ap * pz
                # north and south z-faces (none through the top and bottom)
                fn = 0.0
                if i < nz - 1:
                    pz = (phi[ip, j] - phi[i, j]) / dx
                    px = (phi[i, jr] + phi[ip, jr] - phi[i, jl] - phi[ip, jl]) / (4.0 * dx)
                    a, ap, g2 = aniso(px, pz, eps, c0, s0)
                    fn = a * a * pz + a * ap * px
                fs = 0.0
                if i > 0:
                    pz = (phi[i, j] - phi[im, j]) / dx
                    px = (phi[im, jr] + phi[i, jr] - phi[im, jl] - phi[i, jl]) / (4.0 * dx)
                    a, ap, g2 = aniso(px, pz, eps, c0, s0)
                    fs = a * a * pz + a * ap * px
                a, ap, g2 = aniso((phi[i, jr] - phi[i, jl]) / (2.0 * dx), (phi[ip, j] - phi[im, j]) / (2.0 * dx),
                                  eps, c0, s0)
                p = phi[i, j]
                g = 1.0 - p * p
                R[i, j] = ((fe - fw + fn - fs) / dx + p - p * p * p - lam * g * g * (U[i, j] + theta)) / (a * a * tf)
        for i in numba.prange(nz):
            ip = min(i + 1, nz - 1)
            im = max(i - 1, 0)
            for j in range(nx):
                jr = j + 1 if j + 1 < nx else 0
                jl = j - 1 if j > 0 else nx - 1
                Fc = 1.0 + (1.0 - k) * U[i, j]
                # east face
                px = (phi[i, jr] - phi[i, j]) / dx
                pz = (phi[ip, j] + phi[ip, jr] - phi[im, j] - phi[im, jr]) / (4.0 * dx)
                je = (Dt * 0.25 * (2.0 - phi[i, j] - phi[i, jr]) * (U[i, jr] - U[i, j]) / dx
                      + at * 0.25 * (Fc + 1.0 + (1.0 - k) * U[i, jr]) * (R[i, j] + R[i, jr]) * px
                      / np.sqrt(px * px + pz * pz + 1e-24))
                # west face
                px = (phi[i, j] - phi[i, jl]) / dx
                pz = (phi[ip, jl] + phi[ip, j] - phi[im, jl] - phi[im, j]) / (4.0 * dx)
                jw = (Dt * 0.25 * (2.0 - phi[i, jl] - phi[i, j]) * (U[i, j] - U[i, jl]) / dx
                      + at * 0.25 * (1.0 + (1.0 - k) * U[i, jl] + Fc) * (R[i, jl] + R[i, j]) * px
                      / np.sqrt(px * px + pz * pz + 1e-24))
                jn = 0.0
                if i < nz - 1:
                    pz = (phi[ip, j] - phi[i, j]) / dx
                    px = (phi[i, jr] + phi[ip, jr] - phi[i, jl] - phi[ip, jl]) / (4.0 * dx)
                    jn = (Dt * 0.25 * (2.0 - phi[i, j] - phi[ip, j]) * (U[ip, j] - U[i, j]) / dx
                          + at * 0.25 * (Fc + 1.0 + (1.0 - k) * U[ip, j]) * (R[i, j] + R[ip, j]) * pz
                          / np.sqrt(px * px + pz * pz + 1e-24))
                js = 0.0
                if i > 0:
                    pz = (phi[i, j] - phi[im, j]) / dx
                    px = (phi[im, jr] + phi[i, jr] - phi[im, jl] - phi[i, jl]) / (4.0 * dx)
                    js = (Dt * 0.25 * (2.0 - phi[im, j] - phi[i, j]) * (U[i, j] - U[im, j]) / dx
                          + at * 0.25 * (1.0 + (1.0 - k) * U[im, j] + Fc) * (R[im, j] + R[i, j]) * pz
                          / np.sqrt(px * px + pz * pz + 1e-24))
                pn = phi[i, j] + dt * R[i, j]
                phin[i, j] = pn
                Un[i, j] = U[i, j] + dt * ((je - jw + jn - js) / dx + 0.5 * Fc * R[i, j]) / ((1.0 + k - (1.0 - k) * pn) / 2.0)

    _KERNELS["step"] = step
    return step


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except ImportError:
        return False


def constitutional_velocity(alloy: DiluteAlloy, G: float):
    """Planar-front stability limit V_cs = D G / dT0 [m/s] (constitutional supercooling)."""
    return alloy.D * G / alloy.freezing_range


def absolute_stability_velocity(alloy: DiluteAlloy):
    """V_abs = D dT0 / (k Gamma) [m/s] (Mullins-Sekerka absolute stability)."""
    return alloy.D * alloy.freezing_range / (alloy.k * alloy.gibbs_thomson)


def mullins_sekerka_rate(alloy: DiluteAlloy, G: float, V: float, q, stiffness: float = 1.0):
    """Growth rate omega [1/s] of a planar-front perturbation of wavenumber ``q`` [1/m] in the
    sharp-interface limit the phase-field model reproduces (one-sided diffusion, frozen
    temperature, local equilibrium, no kinetic undercooling); full (non-quasi-static)
    relation, e.g. Warren & Langer, PRE 47 (1993) 2702:

        omega + k V^2/D + (G/dT0 + S d0 q^2 - V/D) (D q_l - (1 - k) V) = 0,
        q_l = V/(2D) + sqrt((V/2D)^2 + q^2 + omega/D),

    solved in closed form: with ``s = q_l - V/(2D)`` the relation is a quadratic in ``s``;
    of its roots with ``Re s >= 0`` (bounded diffusion field) the one with the largest growth
    rate is returned as ``Re omega`` (``S = stiffness``: surface-stiffness factor, ``1 - 15 eps4`` for a
    <10> growth axis). NaN if neither root is admissible."""
    q = np.asarray(q, dtype=float)
    D, k = alloy.D, alloy.k
    B = G / alloy.freezing_range + stiffness * alloy.d0 * q * q - V / D
    C = -V * V / (4.0 * D) - D * q * q + k * V * V / D + B * V * (k - 0.5)
    root = np.sqrt((B * B - 4.0 * C / D).astype(complex))
    best = np.full(q.shape, -np.inf)
    for sq in (0.5 * (-B + root), 0.5 * (-B - root)):
        w = (D * (sq * sq - (V / (2.0 * D)) ** 2 - q * q)).real
        best = np.where(sq.real >= 0.0, np.maximum(best, w), best)
    return np.where(np.isfinite(best), best, np.nan)


def front_profile(phi: np.ndarray, dx: float):
    """Height of the phi = 0 level in each column [same unit as dx], linearly interpolated
    at the highest solid-liquid crossing; z of cell i is (i + 0.5) dx. A column with no solid
    (a liquid groove reaching the window bottom) gets 0."""
    phi = np.asarray(phi)
    nz, nx = phi.shape
    solid = phi > 0.0
    has = solid.any(axis=0)
    top = nz - 1 - np.argmax(solid[::-1], axis=0)                       # highest solid cell per column
    top = np.clip(top, 0, nz - 2)
    j = np.arange(nx)
    p0, p1 = phi[top, j], phi[top + 1, j]
    frac = np.where(p0 != p1, p0 / (p0 - p1 + 1e-300), 0.0)
    return np.where(has, (top + 0.5 + np.clip(frac, 0.0, 1.0)) * dx, 0.0)   # all-liquid column: 0
