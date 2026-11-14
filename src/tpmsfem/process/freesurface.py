"""Free-surface lattice Boltzmann melt-pool model (powder-scale LPBF, stage 1).

The melt is a free-surface flow on a D3Q19 lattice; the gas above it is not simulated
(Koerner et al., J. Mater. Process. Technol. 168 (2005) 263; Thuerey, PhD thesis,
Erlangen 2007):

* each cell is GAS, INTERFACE (partly filled, fill level eps = m / rho), LIQUID or WALL;
* interface cells exchange mass with their neighbours from the streamed populations;
  populations that would come from the gas are rebuilt from the gas pressure, which carries
  the Laplace pressure sigma * kappa (curvature from the fill-level field);
* cells fill (m > rho) or empty (m < 0) and the interface layer moves; the excess mass goes
  to the neighbouring interface cells, so the metal mass is conserved to round-off;
* collision: two-relaxation-time (TRT) with Guo forcing, so that the very low viscosity
  of liquid metal (tau ~ 0.51) stays stable;
* melting and solidification act on the flow through a Carman-Kozeny (Darcy) damping
  -C (1 - f_l)^2 / (f_l^3 + b) rho u, applied semi-implicitly so that a solid cell holds
  still for any C. Unmelted metal is therefore part of the lattice and the plate,
  the powder and the melt share one set of cells.

Heat: volumetric enthalpy on the same cells (metal only; finite volume, explicit), with
conduction, first-order upwind advection by the lattice velocity, and the laser absorbed
below the local surface height with an exponential depth (the same discretisation as
:class:`tpmsfem.process.MeltPoolModel`, so the two can be compared).

Stage 1 contains no Marangoni stress, recoil pressure or evaporation; those come next.

Arrays live on the active backend (NumPy or CuPy): the same code runs on a GPU.
Units: SI for the physical interface; the lattice works in lattice units
(dx = dt = 1, reference density 1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..backend import asnumpy, get_xp

GAS, INTERFACE, LIQUID, WALL = 0, 1, 2, 3

# D3Q19: rest, 6 faces, 12 edges
C = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
              [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], dtype=np.int64)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
Q = 19


def _roll(xp, a, c):
    """Value at x - c (pull from the upstream neighbour along c)."""
    return xp.roll(a, shift=(int(c[0]), int(c[1]), int(c[2])), axis=(0, 1, 2))


def _nb(xp, a, c):
    """Value at the neighbour x + c."""
    return xp.roll(a, shift=(-int(c[0]), -int(c[1]), -int(c[2])), axis=(0, 1, 2))


class FreeSurfaceLBM:
    """Free-surface D3Q19 TRT lattice Boltzmann solver in lattice units.

    ``flag``: int8 array (nx, ny, nz) with GAS / INTERFACE / LIQUID / WALL; ``fill``: fill
    level of interface cells (1 for liquid). Domain faces should be WALL or GAS cells: the
    shifts are periodic, and only cells next to a wall see across it.
    """

    def __init__(self, flag, fill=None, tau: float = 0.6, sigma: float = 0.0, gravity=(0.0, 0.0, 0.0),
                 magic: float = 3.0 / 16.0, rho_gas: float = 1.0, dtype=np.float64, fill_offset: float = 1e-3):
        self.xp = xp = get_xp()
        self.dtype = dtype
        flag = np.asarray(flag, dtype=np.int8)
        self.shape = flag.shape
        self.flag = xp.asarray(flag)
        fill = np.where(flag == LIQUID, 1.0, 0.0) if fill is None else np.asarray(fill, float)
        fill = np.where(flag == LIQUID, 1.0, np.where(flag == INTERFACE, fill, 0.0))
        self.rho = xp.ones(self.shape, dtype=dtype)
        self.u = xp.zeros((3,) + self.shape, dtype=dtype)
        self.mass = xp.asarray(fill, dtype=dtype)
        self.f = self._feq(self.rho, self.u)
        self.set_tau(tau, magic)
        self.sigma = float(sigma)
        self.g = np.asarray(gravity, float)
        self.rho_gas = float(rho_gas)
        self.off = float(fill_offset)
        self.force = None              # extra body force (3, nx, ny, nz), lattice units, set by a coupler
        self.damping = None            # Darcy coefficient K (lattice units, per step), set by a coupler
        self.gas_pressure = None       # extra gas density (pressure * 3) on interface cells (recoil)
        self.solid = None              # bool mask of solid (unmelted) cells, set by a coupler: two
                                       # touching solid cells exchange no mass
        self.lost_mass = 0.0
        self.kappa = xp.zeros(self.shape, dtype=dtype)
        self.smooth = 1                # smoothing passes for the normal and curvature
        self.u_limit = 0.25            # lattice speed limit (Mach ~0.43); 0 or None: off
        self.limited = 0               # most cells limited in one step
        self.curvature_method = "gradient"  # or "height": height function (plain, no consistency checks yet: noisier here)
        self.rebuild_normal = False    # also rebuild populations arriving from the gas side (Koerner); off: no gain here
        self.steps = 0

    # ------------------------------------------------------------------ basics
    def set_tau(self, tau, magic=3.0 / 16.0):
        self.tau = float(tau)
        self.w_plus = 1.0 / self.tau
        lam = magic / (self.tau - 0.5)
        self.w_minus = 1.0 / (lam + 0.5)

    @property
    def nu(self):
        return (self.tau - 0.5) / 3.0

    def _feq(self, rho, u):
        xp = self.xp
        usq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
        f = xp.empty((Q,) + self.shape, dtype=self.dtype)
        for i in range(Q):
            cu = C[i, 0] * u[0] + C[i, 1] * u[1] + C[i, 2] * u[2]
            f[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        return f

    def _feq_i(self, i, rho, u):
        cu = C[i, 0] * u[0] + C[i, 1] * u[1] + C[i, 2] * u[2]
        usq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
        return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

    @property
    def fill(self):
        xp = self.xp
        eps = xp.where(self.flag == LIQUID, 1.0, xp.where(self.flag == INTERFACE, self.mass / self.rho, 0.0))
        return xp.clip(eps, 0.0, 1.0)

    def total_mass(self):
        xp = self.xp
        fluid = (self.flag == LIQUID) | (self.flag == INTERFACE)
        return float(xp.sum(xp.where(fluid, self.mass, 0.0))) + self.lost_mass

    # ------------------------------------------------------------------ curvature
    def _shift_mirror(self, a, ax, d, wall):
        """Neighbour value along axis ``ax`` (d = +1 / -1); across a wall the cell's own value
        (a zero-gradient wall: 90 degree contact angle)."""
        xp = self.xp
        nb = xp.roll(a, -d, ax)
        return xp.where(xp.roll(wall, -d, ax), a, nb)

    def normal_curvature(self, smooth: int = 1, eps=None):
        """Interface normal (unit, pointing into the liquid) and mean curvature (sum of the
        principal curvatures, positive for a convex liquid) from the fill-level field,
        smoothed ``smooth`` times with a [1 2 1] filter per axis. Walls act as mirrors."""
        xp = self.xp
        wall = self.flag == WALL
        e = self.fill if eps is None else eps
        sm = self._shift_mirror
        for _ in range(smooth):
            for ax in range(3):
                e = 0.25 * (sm(e, ax, -1, wall) + 2.0 * e + sm(e, ax, 1, wall))
        g = [0.5 * (sm(e, ax, 1, wall) - sm(e, ax, -1, wall)) for ax in range(3)]
        norm = xp.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-12
        n = [gi / norm for gi in g]
        div = sum(0.5 * (sm(n[ax], ax, 1, wall) - sm(n[ax], ax, -1, wall)) for ax in range(3))
        return xp.stack(n), -div

    def curvature(self):
        return self.normal_curvature()[1]

    def height_function_curvature(self, normal=None, half: int = 3):
        """Mean curvature by the height-function method (Cummins, Francois & Kothe 2005): the
        fill levels are summed along the axis closest to the interface normal over a
        (2 half + 1)-cell column, giving local surface heights on a 3 x 3 stencil, and
        kappa = -div(grad h / sqrt(1 + |grad h|^2)). Second-order accurate, unlike the
        smoothed-gradient estimate."""
        xp = self.xp
        e = self.fill
        if normal is None:
            normal = self.normal_curvature(1)[0]
        kap = xp.zeros(self.shape, dtype=self.dtype)
        best = xp.argmax(xp.abs(normal), axis=0)
        for a in range(3):
            S = sum(xp.roll(e, k, a) for k in range(-half, half + 1))
            b, c = [ax for ax in range(3) if ax != a]
            Sp_b, Sm_b = xp.roll(S, -1, b), xp.roll(S, 1, b)
            Sp_c, Sm_c = xp.roll(S, -1, c), xp.roll(S, 1, c)
            hb, hc = 0.5 * (Sp_b - Sm_b), 0.5 * (Sp_c - Sm_c)
            hbb, hcc = Sp_b - 2.0 * S + Sm_b, Sp_c - 2.0 * S + Sm_c
            hbc = 0.25 * (xp.roll(Sp_b, -1, c) - xp.roll(Sp_b, 1, c) - xp.roll(Sm_b, -1, c) + xp.roll(Sm_b, 1, c))
            k_a = -(hbb * (1.0 + hc * hc) + hcc * (1.0 + hb * hb) - 2.0 * hb * hc * hbc) / (1.0 + hb * hb + hc * hc) ** 1.5
            kap = xp.where(best == a, k_a, kap)
        return kap

    # ------------------------------------------------------------------ one step
    def step(self):
        xp = self.xp
        flag, f = self.flag, self.f
        fluid = (flag == LIQUID) | (flag == INTERFACE)
        itf = flag == INTERFACE
        # gas density at the interface: Laplace pressure (+ recoil, set by a coupler)
        nrm, kap = self.normal_curvature(self.smooth)
        if self.sigma != 0.0 and self.curvature_method == "height":
            kap = self.height_function_curvature(nrm)
        if self.sigma != 0.0:
            self.kappa = xp.clip(kap, -0.5, 0.5)
            rho_g = self.rho_gas + 3.0 * self.sigma * self.kappa
        else:
            rho_g = xp.full(self.shape, self.rho_gas, dtype=self.dtype)
        if self.gas_pressure is not None:
            rho_g = rho_g + self.gas_pressure
        u = self.u
        eps = self.fill
        # streaming (pull), wall bounce-back, gas reconstruction, mass exchange
        fs = xp.empty_like(f)
        dm = xp.zeros(self.shape, dtype=self.dtype)
        for i in range(Q):
            c = C[i]
            fi = _roll(xp, f[i], c)
            if i == 0:
                fs[0] = fi
                continue
            src = _roll(xp, flag, c)
            o = OPP[i]
            fi = xp.where(src == WALL, f[o], fi)
            ex = fi - f[o]                                   # mass exchange uses the streamed values
            # rebuild what comes from the gas, and (Koerner 2005) everything arriving from the
            # gas side of the interface, so the gas pressure acts evenly on a staircase surface
            cn = C[i, 0] * nrm[0] + C[i, 1] * nrm[1] + C[i, 2] * nrm[2]
            rebuild = itf & ((src == GAS) | ((cn > 0.0) & (src != WALL) & self.rebuild_normal))
            if bool(rebuild.any()):
                rec = self._feq_i(i, rho_g, u) + self._feq_i(o, rho_g, u) - f[o]
                fi = xp.where(rebuild, rec, fi)
            fs[i] = fi
            # mass exchange of interface cells with liquid / interface neighbours
            src_eps = _roll(xp, eps, c)
            wgt = xp.where(src == LIQUID, 1.0, xp.where(src == INTERFACE, 0.5 * (eps + src_eps), 0.0))
            if self.solid is not None:                 # touching solid interface cells (cold powder): nothing
                wgt = xp.where(self.solid & _roll(xp, self.solid, c) & (src == INTERFACE), 0.0, wgt)
            dm += xp.where(itf, wgt * ex, 0.0)
        self.mass = xp.where(itf, self.mass + dm, self.mass)
        # macroscopic fields
        rho = fs.sum(axis=0)
        rho = xp.where(fluid, rho, 1.0)
        j = xp.stack([sum(float(C[i, a]) * fs[i] for i in range(Q) if C[i, a] != 0) for a in range(3)])
        F = xp.zeros((3,) + self.shape, dtype=self.dtype)
        for a in range(3):
            if self.g[a] != 0.0:
                F[a] = rho * self.g[a]
        if self.force is not None:
            F = F + self.force
        K = self.damping if self.damping is not None else 0.0
        u = (j + 0.5 * F) / rho / (1.0 + 0.5 * K)
        if self.u_limit:
            # speed limiter (lattice Mach number): keeps violent recoil-driven spots stable;
            # counts how many cells it touched
            sp = xp.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            over = sp > self.u_limit
            if bool(over.any()):
                self.limited = max(self.limited, int(xp.sum(over & fluid)))
                u = u * xp.where(over, self.u_limit / xp.maximum(sp, 1e-30), 1.0)
        F = F - K * rho * u
        u = xp.where(fluid, u, 0.0)
        F = xp.where(fluid, F, 0.0)
        # TRT collision with Guo forcing
        feq = self._feq(rho, u)
        wp, wm = self.w_plus, self.w_minus
        uF = u[0] * F[0] + u[1] * F[1] + u[2] * F[2]
        out = xp.empty_like(fs)
        for i in range(Q):
            o = OPP[i]
            fp, fm = 0.5 * (fs[i] + fs[o]), 0.5 * (fs[i] - fs[o])
            ep, em = 0.5 * (feq[i] + feq[o]), 0.5 * (feq[i] - feq[o])
            cF = C[i, 0] * F[0] + C[i, 1] * F[1] + C[i, 2] * F[2]
            cu = C[i, 0] * u[0] + C[i, 1] * u[1] + C[i, 2] * u[2]
            Sp = W[i] * (9.0 * cu * cF - 3.0 * uF)
            Sm = W[i] * 3.0 * cF
            out[i] = fs[i] - wp * (fp - ep) - wm * (fm - em) + (1.0 - 0.5 * wp) * Sp + (1.0 - 0.5 * wm) * Sm
        self.f = xp.where(fluid[None], out, fs)
        self.rho, self.u = rho, u
        self.mass = xp.where(flag == LIQUID, rho, self.mass)
        self._convert()
        self.steps += 1

    # ------------------------------------------------------------------ interface motion
    def _any_nb(self, mask):
        xp = self.xp
        out = xp.zeros(self.shape, dtype=bool)
        for i in range(1, Q):
            out |= _nb(xp, mask, C[i])
        return out

    def _distribute(self, excess, src_mask, recv_mask):
        """Spread ``excess`` of the cells in ``src_mask`` equally over their neighbours in
        ``recv_mask``; what has no receiver is spread over all interface cells."""
        xp = self.xp
        ex = xp.where(src_mask, excess, 0.0)
        n = xp.zeros(self.shape, dtype=self.dtype)
        rcv = recv_mask.astype(self.dtype)
        for i in range(1, Q):
            n += _nb(xp, rcv, C[i])
        share = xp.where(n > 0, ex / xp.maximum(n, 1.0), 0.0)
        got = xp.zeros(self.shape, dtype=self.dtype)
        for i in range(1, Q):
            got += _nb(xp, share, C[i])
        self.mass = self.mass + xp.where(recv_mask, got, 0.0)
        orphan = float(xp.sum(xp.where(n > 0, 0.0, ex)))
        if orphan != 0.0:
            n_itf = int(xp.sum(recv_mask))
            if n_itf:
                self.mass = self.mass + xp.where(recv_mask, orphan / n_itf, 0.0)
            else:
                self.lost_mass += orphan

    def _convert(self):
        xp = self.xp
        flag, rho, m = self.flag, self.rho, self.mass
        itf = flag == INTERFACE
        fill = itf & (m > (1.0 + self.off) * rho)
        empty = itf & (m < -self.off * rho) & ~fill
        if not bool(fill.any()) and not bool(empty.any()):
            self._close_layer()
            return
        excess_fill = m - rho
        excess_empty = m.copy()
        # filling: gas neighbours become interface, initialised from their fluid neighbours
        new_itf = (flag == GAS) & self._any_nb(fill)
        if bool(new_itf.any()):
            fluid = (flag == LIQUID) | (flag == INTERFACE)
            s_rho = xp.zeros(self.shape, dtype=self.dtype)
            s_u = xp.zeros((3,) + self.shape, dtype=self.dtype)
            cnt = xp.zeros(self.shape, dtype=self.dtype)
            fl = fluid.astype(self.dtype)
            for i in range(1, Q):
                c = C[i]
                w = _nb(xp, fl, c)
                cnt += w
                s_rho += w * _nb(xp, rho, c)
                for a in range(3):
                    s_u[a] += w * _nb(xp, self.u[a], c)
            r0 = xp.where(cnt > 0, s_rho / xp.maximum(cnt, 1.0), 1.0)
            u0 = s_u / xp.maximum(cnt, 1.0)
            feq = self._feq(r0, u0)
            self.f = xp.where(new_itf[None], feq, self.f)
            self.rho = xp.where(new_itf, r0, self.rho)
            self.u = xp.where(new_itf[None], u0, self.u)
            self.mass = xp.where(new_itf, 0.0, self.mass)
            flag = xp.where(new_itf, INTERFACE, flag).astype(np.int8)
        flag = xp.where(fill, LIQUID, flag).astype(np.int8)
        # emptying: liquid neighbours become interface (full)
        new_from_liq = (flag == LIQUID) & self._any_nb(empty)
        flag = xp.where(new_from_liq, INTERFACE, flag).astype(np.int8)
        flag = xp.where(empty, GAS, flag).astype(np.int8)
        self.flag = flag
        self.mass = xp.where(fill, self.rho, self.mass)
        self.mass = xp.where(empty, 0.0, self.mass)
        itf_now = self.flag == INTERFACE
        self._distribute(excess_fill, fill, itf_now)
        self._distribute(excess_empty, empty, itf_now)
        self._close_layer()

    def _close_layer(self):
        """Keep the interface closed: liquid cells next to gas become (full) interface
        cells; interface cells without gas neighbours become liquid."""
        xp = self.xp
        gas = self.flag == GAS
        liq_gas = (self.flag == LIQUID) & self._any_nb(gas)
        if bool(liq_gas.any()):
            self.flag = xp.where(liq_gas, INTERFACE, self.flag).astype(np.int8)
        itf = self.flag == INTERFACE
        no_gas = itf & ~self._any_nb(self.flag == GAS)
        if bool(no_gas.any()):
            excess = self.mass - self.rho
            self.flag = xp.where(no_gas, LIQUID, self.flag).astype(np.int8)
            self.mass = xp.where(no_gas, self.rho, self.mass)
            self._distribute(excess, no_gas, self.flag == INTERFACE)


# ====================================================================== LPBF coupling
@dataclass
class LBMResult:
    dx: float
    dt: float
    T: np.ndarray
    T_peak: np.ndarray
    flag: np.ndarray
    fill: np.ndarray
    surface: np.ndarray            # surface height per column [m] (metal fill)
    u_max: float                   # m/s
    mass_error: float              # relative
    energy_balance: float          # (dE - absorbed + lost) / absorbed
    absorbed: float                # J
    steps: int
    seconds: float
    mlups: float


class FreeSurfaceMeltPool:
    """Laser melting of a metal plate (optionally with powder) on the free-surface lattice.

    The domain is ``nx * ny * nz`` cells of size ``dx`` [m]; ``metal`` is a boolean array
    (True where there is metal at the start: plate and powder particles), with gas above.
    ``material``: :class:`tpmsfem.fem.materials.Material` with ``ThermalProperties`` (tpmsfem
    units; converted to SI here). ``viscosity`` [Pa s], ``sigma`` [N/m] surface tension.
    ``dt`` [s] sets the lattice time step; tau follows from the viscosity.
    ``mushy_C`` [1/s]: Carman-Kozeny constant of the Darcy damping.
    """

    def __init__(self, material, dx: float, metal, *, dt: float = 1e-7, viscosity: float = 5e-3,
                 sigma: float = 1.8, mushy_C: float = 1e9, T0: float = 25.0, T_max: float = 6000.0,
                 dtype=np.float64, magic: float = 3.0 / 16.0, smooth_surface_tension: bool = True,
                 dsigma_dT: float = 0.0, evaporation=None, recoil_factor: float = 0.54, emissivity: float = 0.0,
                 T_ambient: float = 25.0, gravity=(0.0, 0.0, -9.81)):
        th = material.thermal
        self.xp = xp = get_xp()
        self.dtype = dtype
        self.dx, self.dt = float(dx), float(dt)
        metal = np.asarray(metal)
        self.shape = metal.shape
        wall = np.zeros(self.shape, bool)
        wall[0, :, :] = wall[-1, :, :] = wall[:, 0, :] = wall[:, -1, :] = wall[:, :, 0] = wall[:, :, -1] = True
        if metal.dtype == bool:
            fill = metal.astype(float)
        else:                                       # fill levels, e.g. from process.powder.voxelise
            fill = np.clip(metal.astype(float), 0.0, 1.0)
            fill[fill < 0.02] = 0.0
        fill[wall] = 0.0
        metal = fill > 0.0
        # SI properties
        self.rho0 = th.rho * 1e12
        tt, HH = th.enthalpy_table(T_lo=T0 - 50.0, T_hi=T_max, dT=1.0)
        self._tt = xp.asarray(tt, dtype=dtype)
        self._HH = xp.asarray(HH * 1e6, dtype=dtype)                    # mJ/mm^3 -> J/m^3
        self._kT = xp.asarray(th.conductivity(tt), dtype=dtype)
        self.Ts, self.Tl = float(th.T_solidus), float(th.T_liquidus)
        self.T0 = float(T0)
        rc = self.rho0 * np.interp(tt, th.T, th.cp) * 1e-6
        alpha = th.conductivity(tt) / rc
        dt_heat = dx * dx / (6.0 * alpha.max())
        if self.dt > 0.9 * dt_heat:
            raise ValueError(f"dt = {dt:.2e} s exceeds the conduction limit {0.9 * dt_heat:.2e} s")
        # lattice
        nu = viscosity / self.rho0
        tau = 0.5 + 3.0 * nu * dt / dx ** 2
        sigma_lat = sigma * dt ** 2 / (self.rho0 * dx ** 3)
        flag = np.full(self.shape, GAS, np.int8)
        flag[metal] = LIQUID
        gas = ~metal & ~wall
        # partly filled cells, and full cells next to gas, are interface cells
        nbgas = np.zeros(self.shape, bool)
        for c in C[1:]:
            nbgas |= np.roll(gas, shift=(-c[0], -c[1], -c[2]), axis=(0, 1, 2))
        flag[metal & (nbgas | (fill < 1.0))] = INTERFACE
        flag[wall] = WALL
        # partly filled cells with no gas around them (necks between touching particles) are
        # filled: a voxelisation detail, not worth a mass redistribution on the first step
        enclosed = (flag == INTERFACE) & ~nbgas
        flag[enclosed] = LIQUID
        fill[enclosed] = 1.0
        g_lat = tuple(float(g) * dt * dt / dx for g in gravity)
        self.lbm = FreeSurfaceLBM(flag, fill, tau=tau, sigma=sigma_lat if smooth_surface_tension else 0.0,
                                  magic=magic, dtype=dtype, gravity=g_lat)
        self.mushy_C = float(mushy_C)
        self.tau, self.sigma_lat = tau, sigma_lat
        # heat
        self.H = xp.full(self.shape, float(np.interp(T0, tt, HH * 1e6)), dtype=dtype)
        self.T = xp.full(self.shape, T0, dtype=dtype)
        self.z = (np.arange(self.shape[2]) + 0.5) * dx
        self.x = (np.arange(self.shape[0]) + 0.5) * dx
        self.y = (np.arange(self.shape[1]) + 0.5) * dx
        self.t = 0.0
        self.H_ref = float(np.interp(T0, tt, HH * 1e6))
        # surface physics (stage 2): Marangoni stress, recoil pressure, evaporation and radiation
        self.dsigma_dT = float(dsigma_dT)
        self.evaporation = evaporation
        self.recoil_factor = float(recoil_factor)
        self.emissivity = float(emissivity)
        self.T_amb = float(T_ambient)
        self.lost = 0.0                       # J, through the surface
        self.heat_weight = "mass"           # lattice mass (conserves heat through mass exchange) or "fill"
        self.recoil_cap = 0.1                 # largest recoil density jump on the lattice (3 p dt^2 / rho dx^2)
        self.recoil_clipped = 0               # most interface cells clipped in one step
        # heat-capacity floor of nearly empty interface cells: low enough to hold their heat,
        # high enough for the explicit conduction step to stay stable (6 alpha dt / dx^2 / cap < 1)
        self.eps_min = max(0.1, 1.25 * 6.0 * float(alpha.max()) * self.dt / dx ** 2)

    # ------------------------------------------------------------------ helpers
    def _T_of_H(self, H):
        return self.xp.interp(H.ravel(), self._HH, self._tt).reshape(H.shape)

    def metal(self):
        f = self.lbm.flag
        return (f == LIQUID) | (f == INTERFACE)

    def surface_height(self):
        """Height [m] of the first metal surface seen from above, per column: the bottom of
        the topmost cell holding metal plus its fill level (z = 0 is the bottom of the
        domain, i.e. of the wall layer). Powder gaps below it are not counted."""
        xp, dx = self.xp, self.dx
        eps = self.lbm.fill
        nz = self.shape[2]
        has = eps > 1e-6
        k_top = nz - 1 - xp.argmax(has[:, :, ::-1], axis=2)                       # topmost cell with metal
        any_m = has.any(axis=2)
        f_top = xp.take_along_axis(eps, k_top[:, :, None], axis=2)[:, :, 0]
        return xp.where(any_m, (k_top + f_top) * dx, 0.0)

    def _laser_source(self, x_l, y_l, power, absorptivity, radius, depth):
        """Absorbed power density [W/m^3], exactly A*P in total. The beam comes straight down
        and is absorbed along each column by Beer-Lambert over the *metal* it crosses (depth
        ``depth`` of metal): each cell takes exp(-s/d) - exp(-(s + eps dx)/d), with s the
        metal above it and eps its fill level. Gaps between powder particles let the light
        through, and a thin sliver of metal takes only its share."""
        xp, dx = self.xp, self.dx
        X, Y = xp.meshgrid(xp.asarray(self.x), xp.asarray(self.y), indexing="ij")
        f = xp.exp(-2.0 * ((X - x_l) ** 2 + (Y - y_l) ** 2) / radius ** 2)
        f = f / (f.sum() * dx * dx)
        qs = absorptivity * power * f                                         # W/m^2
        eps = xp.where(self.metal(), self.lbm.fill, 0.0)
        d = depth if depth > 0 else 0.25 * dx
        below_top = xp.cumsum(eps[:, :, ::-1], axis=2)[:, :, ::-1] * dx           # metal from the top down to the cell bottom
        above = below_top - eps * dx                                           # metal above the cell
        wz = (xp.exp(-above / d) - xp.exp(-below_top / d)) / dx
        norm = wz.sum(axis=2, keepdims=True) * dx
        wz = xp.where(norm > 0, wz / xp.maximum(norm, 1e-30), 0.0)
        return qs[:, :, None] * wz

    def energy(self):
        """Thermal energy of the metal [J] above the initial state: each cell weighted by its
        metal mass (lattice mass, 1 for a full cell at the reference density)."""
        xp = self.xp
        return float(xp.sum(xp.where(self.metal(), self._weight() * (self.H - self.H_ref), 0.0))) * self.dx ** 3

    def _weight(self):
        """Metal content of a cell for the heat balance: its fill level (volume fraction,
        ``heat_weight='fill'``) or its lattice mass (``'mass'``, which also counts the small
        lattice compressions)."""
        return self.lbm.fill if self.heat_weight == "fill" else self.lbm.mass

    def _heat_step(self, q, loss=None):
        """Advance the enthalpy one step. The heat capacity of a cell is its metal mass
        (floored at ``eps_min`` so that nearly empty interface cells stay stable). Returns the
        energy [J] that the source ``q`` and the surface ``loss`` [W/m^3] put in and took out."""
        xp, dx, dt = self.xp, self.dx, self.dt
        T, H = self.T, self.H
        m = self.metal()
        k = xp.interp(T.ravel(), self._tt, self._kT).reshape(T.shape)
        u = self.lbm.u * (dx / dt)                                            # m/s
        div = xp.zeros(self.shape, dtype=self.dtype)
        for ax in range(3):
            a_ = [slice(None)] * 3
            b_ = [slice(None)] * 3
            a_[ax], b_[ax] = slice(None, -1), slice(1, None)
            a_, b_ = tuple(a_), tuple(b_)
            both = m[a_] & m[b_]
            ka, kb = k[a_], k[b_]
            kf = 2.0 * ka * kb / (ka + kb)
            flux = xp.where(both, kf * (T[b_] - T[a_]) / dx, 0.0)              # W/m^2, + in +axis... (into a)
            # advection (upwind), face velocity = mean of the two cells
            uf = 0.5 * (u[ax][a_] + u[ax][b_])
            adv = xp.where(both, uf * xp.where(uf > 0, H[a_], H[b_]), 0.0)
            flux = flux - adv
            div[a_] += flux / dx
            div[b_] -= flux / dx
        mass = xp.where(m, self._weight(), 0.0)
        cap = xp.maximum(mass, self.eps_min)
        rate = div + q - (loss if loss is not None else 0.0)
        self.H = xp.where(m, H + dt * rate / cap, H)
        self.T = xp.where(m, self._T_of_H(self.H), self.T)
        frac = mass / cap
        e_in = float(xp.sum(frac * q)) * dx ** 3 * dt if not np.isscalar(q) else 0.0
        e_out = float(xp.sum(frac * loss)) * dx ** 3 * dt if loss is not None else 0.0
        return e_in, e_out

    def _surface_terms(self):
        """Marangoni force and recoil pressure for the lattice, and the surface heat loss
        [W/m^3], with the interface spread over the cells by |grad eps| (continuum surface
        force)."""
        xp, dx, dt = self.xp, self.dx, self.dt
        m = self.metal()
        # surface area per interface cell [in dx^2]: its faces to the gas, divided by
        # |nx| + |ny| + |nz| (a staircase overstates a tilted surface by that factor);
        # walls are mirrors. A flat surface gets exactly 1 per column.
        flag = self.lbm.flag
        gas = flag == GAS
        n_arr, _ = self.lbm.normal_curvature(1)
        n = [n_arr[a] for a in range(3)]
        faces = xp.zeros(self.shape, dtype=self.dtype)
        for ax in range(3):
            faces += xp.roll(gas, -1, ax).astype(self.dtype) + xp.roll(gas, 1, ax).astype(self.dtype)
        l1 = xp.maximum(xp.abs(n[0]) + xp.abs(n[1]) + xp.abs(n[2]), 1.0)
        delta = xp.where(flag == INTERFACE, faces / l1, 0.0)
        T = self.T
        Tk = xp.where(m, T, 0.0)
        loss = None
        if self.emissivity > 0.0 or self.evaporation is not None:
            q = xp.zeros(self.shape, dtype=self.dtype)
            if self.emissivity > 0.0:
                q += self.emissivity * 5.670374419e-8 * ((T + 273.15) ** 4 - (self.T_amb + 273.15) ** 4)
            if self.evaporation is not None:
                q += self.evaporation.flux(T, xp) * 1e3                                  # mW/mm^2 -> W/m^2
            loss = xp.where(m, q * delta / dx, 0.0)
        if self.dsigma_dT != 0.0:
            gT = []
            for ax in range(3):
                up = xp.roll(m, -1, ax)
                dn = xp.roll(m, 1, ax)
                Tu = xp.where(up, xp.roll(Tk, -1, ax), T)
                Td = xp.where(dn, xp.roll(Tk, 1, ax), T)
                w = xp.where(up & dn, 0.5, 1.0)
                gT.append(w * (Tu - Td) / dx)                                             # K/m
            nT = n[0] * gT[0] + n[1] * gT[1] + n[2] * gT[2]
            Fm = [self.dsigma_dT * (gT[a] - n[a] * nT) * delta / dx for a in range(3)]    # N/m^3
            scale = dt * dt / (self.rho0 * dx)
            hot = m & (T > self.Ts)
            self.lbm.force = xp.stack([xp.where(hot, Fm[a] * scale, 0.0) for a in range(3)])
        if self.evaporation is not None and self.recoil_factor > 0.0:
            ev = self.evaporation
            Tk_ = xp.maximum(T + 273.15, 300.0)
            expo = xp.minimum(ev.L_v * ev.M / 8.314462618 * (1.0 / (ev.T_boil + 273.15) - 1.0 / Tk_), 50.0)
            p_r = self.recoil_factor * ev.p0 * xp.exp(expo)                               # Pa
            p_lat = p_r * dt * dt / (self.rho0 * dx * dx)
            rp = xp.where(self.lbm.flag == INTERFACE, 3.0 * p_lat, 0.0)
            # the lattice carries density jumps of a few percent; beyond that the step is too long
            cap = self.recoil_cap
            self.recoil_clipped = max(self.recoil_clipped, int(xp.sum(rp > cap)))
            self.lbm.gas_pressure = xp.minimum(rp, cap)
        return loss

    def _couple(self):
        """Darcy damping from the liquid fraction (lattice units); cells below the solidus are
        solid for the free surface (no mass exchange, no filling or emptying)."""
        xp = self.xp
        fl = xp.clip((self.T - self.Ts) / max(self.Tl - self.Ts, 1e-9), 0.0, 1.0)
        self.lbm.damping = self.mushy_C * self.dt * (1.0 - fl) ** 2 / (fl ** 3 + 1e-3)
        self.lbm.solid = self.T < self.Ts

    def _new_metal_cells(self, before):
        """Gas cells that became interface: give them the mean enthalpy of their metal
        neighbours."""
        xp = self.xp
        m = self.metal()
        new = m & ~before
        if not bool(new.any()):
            return
        s = xp.zeros(self.shape, dtype=self.dtype)
        n = xp.zeros(self.shape, dtype=self.dtype)
        mb = before.astype(self.dtype)
        for c in C[1:]:
            w = _nb(xp, mb, c)
            s += w * _nb(xp, self.H, c)
            n += w
        self.H = xp.where(new, xp.where(n > 0, s / xp.maximum(n, 1.0), self.H_ref), self.H)
        self.T = xp.where(new, self._T_of_H(self.H), self.T)

    # ------------------------------------------------------------------ run
    def run(self, laser, path, t_extra: float = 0.0, verbose: bool = False, frozen_flow: bool = False,
            snapshot_every: int = 0):
        """Scan ``path`` (list of :class:`tpmsfem.process.ScanVector`, positions in **m**,
        speed in m/s) with ``laser`` (:class:`tpmsfem.process.Laser`, radius and depth in
        **m**), then cool for ``t_extra`` s. ``frozen_flow``: skip the flow (conduction only,
        for comparison with the conduction model)."""
        xp, dx, dt = self.xp, self.dx, self.dt
        seg = []
        tc = 0.0
        for v in path:
            L = float(np.hypot(v.x1 - v.x0, v.y1 - v.y0))
            seg.append((tc, tc + L / v.speed, v))
            tc += L / v.speed
        t_end = tc + t_extra
        m0 = self.lbm.total_mass()
        self.lost = 0.0
        E0 = self.energy()
        absorbed = 0.0
        T_peak = self.T.copy()
        t_start = time.perf_counter()
        n = 0
        snaps = []
        while self.t < t_end - 1e-15:
            tm = self.t + 0.5 * dt
            q = None
            for (a, b, v) in seg:
                if a <= tm < b:
                    P = laser.power if v.power is None else v.power
                    if P > 0:
                        s = (tm - a) / (b - a)
                        q = self._laser_source(v.x0 + s * (v.x1 - v.x0), v.y0 + s * (v.y1 - v.y0), P,
                                               laser.absorptivity, laser.radius, laser.depth)
                    break
            loss = self._surface_terms()
            if not frozen_flow:
                self._couple()
                before = self.metal()
                self.lbm.step()
                self._new_metal_cells(before)
            e_in, e_out = self._heat_step(q if q is not None else 0.0, loss)
            absorbed += e_in
            self.lost += e_out
            T_peak = xp.maximum(T_peak, xp.where(self.metal(), self.T, -1e9))
            self.t += dt
            n += 1
            if snapshot_every and n % snapshot_every == 0:
                snaps.append((self.t, asnumpy(self.T[:, :, :]).astype(np.float32), asnumpy(self.lbm.fill).astype(np.float32)))
            if verbose and n % 500 == 0:
                um = float(xp.sqrt((self.lbm.u ** 2).sum(axis=0)).max()) * dx / dt
                print(f"  t = {self.t * 1e6:7.1f} us  T_max {float(self.T.max()):6.0f} C  u_max {um:5.2f} m/s  "
                      f"mass err {(self.lbm.total_mass() - m0) / m0:.1e}  ({time.perf_counter() - t_start:.0f} s)",
                      flush=True)
        secs = time.perf_counter() - t_start
        E1 = self.energy()
        um = float(xp.sqrt((self.lbm.u ** 2).sum(axis=0)).max()) * dx / dt
        res = LBMResult(dx, dt, asnumpy(self.T), asnumpy(T_peak), asnumpy(self.lbm.flag), asnumpy(self.lbm.fill),
                        asnumpy(self.surface_height()), um, (self.lbm.total_mass() - m0) / m0,
                        ((E1 - E0) - absorbed + self.lost) / max(absorbed, 1e-30), absorbed, n, secs,
                        n * np.prod(self.shape) / secs / 1e6)
        res.snapshots = snaps
        return res
