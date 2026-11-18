"""3-D quantitative phase field of directional solidification in a dilute multicomponent alloy.

Extends :mod:`tpmsfem.process.phasefield` (2-D, one solute) to three dimensions and several
solutes, so that the cells of a melt pool can be resolved in 3-D and each alloying element
mapped as in an EDS measurement (:mod:`tpmsfem.post.eds`).

Model. Each solute ``i`` (composition ``c_i`` in wt%, partition coefficient ``k_i``, liquidus
slope ``m_i``) has its own supersaturation

    U_i = [2 c_i / c_l0,i / (1 + k_i - (1 - k_i) phi) - 1] / (1 - k_i),     c_l0,i = c_inf,i / k_i,

and the interface temperature is the dilute multicomponent liquidus, linearised about the
planar steady state at T0 (the solidus of the alloy), so that the phase field sees the
weighted sum of the supersaturations:

    tau0 a(n)^2 [1 - (1 - k_bar) theta] d_t phi = div(W0^2 a^2 grad phi + anisotropic terms)
                                                  + phi - phi^3 - lambda (1 - phi^2)^2 (sum_i w_i U_i + theta),
    (1 + k_i - (1 - k_i) phi)/2 d_t U_i = div( D q(phi) grad U_i + a_t W0 [1 + (1 - k_i) U_i] d_t phi n )
                                          + [1 + (1 - k_i) U_i]/2 d_t phi,

with ``w_i = dT_i / dT0``, ``dT_i = -m_i (1 - k_i) c_l0,i`` (the share of solute i in the freezing
range ``dT0 = sum_i dT_i``), ``k_bar = sum_i w_i k_i``, ``theta = (z - z0 - V t) / l_T``,
``l_T = dT0 / G``, ``lambda = a1 W0 / d0``, ``d0 = Gamma / dT0`` and ``tau0 = a2 lambda W0^2 / D``.
For one solute this is exactly the 2-D model of Echebarria et al. (PRE 70 (2004) 061604); the
multicomponent form follows the dilute-multicomponent extension of the thin-interface analysis
(e.g. Ohno, PRE 86 (2012) 051603) with one liquid diffusivity ``D`` for all solutes, for which
the same ``tau0`` cancels the interface kinetics. ``[1 - (1 - k_bar) theta]`` generalises the
temperature dependence of the relaxation time with the weighted ``k_bar`` (an approximation;
exact for one solute). Solutes with ``m_i = 0`` partition (and are mapped) but do not change the
interface temperature: use this for elements with ``k`` close to 1, whose contribution to the
undercooling is negligible.

Anisotropy: cubic, ``a(n) = 1 - 3 eps4 + 4 eps4 sum_j n_j^4`` in the crystal frame
(``orientation``: rotation matrix, crystal = R @ lab); in 2-D it reduces to
``1 + eps4 cos 4 psi`` of the 2-D model.

Numerics: explicit finite differences, ``dx = 0.8 W0``, conservative face fluxes (solute is
conserved to round-off), periodic in x and y, no flux in z, a window that moves up by a quarter of
its height when the front passes 60 % of it; the composition of the rows leaving the window is
stored (wt%, float32) as the solidified 3-D microstructure. Kernels: fused numba kernel
(CPU, threaded over z), the same three passes as fused CUDA kernels (GPU, ``kernel="cuda"``,
the default on the CuPy backend), or plain NumPy/CuPy arrays (``kernel="array"``, the
reference implementation). With ``ny = 1`` and one solute the model is
the 2-D model of :mod:`tpmsfem.process.phasefield` (checked in the tests).

Validity: as for the 2-D model, ``W0`` must be well below ``D/V``; check convergence in ``W0``.
Solute trapping: :meth:`MultiAlloy.at_velocity` (continuous growth model, per solute).
The intercellular liquid that leaves the window is stored with its local composition; its
later solidification (further enrichment at the very centre) is not followed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from ..backend import asnumpy, get_xp
from .phasefield import A1, A2, A_AT, DiluteAlloy, numba_available


# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Solute:
    """One alloying element in a dilute multicomponent alloy."""
    name: str
    c_inf: float          # alloy content [wt%]
    k: float              # equilibrium partition coefficient c_s / c_l
    m: float = 0.0        # liquidus slope [K/wt%]; 0: partitions but does not change the interface temperature

    @property
    def c_l0(self) -> float:
        """Liquid composition at the planar steady-state interface (T0): c_inf / k [wt%]."""
        return self.c_inf / self.k

    @property
    def dT(self) -> float:
        """Contribution to the freezing range, -m (1 - k) c_l0 [K]."""
        return -self.m * (1.0 - self.k) * self.c_l0


@dataclass(frozen=True)
class MultiAlloy:
    """Dilute multicomponent alloy with linear liquidus surfaces and one liquid diffusivity."""
    solutes: tuple
    D: float = 3.0e-9                # liquid diffusivity [m^2/s]
    gibbs_thomson: float = 3.65e-7   # Gamma [K m]
    eps4: float = 0.02
    balance: str = "Ni"

    def __post_init__(self):
        object.__setattr__(self, "solutes", tuple(self.solutes))
        if not self.solutes:
            raise ValueError("at least one solute is needed")
        for s in self.solutes:
            if abs(1.0 - s.k) < 1e-6:
                raise ValueError(f"{s.name}: k = 1 does not partition; leave it out")
        if self.freezing_range <= 0.0:
            raise ValueError("the freezing range sum(-m (1 - k) c_l0) must be positive")

    @property
    def names(self):
        return [s.name for s in self.solutes]

    @property
    def freezing_range(self) -> float:
        """dT0 = sum_i -m_i (1 - k_i) c_l0,i [K]."""
        return float(sum(s.dT for s in self.solutes))

    @property
    def weights(self) -> np.ndarray:
        """w_i = dT_i / dT0 (sum 1): weight of each supersaturation in the interface temperature."""
        return np.array([s.dT for s in self.solutes]) / self.freezing_range

    @property
    def k_bar(self) -> float:
        return float(np.dot(self.weights, [s.k for s in self.solutes]))

    @property
    def d0(self) -> float:
        """Chemical capillary length Gamma / dT0 [m]."""
        return self.gibbs_thomson / self.freezing_range

    def at_velocity(self, V: float, V_D: float) -> "MultiAlloy":
        """Solute trapping (continuous growth model, no drag) for each solute, as
        :meth:`tpmsfem.process.phasefield.DiluteAlloy.at_velocity`:
        k(V) = (k_e + V/V_D)/(1 + V/V_D), m(V) = m_e [1 - k(V)(1 - ln(k(V)/k_e))]/(1 - k_e)."""
        r = V / V_D
        out = []
        for s in self.solutes:
            kv = (s.k + r) / (1.0 + r)
            mv = s.m * (1.0 - kv * (1.0 - np.log(kv / s.k))) / (1.0 - s.k)
            out.append(Solute(s.name, s.c_inf, float(kv), float(mv)))
        return replace(self, solutes=tuple(out))

    @classmethod
    def from_dilute(cls, alloy: DiluteAlloy, name: str = "Nb", balance: str = "Ni") -> "MultiAlloy":
        """The one-solute alloy equivalent to a :class:`DiluteAlloy`."""
        return cls((Solute(name, alloy.c_inf, alloy.k, alloy.m),), D=alloy.D, gibbs_thomson=alloy.gibbs_thomson,
                   eps4=alloy.eps4, balance=balance)


#: IN625 of the AM-Bench 2018 feedstock (ICP, Keller et al., Acta Mater. 139 (2017) 244, Table 1:
#: 3.75 Nb, 8.83 Mo, 20.7 Cr, 0.72 Fe wt%) with the equilibrium partition coefficients used by
#: Gan et al. (Integr. Mater. Manuf. Innov. 8 (2019) 178, Table 4, after Hu et al. 2017):
#: Nb 0.48, Mo 0.85, Cr 1.02, Fe 1.09. Liquidus slopes: Ni-Nb -10.5 K/wt% (Ghosh et al. 2017, as
#: in the 2-D model); Ni-Mo -2.9 K/wt% (secant of the Ni-Mo liquidus from 1455 C to the eutectic,
#: about 1315 C at 49 wt% Mo: an estimate); Cr and Fe partition (k near 1) but are taken to leave
#: the interface temperature unchanged (their share of the freezing range is below 2 %).
#: Freezing range 42.7 (Nb) + 4.5 (Mo) = 47.2 K. One liquid diffusivity, 3e-9 m^2/s.
IN625_AMBENCH = MultiAlloy((Solute("Nb", 3.75, 0.48, -10.5), Solute("Mo", 8.83, 0.85, -2.9),
                            Solute("Cr", 20.7, 1.02), Solute("Fe", 0.72, 1.09)), eps4=0.02)


def rotation_matrix(euler_deg: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Crystal-from-lab rotation for Bunge Euler angles (phi1, Phi, phi2) [deg]: a lab vector v
    has crystal components R @ v."""
    p1, P, p2 = np.radians(euler_deg)
    c1, s1, c, s, c2, s2 = np.cos(p1), np.sin(p1), np.cos(P), np.sin(P), np.cos(p2), np.sin(p2)
    return np.array([[c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
                     [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
                     [s1 * s, -c1 * s, c]])


# ----------------------------------------------------------------------------------------
@dataclass
class PF3DResult:
    phi: np.ndarray                  # phase field in the final window (nz, ny, nx)
    comp: dict                       # element -> composition in the window [wt%]
    solid: dict                      # element -> composition of the rows that left the window [wt%], oldest first
    dx_m: float
    t_s: float
    steps: int
    seconds: float
    front_history: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    balance: str = "Ni"

    @property
    def names(self):
        return list(self.comp)

    def grown(self, name: str) -> np.ndarray:
        """Composition [wt%] of everything solidified so far plus the window, bottom first
        (nz_total, ny, nx). ``name`` may be the balance element (100 - sum of the solutes)."""
        if name == self.balance:
            return 100.0 - sum(self.grown(n) for n in self.names)
        w = self.comp[name].astype(np.float32)
        s = self.solid.get(name)
        return w if s is None or s.shape[0] == 0 else np.concatenate([s, w], axis=0)

    def grown_phi(self) -> np.ndarray:
        n = 0 if not self.solid else self.solid[self.names[0]].shape[0]
        return np.concatenate([np.ones((n,) + self.phi.shape[1:], dtype=np.float32), self.phi.astype(np.float32)], axis=0)

    def tip_row(self) -> int:
        """Highest row (in :meth:`grown` indexing) with solid (phi > 0)."""
        rows = np.flatnonzero((self.phi > 0.0).reshape(self.phi.shape[0], -1).any(axis=1))
        n = 0 if not self.solid else self.solid[self.names[0]].shape[0]
        return n + (int(rows.max()) if len(rows) else 0)

    def cells(self, row: int, name: Optional[str] = None, depth_frac: float = 0.1, min_area_frac: float = 0.15):
        """Cells in the transverse section ``row`` (grown indexing), counted by watershed on the
        section of the most strongly segregating solute (``name``; default: smallest k): basins
        grow from the regional minima (maxima for k > 1) deeper than ``depth_frac`` of the
        section's 5-95 % range and meet at the enriched boundaries, whatever their height (a
        global threshold merges cells whose walls are weak next to strongly enriched
        triple junctions). The section is tiled 3 x 3 for periodicity. Returns (labels, number
        of cells, mean spacing sqrt(A / N) [m]); basins smaller than ``min_area_frac`` of the
        mean are not counted. Needs scikit-image."""
        from scipy import ndimage
        from skimage.morphology import h_minima
        from skimage.segmentation import watershed

        if name is None:
            name = min(self.params["solutes"], key=lambda s: s["k"])["name"]
        k = {s["name"]: s["k"] for s in self.params["solutes"]}[name]
        sec = ndimage.gaussian_filter(self.grown(name)[row].astype(float), 1.0, mode="wrap")
        f = sec if k < 1.0 else -sec
        n0, n1 = f.shape
        tile = np.tile(f, (3, 3))
        h = depth_frac * (np.percentile(f, 95) - np.percentile(f, 5))
        markers, nm = ndimage.label(h_minima(tile, max(h, 1e-12)))
        lab = watershed(tile, markers)
        centres = ndimage.center_of_mass(np.ones_like(tile), markers, np.arange(1, nm + 1))
        central = [i + 1 for i, (cy, cx) in enumerate(centres) if n0 <= cy < 2 * n0 and n1 <= cx < 2 * n1]
        area = ndimage.sum(np.ones_like(tile), lab, central)
        if len(central) == 0:
            return lab[n0:2 * n0, n1:2 * n1], 0, np.inf
        keep = np.asarray(area) >= min_area_frac * np.mean(area)
        n_cells = int(keep.sum())
        A = f.size * self.dx_m ** 2
        return lab[n0:2 * n0, n1:2 * n1], n_cells, float(np.sqrt(A / max(n_cells, 1)))

    def segregation(self, name: str, rows: Optional[slice] = None):
        """Statistics [wt%] of one element over solidified rows (default: the stored rows
        beyond the first quarter, i.e. after the initial transient)."""
        v = self.grown(name)
        if rows is None:
            n = 0 if not self.solid else self.solid[self.names[0]].shape[0]
            rows = slice(n // 4, n) if n >= 8 else slice(None)
        v = np.asarray(v[rows], dtype=float).ravel()
        return dict(min=float(v.min()), p05=float(np.percentile(v, 5)), p50=float(np.median(v)),
                    p95=float(np.percentile(v, 95)), max=float(v.max()), mean=float(v.mean()))


# ----------------------------------------------------------------------------------------
class DirectionalPF3D:
    """3-D directional solidification of a :class:`MultiAlloy` at thermal gradient ``G`` [K/m]
    and velocity ``V`` [m/s] in a periodic ``size_m = (Lx, Ly)`` box of window height
    ``height_m``. ``size_m[1]`` equal to one cell (``ny = 1``) gives the 2-D model."""

    def __init__(self, alloy: MultiAlloy, G: float, V: float, size_m: Sequence[float], height_m: float,
                 W0_m: Optional[float] = None, W_over_d0: float = 2.5, dx_over_W: float = 0.8,
                 orientation: Optional[np.ndarray] = None, noise: float = 0.02, seed: int = 0, dtype=np.float64,
                 kernel: str = "auto", ny: Optional[int] = None):
        self.xp = xp = get_xp()
        if kernel == "auto":
            kernel = ("numba" if numba_available() else "array") if xp is np else "cuda"
        if kernel == "numba" and xp is not np:
            raise ValueError("the numba kernel runs on the NumPy backend only")
        if kernel == "cuda" and xp is np:
            raise ValueError("the cuda kernel runs on the CuPy backend only")
        if kernel not in ("numba", "cuda", "array"):
            raise ValueError(f"unknown kernel {kernel!r}; use 'auto', 'numba', 'cuda' or 'array'")
        self.kernel = kernel
        self.alloy, self.G, self.V = alloy, float(G), float(V)
        d0 = alloy.d0
        W_over_d0 = W0_m / d0 if W0_m is not None else W_over_d0
        self.W0 = W_over_d0 * d0
        self.lam = A1 * W_over_d0
        self.tau0 = A2 * self.lam * self.W0 ** 2 / alloy.D
        self.Dt = A2 * self.lam
        self.dx = dx_over_W
        self.lT = alloy.freezing_range / self.G / self.W0
        self.Vt = self.V * self.tau0 / self.W0
        self.ks = np.array([s.k for s in alloy.solutes], dtype=np.float64)
        self.ws = alloy.weights.astype(np.float64)
        self.kbar = alloy.k_bar
        self.nx = max(int(round(size_m[0] / (self.dx * self.W0))), 1)
        self.ny = ny if ny is not None else max(int(round(size_m[1] / (self.dx * self.W0))), 1)
        self.nz = int(round(height_m / (self.dx * self.W0)))
        self.ndim = 1 + (self.nx > 1) + (self.ny > 1)
        self.dt = self._stable_dt(self.kbar)
        self.Rm = np.eye(3) if orientation is None else np.asarray(orientation, dtype=np.float64)
        rng = np.random.default_rng(seed)
        z = (np.arange(self.nz) + 0.5) * self.dx
        z_front = 0.1 * self.nz * self.dx
        prof = -np.tanh((z - z_front) / np.sqrt(2.0))
        phi = np.broadcast_to(prof[:, None, None], (self.nz, self.ny, self.nx)).copy()
        phi += noise * rng.standard_normal(phi.shape) * (1.0 - phi ** 2)
        self.phi = xp.asarray(np.clip(phi, -1.0, 1.0), dtype=dtype)
        lD = alloy.D / self.V / self.W0
        U_liq = np.expm1(-np.maximum(z - z_front, 0.0) / lD)
        h = 0.5 * (1.0 + prof)
        U1 = np.broadcast_to(((1.0 - h) * U_liq)[:, None, None], (self.nz, self.ny, self.nx))
        self.U = xp.asarray(np.broadcast_to(U1, (len(self.ks),) + U1.shape).copy(), dtype=dtype)
        self.z0 = z_front
        self.offset = 0.0
        self.t = 0.0
        self.steps_done = 0
        self._zc = np.ascontiguousarray(z, dtype=np.float64)
        self.solid_rows = []
        self.front = []
        self._work = None

    # ------------------------------------------------------------------
    def _stable_dt(self, f: float) -> float:
        return 0.5 * min(self.dx ** 2 * f / (2.0 * self.ndim), self.dx ** 2 / (2.0 * self.ndim * self.Dt))

    def composition(self, i: Optional[int] = None):
        """Composition [wt%] of solute ``i`` (all solutes, stacked on axis 0, if None)."""
        xp = self.xp
        if i is None:
            return xp.stack([self.composition(j) for j in range(len(self.ks))])
        k = self.ks[i]
        s = self.alloy.solutes[i]
        return s.c_l0 * (1.0 + k - (1.0 - k) * self.phi) / 2.0 * (1.0 + (1.0 - k) * self.U[i])

    def solute(self):
        """Total of each solute in the window, sum of c / c_inf over the cells."""
        return np.array([float(asnumpy(self.composition(i)).sum()) / s.c_inf for i, s in enumerate(self.alloy.solutes)])

    def step(self):
        if self.kernel == "numba":
            if self._work is None or self._work[0].shape != self.phi.shape:
                self._work = [np.empty_like(self.phi) for _ in range(8)] + [np.empty_like(self.U)]
            R, phin, FX, FY, FZ, NX, NY, NZ, Un = self._work
            _numba_kernel()(self.phi, self.U, R, phin, Un, FX, FY, FZ, NX, NY, NZ, self._zc, self.offset, self.z0,
                            self.Vt, self.t, self.lT, self.kbar, self.lam, self.Dt, A_AT, self.dx, self.dt,
                            self.alloy.eps4, self.Rm, self.ks, self.ws)
            self._work = [R, self.phi, FX, FY, FZ, NX, NY, NZ, self.U]
            self.phi, self.U = phin, Un
        elif self.kernel == "cuda":
            self._step_cuda()
        else:
            self._step_array()
        self.t += self.dt
        self.steps_done += 1

    # ------------------------------------------------------------------
    def _flux(self, px, py, pz):
        """Anisotropic flux a^2 p + a |p|^2 da/dp (lab frame) and |p|^2."""
        xp, eps, R = self.xp, self.alloy.eps4, self.Rm
        if eps == 0.0:
            return px, py, pz, px * px + py * py + pz * pz
        qx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz
        qy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz
        qz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
        s2 = qx * qx + qy * qy + qz * qz
        S4 = qx ** 4 + qy ** 4 + qz ** 4
        inv = 1.0 / (s2 * s2 + 1e-24)
        a = 1.0 + eps * (4.0 * S4 - 3.0 * s2 * s2) * inv
        b = 16.0 * eps * a * inv
        fx = a * a * qx + b * (qx ** 3 * s2 - S4 * qx)
        fy = a * a * qy + b * (qy ** 3 * s2 - S4 * qy)
        fz = a * a * qz + b * (qz ** 3 * s2 - S4 * qz)
        return (R[0, 0] * fx + R[1, 0] * fy + R[2, 0] * fz, R[0, 1] * fx + R[1, 1] * fy + R[2, 1] * fz,
                R[0, 2] * fx + R[1, 2] * fy + R[2, 2] * fz, px * px + py * py + pz * pz)

    def _a(self, px, py, pz):
        eps, R = self.alloy.eps4, self.Rm
        qx = R[0, 0] * px + R[0, 1] * py + R[0, 2] * pz
        qy = R[1, 0] * px + R[1, 1] * py + R[1, 2] * pz
        qz = R[2, 0] * px + R[2, 1] * py + R[2, 2] * pz
        s2 = qx * qx + qy * qy + qz * qz
        return 1.0 + eps * (4.0 * (qx ** 4 + qy ** 4 + qz ** 4) - 3.0 * s2 * s2) / (s2 * s2 + 1e-24)

    def _step_array(self):
        """NumPy/CuPy version of the numba kernel (same discretisation)."""
        xp, dx, dt = self.xp, self.dx, self.dt
        P = self.phi

        def E(f):                                   # value at x + 1 (periodic)
            return xp.roll(f, -1, axis=2)

        def Wst(f):
            return xp.roll(f, 1, axis=2)

        def N(f):
            return xp.roll(f, -1, axis=1)

        def S(f):
            return xp.roll(f, 1, axis=1)

        def T(f):                                   # z + 1, mirror at the top
            return xp.concatenate([f[1:], f[-1:]], axis=0)

        def B(f):
            return xp.concatenate([f[:1], f[:-1]], axis=0)

        # x-faces (east of each cell)
        px = (E(P) - P) / dx
        py = (N(P) + E(N(P)) - S(P) - E(S(P))) / (4.0 * dx)
        pz = (T(P) + E(T(P)) - B(P) - E(B(P))) / (4.0 * dx)
        FX, _, _, g = self._flux(px, py, pz)
        NX = px / xp.sqrt(g + 1e-24)
        # y-faces (north)
        py = (N(P) - P) / dx
        px = (E(P) + N(E(P)) - Wst(P) - N(Wst(P))) / (4.0 * dx)
        pz = (T(P) + N(T(P)) - B(P) - N(B(P))) / (4.0 * dx)
        _, FY, _, g = self._flux(px, py, pz)
        NY = py / xp.sqrt(g + 1e-24)
        # z-faces (top); none through the window top
        pz = (T(P) - P) / dx
        px = (E(P) + T(E(P)) - Wst(P) - T(Wst(P))) / (4.0 * dx)
        py = (N(P) + T(N(P)) - S(P) - T(S(P))) / (4.0 * dx)
        _, _, FZ, g = self._flux(px, py, pz)
        NZ = pz / xp.sqrt(g + 1e-24)
        FZ[-1] = 0.0
        NZ[-1] = 0.0

        def Bz(f):                                  # face below (zero at the bottom wall)
            return xp.concatenate([xp.zeros_like(f[:1]), f[:-1]], axis=0)

        div = (FX - Wst(FX) + FY - S(FY) + FZ - Bz(FZ)) / dx
        a = self._a((E(P) - Wst(P)) / (2.0 * dx), (N(P) - S(P)) / (2.0 * dx), (T(P) - B(P)) / (2.0 * dx))
        zc = xp.asarray(self._zc, dtype=P.dtype)[:, None, None]
        theta = (zc + self.offset - self.z0 - self.Vt * self.t) / self.lT
        tf = xp.maximum(1.0 - (1.0 - self.kbar) * theta, self.kbar)
        Usum = sum(float(w) * self.U[s] for s, w in enumerate(self.ws))
        g2 = (1.0 - P * P) ** 2
        R = (div + P - P ** 3 - self.lam * g2 * (Usum + theta)) / (a * a * tf)
        phin = P + dt * R
        Un = xp.empty_like(self.U)
        for s, k in enumerate(self.ks):
            U = self.U[s]
            F = 1.0 + (1.0 - k) * U
            JX = self.Dt * 0.25 * (2.0 - P - E(P)) * (E(U) - U) / dx + A_AT * 0.25 * (F + E(F)) * (R + E(R)) * NX
            JY = self.Dt * 0.25 * (2.0 - P - N(P)) * (N(U) - U) / dx + A_AT * 0.25 * (F + N(F)) * (R + N(R)) * NY
            JZ = self.Dt * 0.25 * (2.0 - P - T(P)) * (T(U) - U) / dx + A_AT * 0.25 * (F + T(F)) * (R + T(R)) * NZ
            JZ[-1] = 0.0
            divJ = (JX - Wst(JX) + JY - S(JY) + JZ - Bz(JZ)) / dx
            Un[s] = U + dt * (divJ + 0.5 * F * R) / ((1.0 + k - (1.0 - k) * phin) / 2.0)
        self.phi, self.U = phin, Un

    def _step_cuda(self):
        """The numba kernel's three passes as fused CUDA kernels (one thread per cell)."""
        import cupy as cp

        phi, U = self.phi, self.U
        dt_ = phi.dtype
        real = dt_.type
        if self._work is None or self._work[0].shape != phi.shape or self._work[0].dtype != dt_:
            self._work = [cp.empty_like(phi) for _ in range(8)] + [cp.empty_like(U)]
            self._cuda_consts = (cp.asarray(self.Rm, dtype=dt_), cp.asarray(self.ks, dtype=dt_),
                                 cp.asarray(self.ws, dtype=dt_), cp.asarray(self._zc, dtype=dt_))
        R, phin, FX, FY, FZ, NX, NY, NZ, Un = self._work
        Rm, ks, ws, zc = self._cuda_consts
        phi = cp.ascontiguousarray(phi)
        U = cp.ascontiguousarray(U)
        k1, k2, k3 = _cuda_kernels(dt_)
        nz, ny, nx = phi.shape
        n = nz * ny * nx
        block = 256
        grid = ((n + block - 1) // block,)
        ni = (np.int32(nz), np.int32(ny), np.int32(nx))
        k1(grid, (block,), (phi, FX, FY, FZ, NX, NY, NZ, Rm, real(self.alloy.eps4), real(self.dx)) + ni)
        k2(grid, (block,), (phi, U, FX, FY, FZ, R, Rm, ws, zc, real(self.alloy.eps4), real(self.dx),
                            real(self.offset), real(self.z0), real(self.Vt), real(self.t), real(self.lT),
                            real(self.kbar), real(self.lam), np.int32(len(self.ks))) + ni)
        k3(grid, (block,), (phi, U, R, NX, NY, NZ, phin, Un, ks, real(self.Dt), real(A_AT), real(self.dx),
                            real(self.dt), np.int32(len(self.ks))) + ni)
        self._work = [R, phi, FX, FY, FZ, NX, NY, NZ, U]
        self.phi, self.U = phin, Un

    # ------------------------------------------------------------------
    def _shift(self, rows: int):
        xp = self.xp
        self.solid_rows.append(asnumpy(self.composition()[:, :rows]).astype(np.float32))
        self.phi = xp.concatenate([self.phi[rows:], xp.full((rows,) + self.phi.shape[1:], -1.0, dtype=self.phi.dtype)], axis=0)
        self.U = xp.concatenate([self.U[:, rows:], xp.full((self.U.shape[0], rows) + self.U.shape[2:], -1.0,
                                                           dtype=self.U.dtype)], axis=1)
        self.offset += rows * self.dx

    def tip_row(self) -> int:
        rows = np.flatnonzero(asnumpy((self.phi > 0.0).reshape(self.nz, -1).any(axis=1)))
        return int(rows.max()) if len(rows) else 0

    def _theta(self, row):
        return ((row + 0.5) * self.dx + self.offset - self.z0 - self.Vt * self.t) / self.lT

    def adapt_dt(self, tip_row: Optional[int] = None, margin: float = 0.1) -> bool:
        """Largest stable step for the current tip position (as in the 2-D model, with k_bar)."""
        k = self.kbar
        tip_row = self.tip_row() if tip_row is None else tip_row
        theta_tip = max(self._theta(tip_row), 0.0)
        theta = theta_tip + 0.65 * self.nz * self.dx / self.lT + margin
        dt = self._stable_dt(min(max(1.0 - (1.0 - k) * theta, k), 1.0))
        self._theta_ok = theta_tip + margin
        if abs(dt - self.dt) <= 1e-3 * self.dt:
            return False
        self.dt = dt
        return True

    def run(self, t_end_s: float, shift_at: Optional[float] = 0.6, record_every: int = 200, verbose: bool = False,
            adapt_dt: bool = False) -> PF3DResult:
        """Advance by ``t_end_s`` (physical time); see :meth:`tpmsfem.process.phasefield.DirectionalPF.run`."""
        t0 = time.perf_counter()
        t_target = self.t + t_end_s / self.tau0
        if adapt_dt:
            self.adapt_dt()
        n = int(np.ceil(t_end_s / (self.dt * self.tau0)))
        s = 0
        while s < n:
            self.step()
            if s % record_every == 0 or s == n - 1:
                tip = self.tip_row()
                self.front.append((self.t * self.tau0, ((tip + 0.5) * self.dx + self.offset - self.z0) * self.W0))
                if shift_at is not None and tip > shift_at * self.nz:
                    self._shift(int(0.25 * self.nz))
                    tip -= int(0.25 * self.nz)
                if adapt_dt and (self._theta(tip) > self._theta_ok or s % (record_every * 20) == 0):
                    if self.adapt_dt(tip):
                        n = s + 1 + max(int(np.ceil((t_target - self.t) / self.dt)), 0)
                if verbose and s % (record_every * 20) == 0:
                    print(f"  step {s}/{n}: t = {self.t * self.tau0 * 1e6:.3f} us, tip at {self.front[-1][1] * 1e6:.3f} um",
                          flush=True)
            s += 1
        return self.result(time.perf_counter() - t0, n)

    def result(self, seconds: float = 0.0, steps: Optional[int] = None) -> PF3DResult:
        names = self.alloy.names
        comp = asnumpy(self.composition())
        solid = {}
        if self.solid_rows:
            st = np.concatenate(self.solid_rows, axis=1)
            solid = {n: st[i] for i, n in enumerate(names)}
        return PF3DResult(
            asnumpy(self.phi), {n: comp[i] for i, n in enumerate(names)}, solid, self.dx * self.W0, self.t * self.tau0,
            self.steps_done if steps is None else steps, seconds, list(self.front),
            dict(G=self.G, V=self.V, W0_m=self.W0, tau0_s=self.tau0, lambda_=self.lam, dt=self.dt, nx=self.nx,
                 ny=self.ny, nz=self.nz, lT_m=self.lT * self.W0, d0_m=self.alloy.d0, freezing_range_K=self.alloy.freezing_range,
                 diffusion_length_m=self.alloy.D / self.V, k_bar=self.kbar, orientation=self.Rm.tolist(),
                 solutes=[dict(name=s.name, c_inf=s.c_inf, k=s.k, m=s.m, w=float(w))
                          for s, w in zip(self.alloy.solutes, self.ws)]),
            balance=self.alloy.balance)

    # ------------------------------------------------------------------
    def checkpoint(self, path: str):
        """Save the state (fields, time, window offset, stored rows, front history) to ``path`` (.npz)."""
        st = np.concatenate(self.solid_rows, axis=1) if self.solid_rows else np.zeros((len(self.ks), 0, self.ny, self.nx),
                                                                                      np.float32)
        tmp = path + ".tmp.npz"
        np.savez(tmp, phi=asnumpy(self.phi), U=asnumpy(self.U), solid=st, front=np.array(self.front, float).reshape(-1, 2),
                 scalars=np.array([self.t, self.offset, self.dt, self.z0, self.steps_done], float),
                 shape=np.array([self.nz, self.ny, self.nx, len(self.ks)]))
        import os
        os.replace(tmp, path)

    def restore(self, path: str):
        """Load a state written by :meth:`checkpoint` into this (identically constructed) model."""
        d = np.load(path)
        if tuple(d["shape"]) != (self.nz, self.ny, self.nx, len(self.ks)):
            raise ValueError("checkpoint does not match the model size")
        xp = self.xp
        self.phi = xp.asarray(d["phi"], dtype=self.phi.dtype)
        self.U = xp.asarray(d["U"], dtype=self.U.dtype)
        self.t, self.offset, self.dt, self.z0, steps = (float(v) for v in d["scalars"])
        self.steps_done = int(steps)
        self.solid_rows = [d["solid"]] if d["solid"].shape[1] else []
        self.front = [tuple(r) for r in d["front"]]
        self._work = None


# ----------------------------------------------------------------------------------------
_KERNELS = {}


def _numba_kernel():
    if "step" in _KERNELS:
        return _KERNELS["step"]
    import numba

    @numba.njit(inline="always", fastmath=False)
    def flux(px, py, pz, eps, Rm):
        qx = Rm[0, 0] * px + Rm[0, 1] * py + Rm[0, 2] * pz
        qy = Rm[1, 0] * px + Rm[1, 1] * py + Rm[1, 2] * pz
        qz = Rm[2, 0] * px + Rm[2, 1] * py + Rm[2, 2] * pz
        s2 = qx * qx + qy * qy + qz * qz
        S4 = qx * qx * qx * qx + qy * qy * qy * qy + qz * qz * qz * qz
        inv = 1.0 / (s2 * s2 + 1e-24)
        a = 1.0 + eps * (4.0 * S4 - 3.0 * s2 * s2) * inv
        b = 16.0 * eps * a * inv
        fx = a * a * qx + b * (qx * qx * qx * s2 - S4 * qx)
        fy = a * a * qy + b * (qy * qy * qy * s2 - S4 * qy)
        fz = a * a * qz + b * (qz * qz * qz * s2 - S4 * qz)
        return (Rm[0, 0] * fx + Rm[1, 0] * fy + Rm[2, 0] * fz, Rm[0, 1] * fx + Rm[1, 1] * fy + Rm[2, 1] * fz,
                Rm[0, 2] * fx + Rm[1, 2] * fy + Rm[2, 2] * fz, px * px + py * py + pz * pz, a)

    @numba.njit(parallel=True, cache=True)
    def step(phi, U, R, phin, Un, FX, FY, FZ, NX, NY, NZ, zc, offset, z0, Vt, t, lT, kbar, lam, Dt, at, dx, dt, eps,
             Rm, ks, ws):
        nz, ny, nx = phi.shape
        ns = ks.shape[0]
        q = 1.0 / (4.0 * dx)
        # pass 1: phi fluxes and unit-normal components on the east, north and top faces
        for i in numba.prange(nz):
            ip = min(i + 1, nz - 1)
            im = max(i - 1, 0)
            for j in range(ny):
                jp = j + 1 if j + 1 < ny else 0
                jm = j - 1 if j > 0 else ny - 1
                for l in range(nx):
                    lp = l + 1 if l + 1 < nx else 0
                    lm = l - 1 if l > 0 else nx - 1
                    px = (phi[i, j, lp] - phi[i, j, l]) / dx
                    py = (phi[i, jp, l] + phi[i, jp, lp] - phi[i, jm, l] - phi[i, jm, lp]) * q
                    pz = (phi[ip, j, l] + phi[ip, j, lp] - phi[im, j, l] - phi[im, j, lp]) * q
                    fx, fy, fz, g, a = flux(px, py, pz, eps, Rm)
                    FX[i, j, l] = fx
                    NX[i, j, l] = px / np.sqrt(g + 1e-24)
                    py = (phi[i, jp, l] - phi[i, j, l]) / dx
                    px = (phi[i, j, lp] + phi[i, jp, lp] - phi[i, j, lm] - phi[i, jp, lm]) * q
                    pz = (phi[ip, j, l] + phi[ip, jp, l] - phi[im, j, l] - phi[im, jp, l]) * q
                    fx, fy, fz, g, a = flux(px, py, pz, eps, Rm)
                    FY[i, j, l] = fy
                    NY[i, j, l] = py / np.sqrt(g + 1e-24)
                    if i < nz - 1:
                        pz = (phi[i + 1, j, l] - phi[i, j, l]) / dx
                        px = (phi[i, j, lp] + phi[i + 1, j, lp] - phi[i, j, lm] - phi[i + 1, j, lm]) * q
                        py = (phi[i, jp, l] + phi[i + 1, jp, l] - phi[i, jm, l] - phi[i + 1, jm, l]) * q
                        fx, fy, fz, g, a = flux(px, py, pz, eps, Rm)
                        FZ[i, j, l] = fz
                        NZ[i, j, l] = pz / np.sqrt(g + 1e-24)
                    else:
                        FZ[i, j, l] = 0.0
                        NZ[i, j, l] = 0.0
        # pass 2: d_t phi
        for i in numba.prange(nz):
            ip = min(i + 1, nz - 1)
            im = max(i - 1, 0)
            theta = (zc[i] + offset - z0 - Vt * t) / lT
            tf = max(1.0 - (1.0 - kbar) * theta, kbar)
            for j in range(ny):
                jp = j + 1 if j + 1 < ny else 0
                jm = j - 1 if j > 0 else ny - 1
                for l in range(nx):
                    lp = l + 1 if l + 1 < nx else 0
                    lm = l - 1 if l > 0 else nx - 1
                    fb = FZ[i - 1, j, l] if i > 0 else 0.0
                    div = (FX[i, j, l] - FX[i, j, lm] + FY[i, j, l] - FY[i, jm, l] + FZ[i, j, l] - fb) / dx
                    fx, fy, fz, g, a = flux((phi[i, j, lp] - phi[i, j, lm]) / (2.0 * dx),
                                            (phi[i, jp, l] - phi[i, jm, l]) / (2.0 * dx),
                                            (phi[ip, j, l] - phi[im, j, l]) / (2.0 * dx), eps, Rm)
                    us = 0.0
                    for s in range(ns):
                        us += ws[s] * U[s, i, j, l]
                    p = phi[i, j, l]
                    g2 = (1.0 - p * p) * (1.0 - p * p)
                    R[i, j, l] = (div + p - p * p * p - lam * g2 * (us + theta)) / (a * a * tf)
        # pass 3: phi and solute updates
        for i in numba.prange(nz):
            ip = min(i + 1, nz - 1)
            for j in range(ny):
                jp = j + 1 if j + 1 < ny else 0
                jm = j - 1 if j > 0 else ny - 1
                for l in range(nx):
                    lp = l + 1 if l + 1 < nx else 0
                    lm = l - 1 if l > 0 else nx - 1
                    pc = phi[i, j, l]
                    rc = R[i, j, l]
                    pn = pc + dt * rc
                    phin[i, j, l] = pn
                    for s in range(ns):
                        k = ks[s]
                        uc = U[s, i, j, l]
                        Fc = 1.0 + (1.0 - k) * uc
                        je = (Dt * 0.25 * (2.0 - pc - phi[i, j, lp]) * (U[s, i, j, lp] - uc) / dx
                              + at * 0.25 * (Fc + 1.0 + (1.0 - k) * U[s, i, j, lp]) * (rc + R[i, j, lp]) * NX[i, j, l])
                        jw = (Dt * 0.25 * (2.0 - phi[i, j, lm] - pc) * (uc - U[s, i, j, lm]) / dx
                              + at * 0.25 * (1.0 + (1.0 - k) * U[s, i, j, lm] + Fc) * (R[i, j, lm] + rc) * NX[i, j, lm])
                        jn = (Dt * 0.25 * (2.0 - pc - phi[i, jp, l]) * (U[s, i, jp, l] - uc) / dx
                              + at * 0.25 * (Fc + 1.0 + (1.0 - k) * U[s, i, jp, l]) * (rc + R[i, jp, l]) * NY[i, j, l])
                        js = (Dt * 0.25 * (2.0 - phi[i, jm, l] - pc) * (uc - U[s, i, jm, l]) / dx
                              + at * 0.25 * (1.0 + (1.0 - k) * U[s, i, jm, l] + Fc) * (R[i, jm, l] + rc) * NY[i, jm, l])
                        jt = 0.0
                        if i < nz - 1:
                            jt = (Dt * 0.25 * (2.0 - pc - phi[ip, j, l]) * (U[s, ip, j, l] - uc) / dx
                                  + at * 0.25 * (Fc + 1.0 + (1.0 - k) * U[s, ip, j, l]) * (rc + R[ip, j, l]) * NZ[i, j, l])
                        jb = 0.0
                        if i > 0:
                            jb = (Dt * 0.25 * (2.0 - phi[i - 1, j, l] - pc) * (uc - U[s, i - 1, j, l]) / dx
                                  + at * 0.25 * (1.0 + (1.0 - k) * U[s, i - 1, j, l] + Fc) * (R[i - 1, j, l] + rc)
                                  * NZ[i - 1, j, l])
                        Un[s, i, j, l] = uc + dt * ((je - jw + jn - js + jt - jb) / dx + 0.5 * Fc * rc) / (
                            (1.0 + k - (1.0 - k) * pn) / 2.0)

    _KERNELS["step"] = step
    return step


# The numba kernel's three passes, one CUDA thread per cell, same arithmetic in the same
# order (so fp64 results match the array and numba kernels to round-off). ``real`` is
# double or float; everything, scalars included, is computed in it.
_CUDA_SRC = r"""
typedef REAL real;

#define IDX(i, j, l) ((((long long)(i)) * ny + (j)) * nx + (l))
#define DECODE                                                            \
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;      \
    long long n = (long long)nz * ny * nx;                                 \
    if (c >= n) return;                                                     \
    const int l = (int)(c % nx);                                            \
    const long long r_ = c / nx;                                            \
    const int j = (int)(r_ % ny);                                           \
    const int i = (int)(r_ / ny);                                           \
    const int ip = i + 1 < nz ? i + 1 : nz - 1;                             \
    const int im = i > 0 ? i - 1 : 0;                                       \
    const int jp = j + 1 < ny ? j + 1 : 0;                                  \
    const int jm = j > 0 ? j - 1 : ny - 1;                                  \
    const int lp = l + 1 < nx ? l + 1 : 0;                                  \
    const int lm = l > 0 ? l - 1 : nx - 1;

__device__ __forceinline__ void flux(real px, real py, real pz, real eps, const real* Rm,
                                     real& ox, real& oy, real& oz, real& g, real& a)
{
    real qx = Rm[0] * px + Rm[1] * py + Rm[2] * pz;
    real qy = Rm[3] * px + Rm[4] * py + Rm[5] * pz;
    real qz = Rm[6] * px + Rm[7] * py + Rm[8] * pz;
    real s2 = qx * qx + qy * qy + qz * qz;
    real S4 = qx * qx * qx * qx + qy * qy * qy * qy + qz * qz * qz * qz;
    real inv = (real)1.0 / (s2 * s2 + (real)1e-24);
    a = (real)1.0 + eps * ((real)4.0 * S4 - (real)3.0 * s2 * s2) * inv;
    real b = (real)16.0 * eps * a * inv;
    real fx = a * a * qx + b * (qx * qx * qx * s2 - S4 * qx);
    real fy = a * a * qy + b * (qy * qy * qy * s2 - S4 * qy);
    real fz = a * a * qz + b * (qz * qz * qz * s2 - S4 * qz);
    ox = Rm[0] * fx + Rm[3] * fy + Rm[6] * fz;
    oy = Rm[1] * fx + Rm[4] * fy + Rm[7] * fz;
    oz = Rm[2] * fx + Rm[5] * fy + Rm[8] * fz;
    g = px * px + py * py + pz * pz;
}

// pass 1: phi fluxes and unit-normal components on the east, north and top faces
extern "C" __global__ void pf3_faces(const real* __restrict__ phi, real* FX, real* FY, real* FZ,
                                     real* NX, real* NY, real* NZ, const real* __restrict__ Rm,
                                     const real eps, const real dx, const int nz, const int ny, const int nx)
{
    DECODE
    const real q = (real)1.0 / ((real)4.0 * dx);
    real fx, fy, fz, g, a, px, py, pz;
    const real pc = phi[c];
    px = (phi[IDX(i, j, lp)] - pc) / dx;
    py = (phi[IDX(i, jp, l)] + phi[IDX(i, jp, lp)] - phi[IDX(i, jm, l)] - phi[IDX(i, jm, lp)]) * q;
    pz = (phi[IDX(ip, j, l)] + phi[IDX(ip, j, lp)] - phi[IDX(im, j, l)] - phi[IDX(im, j, lp)]) * q;
    flux(px, py, pz, eps, Rm, fx, fy, fz, g, a);
    FX[c] = fx;
    NX[c] = px / sqrt(g + (real)1e-24);
    py = (phi[IDX(i, jp, l)] - pc) / dx;
    px = (phi[IDX(i, j, lp)] + phi[IDX(i, jp, lp)] - phi[IDX(i, j, lm)] - phi[IDX(i, jp, lm)]) * q;
    pz = (phi[IDX(ip, j, l)] + phi[IDX(ip, jp, l)] - phi[IDX(im, j, l)] - phi[IDX(im, jp, l)]) * q;
    flux(px, py, pz, eps, Rm, fx, fy, fz, g, a);
    FY[c] = fy;
    NY[c] = py / sqrt(g + (real)1e-24);
    if (i < nz - 1) {
        pz = (phi[IDX(i + 1, j, l)] - pc) / dx;
        px = (phi[IDX(i, j, lp)] + phi[IDX(i + 1, j, lp)] - phi[IDX(i, j, lm)] - phi[IDX(i + 1, j, lm)]) * q;
        py = (phi[IDX(i, jp, l)] + phi[IDX(i + 1, jp, l)] - phi[IDX(i, jm, l)] - phi[IDX(i + 1, jm, l)]) * q;
        flux(px, py, pz, eps, Rm, fx, fy, fz, g, a);
        FZ[c] = fz;
        NZ[c] = pz / sqrt(g + (real)1e-24);
    } else {
        FZ[c] = (real)0.0;
        NZ[c] = (real)0.0;
    }
}

// pass 2: d_t phi
extern "C" __global__ void pf3_rate(const real* __restrict__ phi, const real* __restrict__ U,
                                    const real* __restrict__ FX, const real* __restrict__ FY,
                                    const real* __restrict__ FZ, real* R, const real* __restrict__ Rm,
                                    const real* __restrict__ ws, const real* __restrict__ zc,
                                    const real eps, const real dx, const real offset, const real z0,
                                    const real Vt, const real t, const real lT, const real kbar, const real lam,
                                    const int ns, const int nz, const int ny, const int nx)
{
    DECODE
    const real theta = (zc[i] + offset - z0 - Vt * t) / lT;
    real tf = (real)1.0 - ((real)1.0 - kbar) * theta;
    tf = tf > kbar ? tf : kbar;
    const real fb = i > 0 ? FZ[IDX(i - 1, j, l)] : (real)0.0;
    const real div = (FX[c] - FX[IDX(i, j, lm)] + FY[c] - FY[IDX(i, jm, l)] + FZ[c] - fb) / dx;
    real fx, fy, fz, g, a;
    flux((phi[IDX(i, j, lp)] - phi[IDX(i, j, lm)]) / ((real)2.0 * dx),
         (phi[IDX(i, jp, l)] - phi[IDX(i, jm, l)]) / ((real)2.0 * dx),
         (phi[IDX(ip, j, l)] - phi[IDX(im, j, l)]) / ((real)2.0 * dx), eps, Rm, fx, fy, fz, g, a);
    real us = (real)0.0;
    for (int s = 0; s < ns; ++s) us += ws[s] * U[s * n + c];
    const real p = phi[c];
    const real g2 = ((real)1.0 - p * p) * ((real)1.0 - p * p);
    R[c] = (div + p - p * p * p - lam * g2 * (us + theta)) / (a * a * tf);
}

// pass 3: phi and solute updates
extern "C" __global__ void pf3_update(const real* __restrict__ phi, const real* __restrict__ U,
                                      const real* __restrict__ R, const real* __restrict__ NX,
                                      const real* __restrict__ NY, const real* __restrict__ NZ,
                                      real* phin, real* Un, const real* __restrict__ ks,
                                      const real Dt, const real at, const real dx, const real dt,
                                      const int ns, const int nz, const int ny, const int nx)
{
    DECODE
    const real pc = phi[c];
    const real rc = R[c];
    const real pn = pc + dt * rc;
    phin[c] = pn;
    const long long ce = IDX(i, j, lp), cw = IDX(i, j, lm), cn = IDX(i, jp, l), cs = IDX(i, jm, l);
    const long long ct = IDX(ip, j, l), cb = IDX(i > 0 ? i - 1 : 0, j, l);
    const real q = (real)0.25;
    for (int s = 0; s < ns; ++s) {
        const real k = ks[s];
        const real* Us = U + s * n;
        const real uc = Us[c];
        const real Fc = (real)1.0 + ((real)1.0 - k) * uc;
        const real je = (Dt * q * ((real)2.0 - pc - phi[ce]) * (Us[ce] - uc) / dx
                         + at * q * (Fc + (real)1.0 + ((real)1.0 - k) * Us[ce]) * (rc + R[ce]) * NX[c]);
        const real jw = (Dt * q * ((real)2.0 - phi[cw] - pc) * (uc - Us[cw]) / dx
                         + at * q * ((real)1.0 + ((real)1.0 - k) * Us[cw] + Fc) * (R[cw] + rc) * NX[cw]);
        const real jn = (Dt * q * ((real)2.0 - pc - phi[cn]) * (Us[cn] - uc) / dx
                         + at * q * (Fc + (real)1.0 + ((real)1.0 - k) * Us[cn]) * (rc + R[cn]) * NY[c]);
        const real js = (Dt * q * ((real)2.0 - phi[cs] - pc) * (uc - Us[cs]) / dx
                         + at * q * ((real)1.0 + ((real)1.0 - k) * Us[cs] + Fc) * (R[cs] + rc) * NY[cs]);
        real jt = (real)0.0;
        if (i < nz - 1)
            jt = (Dt * q * ((real)2.0 - pc - phi[ct]) * (Us[ct] - uc) / dx
                  + at * q * (Fc + (real)1.0 + ((real)1.0 - k) * Us[ct]) * (rc + R[ct]) * NZ[c]);
        real jb = (real)0.0;
        if (i > 0)
            jb = (Dt * q * ((real)2.0 - phi[cb] - pc) * (uc - Us[cb]) / dx
                  + at * q * ((real)1.0 + ((real)1.0 - k) * Us[cb] + Fc) * (R[cb] + rc) * NZ[cb]);
        Un[s * n + c] = uc + dt * ((je - jw + jn - js + jt - jb) / dx + (real)0.5 * Fc * rc)
                        / (((real)1.0 + k - ((real)1.0 - k) * pn) / (real)2.0);
    }
}
"""


def _cuda_kernels(dtype):
    """Compiled (faces, rate, update) kernels for float64 or float32 fields (cached)."""
    dtype = np.dtype(dtype)
    key = ("cuda", dtype.str)
    if key not in _KERNELS:
        import cupy as cp

        if dtype == np.float64:
            real = "double"
        elif dtype == np.float32:
            real = "float"
        else:
            raise ValueError(f"the cuda kernel supports float64 and float32 fields, not {dtype}")
        mod = cp.RawModule(code=_CUDA_SRC.replace("REAL", real), options=("--std=c++14",))
        _KERNELS[key] = tuple(mod.get_function(f) for f in ("pf3_faces", "pf3_rate", "pf3_update"))
    return _KERNELS[key]
