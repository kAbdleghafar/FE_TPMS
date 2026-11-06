"""Material definitions (units: N, mm, MPa, s, t/mm^3, K or degC).

The presets are *typical literature values* intended for getting started and
for code-to-code benchmarks. For device work, replace them with properties
measured on your own as-built material (build orientation, heat treatment and
surface condition of LPBF parts change them substantially).

Beyond linear elasticity with linear isotropic hardening, a :class:`Material`
can carry

* ``flow`` -- a :class:`FlowTable`: multilinear (tabulated) isotropic
  hardening ``sigma_y(eps_p[, T])``, as Abaqus' ``*PLASTIC`` table;
* ``temperature`` -- a :class:`TemperatureTable`: ``E(T)``, ``nu(T)`` and the
  mean thermal-expansion coefficient ``alpha(T)``;
* ``thermal`` -- :class:`ThermalProperties` for the heat solver
  (:mod:`tpmsfem.thermal`): conductivity, specific heat, density, latent heat;
* ``anneal_temperature`` -- above it the equivalent plastic strain (hardening
  memory) is reset, as Abaqus' ``*ANNEAL TEMPERATURE``.

Temperature tables are interpolated linearly and *held constant* outside
their range (no extrapolation).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .constitutive import NO_YIELD


def _arr(v):
    return None if v is None else np.atleast_1d(np.asarray(v, float))


@dataclass(frozen=True)
class FlowTable:
    """Multilinear isotropic hardening: yield stress vs equivalent plastic strain.

    ``eps_p`` (K,) is increasing with ``eps_p[0] == 0``; ``sigma`` is (K,) or
    (nT, K) for the temperatures ``T`` (nT,). Beyond the last point the curve
    continues with ``tail_slope`` (0 = perfectly plastic, the Abaqus
    convention). Build one from measured curves with :meth:`from_curves`.
    """
    eps_p: np.ndarray = field(compare=False)
    sigma: np.ndarray = field(compare=False)
    T: Optional[np.ndarray] = field(default=None, compare=False)
    tail_slope: float = 0.0
    name: str = ""

    def __post_init__(self):
        ep, s, T = _arr(self.eps_p), np.asarray(self.sigma, float), _arr(self.T)
        if s.ndim == 1:
            s = s[None, :]
        if T is None:
            T = np.zeros(1)
        if ep[0] != 0.0 or np.any(np.diff(ep) <= 0):
            raise ValueError("eps_p must start at 0 and increase strictly")
        if s.shape != (len(T), len(ep)):
            raise ValueError(f"sigma has shape {s.shape}, expected {(len(T), len(ep))}")
        if len(T) > 1 and np.any(np.diff(T) <= 0):
            raise ValueError("T must increase strictly")
        object.__setattr__(self, "eps_p", ep)
        object.__setattr__(self, "sigma", s)
        object.__setattr__(self, "T", T)

    @classmethod
    def from_curves(cls, curves: dict, eps_p=None, tail_slope: float = 0.0, name: str = ""):
        """``curves = {T: (eps_p, sigma), ...}`` (one or more temperatures).

        The curves are resampled on a common plastic-strain grid (the union of
        their points, or ``eps_p``); a curve is extended flat beyond its last
        measured point."""
        Ts = sorted(curves)
        grid = np.unique(np.concatenate([np.asarray(curves[t][0], float) for t in Ts])) if eps_p is None \
            else np.asarray(eps_p, float)
        if grid[0] != 0.0:
            grid = np.r_[0.0, grid]
        S = np.array([np.interp(grid, np.asarray(curves[t][0], float), np.asarray(curves[t][1], float))
                      for t in Ts])
        return cls(grid, S, np.array(Ts, float), tail_slope, name)

    def at(self, T=None) -> np.ndarray:
        """Hardening curve (K,) at temperature T (clamped to the table range)."""
        if len(self.T) == 1 or T is None:
            return self.sigma[0].copy()
        return np.array([np.interp(T, self.T, self.sigma[:, k]) for k in range(len(self.eps_p))])

    def __call__(self, eps_p, T=None):
        s = self.at(T)
        e = np.asarray(eps_p, float)
        return np.where(e <= self.eps_p[-1], np.interp(e, self.eps_p, s),
                        s[-1] + self.tail_slope * (e - self.eps_p[-1]))


@dataclass(frozen=True)
class TemperatureTable:
    """Temperature-dependent elasticity and thermal expansion.

    ``alpha`` is the *mean* (secant) expansion coefficient referred to
    ``T_ref`` (Abaqus ``*EXPANSION, ZERO=T_ref``): the thermal strain is
    ``alpha(T) (T - T_ref)`` minus its value at the stress-free temperature.
    Instantaneous (tangent) coefficients ``a(T)`` must first be converted with
    :meth:`mean_from_instantaneous`. Above ``T_cap`` (e.g. the solidus) the
    thermal strain is frozen: molten material carries no stress and its
    shrinkage is not tracked.
    """
    T: np.ndarray = field(compare=False)
    E: np.ndarray = field(compare=False)
    alpha: Optional[np.ndarray] = field(default=None, compare=False)
    nu: Optional[np.ndarray] = field(default=None, compare=False)
    T_ref: float = 20.0
    T_cap: Optional[float] = None

    def __post_init__(self):
        for k in ("T", "E", "alpha", "nu"):
            object.__setattr__(self, k, _arr(getattr(self, k)))
        n = len(self.T)
        for k in ("E", "alpha", "nu"):
            v = getattr(self, k)
            if v is not None and len(v) != n:
                raise ValueError(f"{k} has {len(v)} values for {n} temperatures")
        if n > 1 and np.any(np.diff(self.T) <= 0):
            raise ValueError("T must increase strictly")

    @staticmethod
    def mean_from_instantaneous(T, a_inst, T_ref):
        """Mean CTE alpha(T) = 1/(T - T_ref) int_{T_ref}^T a(t) dt (trapezoidal)."""
        T, a = np.asarray(T, float), np.asarray(a_inst, float)
        tt = np.union1d(T, [T_ref])
        aa = np.interp(tt, T, a)
        cum = np.r_[0.0, np.cumsum(0.5 * (aa[1:] + aa[:-1]) * np.diff(tt))]
        cum -= np.interp(T_ref, tt, cum)
        out = np.interp(T, tt, cum) / np.where(np.abs(T - T_ref) > 1e-9, T - T_ref, 1.0)
        return np.where(np.abs(T - T_ref) > 1e-9, out, np.interp(T_ref, T, a))

    def E_at(self, T):
        return np.interp(T, self.T, self.E)

    def nu_at(self, T, default):
        return default if self.nu is None else np.interp(T, self.T, self.nu)

    def thermal_strain(self, T):
        """Total (isotropic) thermal strain alpha(T) (T - T_ref), frozen above T_cap."""
        if self.alpha is None:
            return np.zeros_like(np.asarray(T, float))
        Tc = np.asarray(T, float) if self.T_cap is None else np.minimum(T, self.T_cap)
        return np.interp(Tc, self.T, self.alpha) * (Tc - self.T_ref)


@dataclass(frozen=True)
class ThermalProperties:
    """Heat-conduction properties in the code's unit system
    (mm, s, t, N; energy mJ = N mm; power mW; temperature K or degC):
    conductivity ``k`` [mW/(mm K)] (numerically equal to W/(m K)), specific
    heat ``cp`` [mm^2/(s^2 K)] (= J/(kg K) x 1e6), density ``rho`` [t/mm^3],
    latent heat ``latent`` [mm^2/s^2] (= J/kg x 1e6) released linearly between
    ``T_solidus`` and ``T_liquidus``. Use :meth:`from_si` to enter SI data."""
    T: np.ndarray = field(compare=False)
    k: np.ndarray = field(compare=False)
    cp: np.ndarray = field(compare=False)
    rho: float = 0.0
    latent: float = 0.0
    T_solidus: float = np.inf
    T_liquidus: float = np.inf

    def __post_init__(self):
        for k in ("T", "k", "cp"):
            object.__setattr__(self, k, _arr(getattr(self, k)))

    @classmethod
    def from_si(cls, T, k_W_mK, cp_J_kgK, rho_kg_m3, latent_J_kg=0.0, T_solidus=np.inf, T_liquidus=np.inf):
        return cls(T, k_W_mK, np.asarray(cp_J_kgK, float) * 1e6, rho_kg_m3 * 1e-12,
                   latent_J_kg * 1e6, T_solidus, T_liquidus)

    def conductivity(self, T):
        return np.interp(T, self.T, self.k)

    def enthalpy_table(self, T_lo=-300.0, T_hi=4000.0, dT=1.0):
        """(T grid, volumetric enthalpy [mJ/mm^3]) with H(0) = 0, latent heat included."""
        tt = np.arange(T_lo, T_hi + dT, dT)
        rc = self.rho * np.interp(tt, self.T, self.cp)
        H = np.r_[0.0, np.cumsum(0.5 * (rc[1:] + rc[:-1]) * dT)]
        if self.latent > 0 and np.isfinite(self.T_solidus):
            span = max(self.T_liquidus - self.T_solidus, 1e-6)
            H = H + self.rho * self.latent * np.clip((tt - self.T_solidus) / span, 0.0, 1.0)
        H -= np.interp(0.0, tt, H)
        return tt, H


@dataclass(frozen=True)
class Material:
    name: str
    E: float                       # Young's modulus [MPa] (room temperature if ``temperature`` is set)
    nu: float                      # Poisson's ratio
    yield_stress: Optional[float] = None   # initial von Mises yield [MPa]; None => elastic
    hardening: float = 0.0         # linear isotropic hardening modulus H [MPa]
    density: float = 0.0           # [t/mm^3], informative
    flow: Optional[FlowTable] = None                 # tabulated hardening (overrides yield_stress/hardening)
    temperature: Optional[TemperatureTable] = None   # E(T), nu(T), alpha(T)
    thermal: Optional[ThermalProperties] = None      # heat-conduction properties
    anneal_temperature: Optional[float] = None       # reset of the equivalent plastic strain above it

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @property
    def bulk(self) -> float:
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def is_plastic(self) -> bool:
        return self.yield_stress is not None or self.flow is not None

    @property
    def sy0(self) -> float:
        if self.flow is not None:
            return float(self.flow.sigma[0, 0])
        return self.yield_stress if self.yield_stress is not None else NO_YIELD

    def elastic(self) -> "Material":
        return replace(self, yield_stress=None, hardening=0.0, flow=None, anneal_temperature=None)

    def flow_table(self) -> FlowTable:
        """The hardening law as a table (linear hardening -> 1 point + tail slope;
        elastic -> a yield stress that is never reached)."""
        if self.flow is not None:
            return self.flow
        return FlowTable([0.0], [self.sy0], None, self.hardening if self.yield_stress is not None else 0.0)


# Typical values -- see module docstring.
TI6AL4V_LPBF = Material("Ti6Al4V_LPBF", E=110_000.0, nu=0.32, yield_stress=950.0, hardening=1_000.0, density=4.43e-9)
COCRMO = Material("CoCrMo", E=210_000.0, nu=0.30, yield_stress=700.0, hardening=2_000.0, density=8.3e-9)
SS316L = Material("316L", E=193_000.0, nu=0.30, yield_stress=450.0, hardening=1_500.0, density=8.0e-9)
TANTALUM = Material("Ta", E=186_000.0, nu=0.34, yield_stress=300.0, hardening=1_000.0, density=16.6e-9)
CORTICAL_BONE = Material("cortical_bone", E=17_000.0, nu=0.30)
CANCELLOUS_BONE = Material("cancellous_bone", E=500.0, nu=0.30)
PMMA_CEMENT = Material("PMMA", E=2_500.0, nu=0.30)

PRESETS = {m.name: m for m in [TI6AL4V_LPBF, COCRMO, SS316L, TANTALUM, CORTICAL_BONE,
                               CANCELLOUS_BONE, PMMA_CEMENT]}
