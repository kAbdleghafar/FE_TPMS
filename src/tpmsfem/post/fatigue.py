"""Strain-life fatigue post-processing (uniaxial, vectorised).

* stabilised cyclic stress-strain curve (Ramberg-Osgood)
  ``eps_a = sigma_a / E + (sigma_a / K') ** (1 / n')``;
* Basquin-Coffin-Manson strain-life ``eps_a = sigma_f'/E (2N)^b + eps_f' (2N)^c``,
  optionally with the Morrow (``sigma_f' - sigma_m``) or Smith-Watson-Topper
  (``sigma_max eps_a = sigma_f'^2/E (2N)^2b + sigma_f' eps_f' (2N)^(b+c)``)
  mean-stress corrections -- e.g. with the residual stress of an AM build as the
  mean stress;
* Neuber's rule for notch-root amplitudes from an elastic FE amplitude.

Parameters are material/process specific (as-built LPBF data differ from
wrought); :data:`IN625_LPBF_CONCLI2024` holds the as-built LPBF IN625 values
of Concli et al. (2024), doi:10.1007/s40964-023-00545-1, as listed in the IN625
data guide (R = -1, 0.1 Hz, 5 specimens) -- a literature starting point, not a
design curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StrainLife:
    E: float
    sigma_f: float
    b: float
    eps_f: float
    c: float
    K_cyc: float = np.nan        # cyclic strength coefficient K'
    n_cyc: float = np.nan        # cyclic hardening exponent n'

    # ---------------------------------------------------------- cyclic curve
    def strain_amplitude(self, sigma_a):
        """Total strain amplitude of the stabilised cyclic curve."""
        s = np.asarray(sigma_a, float)
        return s / self.E + (np.abs(s) / self.K_cyc) ** (1.0 / self.n_cyc)

    def stress_amplitude(self, eps_a, iters: int = 60):
        """Inverse of :meth:`strain_amplitude` (vectorised bisection)."""
        e = np.asarray(eps_a, float)
        lo, hi = np.zeros_like(e), np.minimum(self.E * e, self.K_cyc * np.maximum(e, 1e-12) ** self.n_cyc) + 1.0
        hi = np.maximum(hi, self.E * e)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            up = self.strain_amplitude(mid) > e
            hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
        return 0.5 * (lo + hi)

    def neuber(self, sigma_a_elastic):
        """Notch-root (sigma_a, eps_a) from a linear-elastic amplitude:
        sigma_a eps_a = sigma_el^2 / E on the cyclic curve."""
        target = np.asarray(sigma_a_elastic, float) ** 2 / self.E
        lo, hi = np.zeros_like(target), np.abs(np.asarray(sigma_a_elastic, float)) + 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            up = mid * self.strain_amplitude(mid) > target
            hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
        s = 0.5 * (lo + hi)
        return s, self.strain_amplitude(s)

    # ---------------------------------------------------------- strain life
    def strain_from_life(self, N, sigma_m=0.0, correction: str = "none"):
        rev = 2.0 * np.asarray(N, float)
        sf = self.sigma_f - (np.asarray(sigma_m, float) if correction == "morrow" else 0.0)
        return sf / self.E * rev**self.b + self.eps_f * rev**self.c

    def life(self, eps_a, sigma_m=0.0, correction: str = "none", sigma_max=None, N_max: float = 1e12):
        """Cycles to failure N_f for strain amplitude(s) ``eps_a``.

        ``correction``: ``"none"``, ``"morrow"`` (mean stress ``sigma_m``) or
        ``"swt"`` (uses ``sigma_max``, default ``sigma_m`` + the cyclic-curve
        amplitude). Lives above ``N_max`` are reported as ``N_max``."""
        e = np.asarray(eps_a, float)
        sm = np.broadcast_to(np.asarray(sigma_m, float), e.shape)
        if correction == "swt":
            smax = (sm + self.stress_amplitude(e)) if sigma_max is None else np.broadcast_to(sigma_max, e.shape)

            def f(logN):
                rev = 2.0 * 10.0**logN
                return (self.sigma_f**2 / self.E * rev ** (2 * self.b)
                        + self.sigma_f * self.eps_f * rev ** (self.b + self.c)) - np.maximum(smax, 1e-12) * e
        elif correction in ("none", "morrow"):
            def f(logN):
                return self.strain_from_life(10.0**logN, sm, correction) - e
        else:
            raise ValueError(correction)
        lo, hi = np.full(e.shape, -1.0), np.full(e.shape, np.log10(N_max))
        for _ in range(100):                     # strain(N) decreases with N: bisection in log N
            mid = 0.5 * (lo + hi)
            pos = f(mid) > 0
            lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
        return 10.0 ** (0.5 * (lo + hi))


IN625_LPBF_CONCLI2024 = StrainLife(E=210_000.0, sigma_f=500.0, b=-0.050, eps_f=0.077, c=-0.178,
                                   K_cyc=1141.0, n_cyc=0.1139)
