"""Strain-induced FCC -> HCP (epsilon) transformation: how much HCP forms with plastic strain.

Level 1 of the FCC -> HCP question (Level 0, tpmsfem.post.fcc_faults, says where faults can
extend). The HCP volume fraction follows an Olson-Cohen type law in the equivalent plastic
strain (Olson & Cohen, Metall. Trans. A 6 (1975) 791):

    f = f0 + (1 - f0) [1 - exp(-beta (1 - exp(-alpha eps_p))^n)],

with f0 the HCP present before straining (athermal martensite of the as-built material). In
Olson & Cohen's picture (1 - exp(-alpha eps_p)) is the volume fraction of shear bands and the
exponent n counts how many must meet to nucleate martensite (n ~ 4.5 for alpha' at shear-band
intersections). For epsilon martensite, which grows from overlapping stacking faults inside
single bands, a lower n is appropriate. Two shapes are offered:

* ``"exponential"``: n = 1 in the small-strain limit, f = f0 + (1 - f0)(1 - exp(-k eps_p));
* ``"sigmoidal"``: n = 2 with a given alpha (default 5): slow at small strain, then faster.

Both have one free rate constant, fitted with :meth:`StrainInducedHCP.fit` to one measured
(plastic strain, fraction) point. With a single data point the shape at small strains is not
constrained: use the two shapes as a band.

Coupling is one-way (Level 1a): the fraction is evaluated from the plastic-strain history of a
J2 model whose flow curve is the measured one (which already contains the TRIP hardening); the
transformation strain and its stress-state dependence (Stringfellow et al., Acta Metall. Mater.
40 (1992) 1703) are not included.

Calibration preset: LPBF Fe50Mn30Co10Cr10 (at%), room-temperature tension, EBSD: 0.94 % HCP as
built, 28.1 % at 20 % strain; yield 487.6 MPa, tensile strength 744.9 MPa, elongation 22.3 %
(Micromachines 15 (2024) 123, doi:10.3390/mi15010123, Table 1 and Fig. 16). The equiatomic
CoCrFeMnNi does not transform at room temperature (it twins), so f stays at 0 for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: LPBF Fe50Mn30Co10Cr10, room temperature (Micromachines 15 (2024) 123)
FEMNCOCR_LPBF = dict(source="Micromachines 15 (2024) 123, doi:10.3390/mi15010123", f0=0.0094, strain=0.20, f=0.281,
                     yield_MPa=487.6, uts_MPa=744.9, elongation=0.223)


@dataclass(frozen=True)
class StrainInducedHCP:
    """HCP fraction against equivalent plastic strain (see the module docstring)."""
    f0: float = 0.0
    beta: float = 0.0
    alpha: float = 5.0
    n: float = 2.0
    shape: str = "sigmoidal"          # or "exponential" (then beta is the rate k)

    def fraction(self, eps_p):
        e = np.maximum(np.asarray(eps_p, dtype=float), 0.0)
        if self.shape == "exponential":
            g = 1.0 - np.exp(-self.beta * e)
        else:
            g = 1.0 - np.exp(-self.beta * (1.0 - np.exp(-self.alpha * e)) ** self.n)
        return self.f0 + (1.0 - self.f0) * g

    @classmethod
    def fit(cls, eps_p: float, f: float, f0: float = 0.0, shape: str = "sigmoidal", alpha: float = 5.0,
            n: float = 2.0) -> "StrainInducedHCP":
        """The law of the given shape through (eps_p, f), starting at f0."""
        if not f0 <= f < 1.0:
            raise ValueError("need f0 <= f < 1")
        g = (f - f0) / (1.0 - f0)
        a = -np.log(1.0 - g)
        if shape == "exponential":
            return cls(f0=f0, beta=a / eps_p, alpha=alpha, n=1.0, shape="exponential")
        return cls(f0=f0, beta=a / (1.0 - np.exp(-alpha * eps_p)) ** n, alpha=alpha, n=n, shape="sigmoidal")


def femncocr_lpbf(shape: str = "sigmoidal", E_MPa: float = 200e3) -> StrainInducedHCP:
    """The law fitted to LPBF Fe50Mn30Co10Cr10 (plastic strain = 20 % - UTS/E)."""
    d = FEMNCOCR_LPBF
    eps_p = d["strain"] - d["uts_MPa"] / E_MPa
    return StrainInducedHCP.fit(eps_p, d["f"], d["f0"], shape=shape)
