"""Material definitions (units: N, mm, MPa).

The presets are *typical literature values* intended for getting started and
for code-to-code benchmarks. For device work, replace them with properties
measured on your own as-built material (build orientation, heat treatment and
surface condition of LPBF parts change them substantially).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constitutive import NO_YIELD


@dataclass(frozen=True)
class Material:
    name: str
    E: float                       # Young's modulus [MPa]
    nu: float                      # Poisson's ratio
    yield_stress: Optional[float] = None   # initial von Mises yield [MPa]; None => elastic
    hardening: float = 0.0         # linear isotropic hardening modulus H [MPa]
    density: float = 0.0           # [t/mm^3], informative

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
        return self.yield_stress is not None

    @property
    def sy0(self) -> float:
        return self.yield_stress if self.yield_stress is not None else NO_YIELD

    def elastic(self) -> "Material":
        return Material(self.name, self.E, self.nu, None, 0.0, self.density)


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
