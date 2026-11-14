"""Process-parameter criteria for porosity in laser powder-bed fusion.

* Lack of fusion: melt pools of neighbouring tracks and layers must overlap.
  - ``lof_criterion``: Tang, Pistorius & Beuth, Addit. Manuf. 14 (2017) 39: full fusion when
    (h / W)^2 + (L / D)^2 <= 1 (hatch h, layer L, melt-pool width W and depth D, D measured
    from the top of the powder layer).
  - ``lof_porosity``: the unfused area fraction of the cross-section, from half-elliptical
    melt pools repeated over hatch and layers (the geometric model behind the criterion), with
    optional track-to-track scatter of W and D.
* Keyhole: ``keyhole_number`` after Gan et al., Nat. Commun. 12 (2021) 2379 (see the docstring
  for the definition and the regime boundaries).

Melt-pool width and depth come from a melt-pool model (e.g. the calibrated conduction model
``MeltPoolModel``); the criteria themselves are geometric or dimensionless.
"""
from __future__ import annotations

import numpy as np


def lof_criterion(W, D, hatch, layer):
    """(h/W)^2 + (L/D)^2: <= 1 full fusion (Tang et al. 2017), > 1 lack of fusion expected."""
    W, D = np.asarray(W, float), np.asarray(D, float)
    with np.errstate(divide="ignore"):
        return (hatch / W) ** 2 + (layer / D) ** 2


def lof_porosity(W, D, hatch, layer, cv=0.0, n=400, samples=1, seed=0):
    """Unfused area fraction of the cross-section normal to the scan direction.

    Tracks sit at y = i * hatch in every layer, each layer ``layer`` higher; a track melts the
    half ellipse of width ``W`` and depth ``D`` below the top of its layer. A point is fused
    if any track of its own layer or a later one covers it. The representative cell is one
    hatch wide and one layer high, sampled on an n x n grid (scan rotation between layers is
    not modelled: a 2-D estimate). With ``cv`` > 0 every track's W and D are drawn from normal
    distributions with that coefficient of variation, and the result is averaged over
    ``samples`` random realisations."""
    rng = np.random.default_rng(seed)
    if W <= 0 or D <= 0:
        return 1.0
    y = (np.arange(n) + 0.5) / n * hatch
    z = -(np.arange(n) + 0.5) / n * layer                     # the cell: layer 0, z in (-L, 0)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    ni = int(np.ceil(0.5 * W * (1 + 4 * cv) / hatch)) + 1
    nj = int(np.ceil(D * (1 + 4 * cv) / layer)) + 1
    out = []
    for _ in range(max(samples, 1)):
        fused = np.zeros(Y.shape, bool)
        for j in range(0, nj + 1):                            # layer j has its top at j * L
            ztop = j * layer
            for i in range(-ni, ni + 2):
                w = W * (1.0 + cv * rng.standard_normal()) if cv > 0 else W
                d = D * (1.0 + cv * rng.standard_normal()) if cv > 0 else D
                if w <= 0 or d <= 0:
                    continue
                yc = i * hatch
                fused |= ((Y - yc) / (0.5 * w)) ** 2 + ((ztop - Z) / d) ** 2 <= 1.0
        out.append(1.0 - fused.mean())
    return float(np.mean(out))


def keyhole_number(P, v, absorptivity, radius, rho, cp, T_l, T0, alpha):
    """Keyhole number of Gan et al. (2021),

        Ke = eta P / ((T_l - T0) pi rho c_p sqrt(alpha v r0^3)),

    with eta the absorptivity, P the power [W], v the scan speed [m/s], r0 the laser spot
    radius [m], rho [kg/m^3], c_p [J/(kg K)], alpha the thermal diffusivity [m^2/s], T_l the
    liquidus and T0 the ambient/plate temperature. Gan et al. place the conduction-to-
    transition boundary near Ke ~ 1.4 and the onset of the keyhole regime near Ke ~ 6 (see
    their paper for the exact property choices)."""
    return absorptivity * P / ((T_l - T0) * np.pi * rho * cp * np.sqrt(alpha * v * radius ** 3))


def volumetric_energy_density(P, v, hatch, layer):
    """P / (v h t) [J/m^3] (divide by 1e9 for J/mm^3): the usual, much-debated, single number."""
    return P / (v * hatch * layer)
