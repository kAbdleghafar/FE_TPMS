"""Process-parameter criteria for porosity in laser powder-bed fusion.

* Lack of fusion: melt pools of neighbouring tracks and layers must overlap.
  - ``lof_criterion``: Tang, Pistorius & Beuth, Addit. Manuf. 14 (2017) 39: full fusion when
    (h / W)^2 + (L / D)^2 <= 1 (hatch h, layer L, melt-pool width W and depth D, D measured
    from the top of the powder layer).
  - ``lof_porosity``: the unfused area fraction of the cross-section, from half-elliptical
    melt pools repeated over hatch and layers (the geometric model behind the criterion), with
    optional track-to-track scatter of W and D.
* Keyhole: the scaling laws of Gan et al., Nat. Commun. 12 (2021) 2379, fitted to operando
  X-ray keyhole depths (Ti-6Al-4V, Al6061, SS316; 100-520 W, 0.3-1.2 m/s, r0 44-70 um):
  - ``keyhole_number`` Ke, and ``keyhole_absorptivity`` (their Eq. 6) for the absorptivity it
    needs;
  - ``keyhole_aspect_ratio`` / ``keyhole_depth`` (Eq. 2), ``keyhole_fluctuation`` (Eq. 5);
  - ``keyhole_regime`` (conduction / transition / keyhole) and ``keyhole_stability`` (stable /
    intermediate / chaotic);
  - ``normalized_energy_density`` and ``gan_porosity`` (Eqs. 8-9);
  - ``keyhole_state``: all of the above for arrays of power and speed.
  Their Ke uses liquid properties at the liquidus and the process absorptivity of Eq. 6, not
  the flat-surface one; the regime and stability thresholds hold only with that convention.

Melt-pool width and depth come from a melt-pool model (e.g. the calibrated conduction model
``MeltPoolModel``); the criteria themselves are geometric or dimensionless.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf, erfc

# Gan et al. (2021): Ke at the conduction -> transition and transition -> keyhole boundaries, and
# below which the keyhole is stable (small fluctuations, almost no pores) / above which chaotic
KE_TRANSITION, KE_KEYHOLE = 1.4, 6.0
KE_STABLE, KE_CHAOTIC = 16.0, 30.0


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
    liquidus and T0 the ambient/plate temperature. Gan et al. use liquid properties at the
    liquidus and eta from ``keyhole_absorptivity``; with those, conduction below Ke = 1.4,
    transition up to 6, keyhole above, stable below 16 and chaotic above 30."""
    return absorptivity * P / ((T_l - T0) * np.pi * rho * cp * np.sqrt(alpha * v * radius ** 3))


def volumetric_energy_density(P, v, hatch, layer):
    """P / (v h t) [J/m^3] (divide by 1e9 for J/mm^3): the usual, much-debated, single number."""
    return P / (v * hatch * layer)


def _regime(Ke, edges, names):
    Ke = np.asarray(Ke, float)
    idx = np.searchsorted(np.asarray(edges), Ke, side="left")      # edges belong to the lower class
    return names[int(idx)] if idx.ndim == 0 else np.asarray(names, dtype=object)[idx]


def keyhole_aspect_ratio(Ke):
    """e* = e / r0 = 0.4 (Ke - 1.4) (Gan et al. Eq. 2): keyhole depth e over the spot radius r0;
    zero in the conduction mode (Ke <= 1.4)."""
    return np.maximum(0.4 * (np.asarray(Ke, float) - KE_TRANSITION), 0.0)


def keyhole_depth(Ke, radius):
    """Keyhole (vapour-depression) depth e = r0 e*(Ke) below the original surface [unit of r0]."""
    return radius * keyhole_aspect_ratio(Ke)


def keyhole_fluctuation(Ke):
    """Delta e* = 0.36 Ke^0.86 (Gan et al. Eq. 5): the range (max - min) of the aspect ratio
    over time in their X-ray series, i.e. how far the keyhole tip moves up and down. Zero in the
    conduction mode, where there is no keyhole."""
    Ke = np.asarray(Ke, float)
    return np.where(Ke > KE_TRANSITION, 0.36 * np.abs(Ke) ** 0.86, 0.0)


def keyhole_regime(Ke):
    """'conduction' (Ke <= 1.4), 'transition' (<= 6) or 'keyhole' (Gan et al.)."""
    return _regime(Ke, (KE_TRANSITION, KE_KEYHOLE), ("conduction", "transition", "keyhole"))


def keyhole_stability(Ke):
    """'stable' (Ke <= 16: small fluctuations, almost no pores or spatter), 'intermediate' or
    'chaotic' (Ke > 30: pores and spatter) after Gan et al. The thresholds are rounded values
    from their X-ray series, not sharp limits."""
    return _regime(Ke, (KE_STABLE, KE_CHAOTIC), ("stable", "intermediate", "chaotic"))


def normalized_diffusion_length(v, radius, alpha):
    """L_d* = sqrt(alpha / (v r0)): thermal diffusion length over the spot radius (Gan et al.)."""
    return np.sqrt(alpha / (np.asarray(v, float) * radius))


def keyhole_porosity_number(P, v, radius, eta_min, rho, cp, T_l, T0):
    """Ke_m L_d* = eta_m P / ((T_l - T0) pi rho c_p v r0^2) (Gan et al. Eq. 7): the keyhole number
    with the flat-melt absorptivity eta_m times L_d*; the diffusivity cancels. It sets the
    process absorptivity (Eq. 6) and the keyhole-porosity term of Eq. 8."""
    return eta_min * np.asarray(P, float) / ((T_l - T0) * np.pi * rho * cp * np.asarray(v, float) * radius ** 2)


def keyhole_absorptivity(P, v, radius, eta_min, rho, cp, T_l, T0, floor=True):
    """Process absorptivity eta = 0.7 [1 - exp(-0.6 Ke_m L_d*)] (Gan et al. Eq. 6), fitted to
    ray-tracing simulations and measured absorptivities: it rises from the flat-melt value
    eta_m towards 0.7 as multiple reflections set in inside the depression. The fit falls below
    eta_m at low power; ``floor`` returns max(eta, eta_m) there, as their Supplementary Data 1
    does. eta_m (Gan et al. Supplementary Table 1): Ti-6Al-4V 0.26, SS316 0.33, Al6061 0.01."""
    eta = 0.7 * (1.0 - np.exp(-0.6 * keyhole_porosity_number(P, v, radius, eta_min, rho, cp, T_l, T0)))
    return np.maximum(eta, eta_min) if floor else eta


def normalized_energy_density(P, v, hatch, layer, eta_min, rho, cp, T_l, T0):
    """NED = eta_m P / (v h L) / (rho c_p (T_l - T0)) (Gan et al. Eq. 9): absorbed volumetric
    energy over the sensible heat of melting."""
    return eta_min * np.asarray(P, float) / (np.asarray(v, float) * hatch * layer) / (rho * cp * (T_l - T0))


def gan_porosity(NED, KeL):
    """phi = 0.10 erfc(NED - 1.51) + 0.01 erf(Ke_m L_d* - 2.32) + 0.055 (Gan et al. Eq. 8),
    fitted to porosity data of six LPBF studies (Ti-6Al-4V, IN718, 316L). The first term is
    lack of fusion, the second keyhole porosity. The paper does not state the unit of phi; use
    the two midpoints, NED = 1.51 and Ke_m L_d* = 2.32, as the transitions rather than the
    absolute values."""
    return 0.10 * erfc(np.asarray(NED, float) - 1.51) + 0.01 * erf(np.asarray(KeL, float) - 2.32) + 0.055


def keyhole_state(P, v, radius, *, eta_min, rho, cp, alpha, T_l, T0, absorptivity=None):
    """Gan et al.'s keyhole laws for arrays of power P [W] and speed v [m/s] (broadcast), spot
    radius r0 [m], liquid properties at the liquidus (SI) and plate/part temperature T0.

    The absorptivity is ``keyhole_absorptivity`` unless a fixed ``absorptivity`` is given.
    Returns a dict of arrays: eta, Ke, aspect (e*), depth [m], fluctuation (Delta e*),
    depth_min / depth_max [m] (e -/+ r0 Delta e* / 2, floored at zero), KeL (Ke_m L_d*),
    Ld (L_d*), regime and stability."""
    P, v = np.broadcast_arrays(np.asarray(P, float), np.asarray(v, float))
    eta = (keyhole_absorptivity(P, v, radius, eta_min, rho, cp, T_l, T0) if absorptivity is None
           else np.full(P.shape, float(absorptivity)))
    Ke = keyhole_number(P, v, eta, radius, rho, cp, T_l, T0, alpha)
    e = keyhole_depth(Ke, radius)
    de = keyhole_fluctuation(Ke)
    return dict(eta=eta, Ke=Ke, aspect=keyhole_aspect_ratio(Ke), depth=e, fluctuation=de,
                depth_min=np.maximum(e - 0.5 * radius * de, 0.0), depth_max=e + 0.5 * radius * de,
                KeL=keyhole_porosity_number(P, v, radius, eta_min, rho, cp, T_l, T0),
                Ld=normalized_diffusion_length(v, radius, alpha),
                regime=keyhole_regime(Ke), stability=keyhole_stability(Ke))


def keyhole_power(Ke, v, radius, *, eta_min, rho, cp, alpha, T_l, T0, P_max=5000.0):
    """Laser power [W] that gives the keyhole number ``Ke`` at scan speed ``v`` [m/s], with the
    process absorptivity of Eq. 6 (monotone in P, solved by bisection). Keeping Ke constant
    while the beam slows down (a turn point, the end of a track) keeps the keyhole depth and
    stability of the steady track: a power schedule that avoids the deepening and collapse
    behind turn-point pores."""
    def ke(P):
        eta = keyhole_absorptivity(P, v, radius, eta_min, rho, cp, T_l, T0)
        return keyhole_number(P, v, eta, radius, rho, cp, T_l, T0, alpha)

    lo, hi = 0.0, float(P_max)
    if ke(hi) < Ke:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ke(mid) < Ke else (lo, mid)
    return 0.5 * (lo + hi)
