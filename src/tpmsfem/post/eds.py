"""Synthetic EDS (energy-dispersive X-ray spectroscopy) element maps of simulated microstructures.

Turns 3-D composition fields (wt%, e.g. from :mod:`tpmsfem.process.phasefield3d`) into the maps
an SEM or a (S)TEM would record on a section, so that simulations can be compared with
measured maps at the measured resolution:

* **SEM** (bulk sample, polished section): X-rays come from a generation volume below the
  surface whose size is set by the beam energy and the line: the Anderson-Hasler range
  ``R = 0.064 (E0^1.68 - Ec^1.68) / rho`` um (keV, g/cm^3). Model of the volume: depth weight
  ``exp(-((d - 0.25 R) / 0.3 R)^2)`` for ``0 <= d <= R`` (a phi(rho z)-like profile), and at depth
  ``d`` a lateral Gaussian of ``sigma(d) = 0.92 R (d/R)^1.5`` (the beam spreads as depth^1.5, as in
  single-scattering broadening). The constant puts 90 % of the signal within a diameter ``R``
  (the usual statement of SEM-EDS resolution); half of it comes from within a diameter 0.28 R,
  so the point-spread function is cusped. The contrast kept for features finer than ``R``
  depends on this shape: a Gaussian with the same 90 % diameter keeps far less. Treat the maps
  as estimates and calibrate on a measured line scan where one exists. At 15 kV in a Ni
  alloy ``R`` is ~0.7 um for Nb L and Mo L (twice the LPBF cell spacing); at 5 kV it is below
  0.1 um. The line is the K line if the overvoltage ``E0/Ec >= 1.5``, else the L line.
* **STEM** (thin foil): the signal is the average through the foil thickness at the beam
  position, widened by beam broadening (single scattering, Goldstein et al.:
  ``b = 6.25e5 (Z/E0) (rho/A)^0.5 t^1.5`` cm, E0 in eV, t in cm; half of the exit value is used as
  the FWHM) and the probe size.

``EDSMap.resolution_m``: the diameter that contains 90 % of the signal of a point.

Counting statistics (optional): counts = Poisson(N c/100 + background), with ``N`` the counts
per pixel for a pure element and a background ``background * N * mean(c)/100`` (continuum
under the peak), as in a raw window-integral map. Concentration to intensity is taken as
linear (no matrix (ZAF) corrections).

Arrays are indexed (z, y, x) as in the phase-field results; x and y are periodic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

#: X-ray lines used for mapping: element -> [(line, energy keV, critical excitation energy keV)],
#: K line first.
LINES = {
    "Ni": [("Ka", 7.478, 8.333), ("La", 0.851, 0.855)],
    "Fe": [("Ka", 6.404, 7.112), ("La", 0.705, 0.708)],
    "Cr": [("Ka", 5.415, 5.989), ("La", 0.573, 0.574)],
    "Mn": [("Ka", 5.899, 6.539), ("La", 0.637, 0.639)],
    "Co": [("Ka", 6.930, 7.709), ("La", 0.776, 0.779)],
    "Ti": [("Ka", 4.511, 4.966), ("La", 0.452, 0.454)],
    "Al": [("Ka", 1.487, 1.560)],
    "Si": [("Ka", 1.740, 1.839)],
    "Nb": [("Ka", 16.615, 18.986), ("La", 2.166, 2.371)],
    "Mo": [("Ka", 17.479, 20.000), ("La", 2.293, 2.520)],
}
#: atomic number and mass (for beam broadening)
ATOMS = {"Ni": (28, 58.69), "Fe": (26, 55.85), "Cr": (24, 52.00), "Mn": (25, 54.94), "Co": (27, 58.93),
         "Ti": (22, 47.87), "Al": (13, 26.98), "Si": (14, 28.09), "Nb": (41, 92.91), "Mo": (42, 95.95)}
#: map colours (black -> colour), chosen to stay distinct in an overlay
COLOURS = {"Nb": (1.0, 0.22, 0.18), "Mo": (1.0, 0.78, 0.10), "Cr": (0.25, 0.55, 1.0), "Fe": (0.95, 0.35, 0.95),
           "Ni": (0.30, 0.90, 0.40), "Mn": (0.20, 0.90, 0.95), "Co": (0.65, 0.45, 1.0), "Ti": (1.0, 0.55, 0.2)}


def select_line(element: str, E0_keV: float, min_overvoltage: float = 1.5):
    """The line mapped at beam energy ``E0_keV``: K if E0/Ec >= ``min_overvoltage``, else L."""
    for line in LINES[element]:
        if E0_keV / line[2] >= min_overvoltage:
            return line
    for line in LINES[element]:
        if E0_keV / line[2] > 1.05:
            return line
    raise ValueError(f"{element}: no line excited at {E0_keV} kV")


def generation_range_um(E0_keV: float, Ec_keV: float, density: float) -> float:
    """Anderson-Hasler X-ray generation range [um] (keV, g/cm^3)."""
    return 0.064 * (E0_keV ** 1.68 - Ec_keV ** 1.68) / density


def beam_broadening_m(foil_m: float, E0_keV: float, Z: float, A: float, density: float) -> float:
    """Single-scattering beam broadening at the exit of a foil [m] (Goldstein et al.)."""
    t_cm = foil_m * 100.0
    return 6.25e5 * (Z / (E0_keV * 1e3)) * np.sqrt(density / A) * t_cm ** 1.5 / 100.0


@dataclass
class EDSMap:
    """Element maps of one section: ``maps[element]`` (2-D, counts or wt%-equivalent)."""
    maps: dict
    lines: dict
    resolution_m: dict                 # diameter containing 90 % of a point's signal, per element [m]
    pixel_m: float
    mode: str
    E0_keV: float
    axes: tuple = ("y", "x")
    info: dict = field(default_factory=dict)

    @property
    def extent_m(self):
        a = next(iter(self.maps.values()))
        return a.shape[1] * self.pixel_m, a.shape[0] * self.pixel_m

    def scaled(self, element: str, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
        """Map scaled to [0, 1] between the ``lo`` and ``hi`` percentiles (contrast stretch)."""
        a = self.maps[element]
        p0, p1 = np.percentile(a, [lo, hi])
        return np.clip((a - p0) / max(p1 - p0, 1e-12), 0.0, 1.0)

    def rgb(self, element: str, colour=None, **kw) -> np.ndarray:
        """Black-to-colour image of one element (as an EDS map display)."""
        c = np.asarray(colour if colour is not None else COLOURS.get(element, (1.0, 1.0, 1.0)))
        return self.scaled(element, **kw)[..., None] * c[None, None, :]

    def overlay(self, elements: Sequence[str], **kw) -> np.ndarray:
        """Layered image: the element images added and clipped (as 'layered image' displays)."""
        out = sum(self.rgb(e, **kw) for e in elements)
        return np.clip(out, 0.0, 1.0)


def _bin(a: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return a
    n0, n1 = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    return a[:n0, :n1].reshape(n0 // f, f, n1 // f, f).mean(axis=(1, 3))


def _noise(maps, total_counts, background, rng):
    out = {}
    for e, a in maps.items():
        mean = total_counts * a / 100.0
        bg = background * total_counts * float(a.mean()) / 100.0
        out[e] = rng.poisson(np.maximum(mean + bg, 0.0)).astype(float)
    return out


def _move(f, axis):
    """(normal, a, b) view and the periodic flags of (a, b) for (z, y, x) arrays."""
    per = {0: False, 1: True, 2: True}
    rest = [a for a in (0, 1, 2) if a != axis]
    return np.moveaxis(f, [axis] + rest, [0, 1, 2]), (per[rest[0]], per[rest[1]]), per[axis]


#: depth-resolved generation volume (fractions of the X-ray range R): see the module docstring
SEM_DEPTH_CENTRE, SEM_DEPTH_WIDTH, SEM_SPREAD = 0.25, 0.3, 0.92


def sem_psf_transfer(wavelength_m: float, R_m: float, n: int = 4001) -> float:
    """Fraction of the amplitude of a sinusoidal composition (uniform in depth) of wavelength
    ``wavelength_m`` that the SEM generation volume of range ``R_m`` keeps."""
    d = np.linspace(0.0, 1.0, n)
    w = np.exp(-((d - SEM_DEPTH_CENTRE) / SEM_DEPTH_WIDTH) ** 2)
    sig = SEM_SPREAD * R_m * d ** 1.5
    return float(np.sum(w * np.exp(-0.5 * (2.0 * np.pi * sig / wavelength_m) ** 2)) / w.sum())


def sem_eds(fields: dict, dx_m: float, index: int, axis: int = 0, beam_kV: float = 15.0, density: float = 8.44,
            into: int = -1, total_counts: Optional[float] = None, background: float = 0.2,
            pixel_m: Optional[float] = None, seed: int = 0, elements: Optional[Sequence[str]] = None,
            probe_m: float = 5e-9, n_bins: int = 24) -> EDSMap:
    """SEM-EDS maps of the section ``index`` normal to ``axis`` (0: z, a transverse section; 1: y;
    2: x) of the composition ``fields`` (element -> (z, y, x) wt%). ``into`` (+1 or -1): side of
    the section where the bulk lies (the beam goes that way). ``total_counts``: counts per pixel
    for 100 wt% (None: no counting noise). Depth layers are grouped into ``n_bins`` groups of
    similar lateral spread, each blurred once."""
    from scipy import ndimage

    elements = list(fields) if elements is None else list(elements)
    rng = np.random.default_rng(seed)
    maps, lines, res = {}, {}, {}
    notes = {}
    for e in elements:
        line = select_line(e, beam_kV)
        R = generation_range_um(beam_kV, line[2], density) * 1e-6
        f, per2, per_n = _move(np.asarray(fields[e], dtype=float), axis)
        n = f.shape[0]
        nd = max(int(np.ceil(R / dx_m)), 1)
        d = (np.arange(nd) + 0.5) * dx_m / R                       # layer depth / R
        w = np.exp(-((d - SEM_DEPTH_CENTRE) / SEM_DEPTH_WIDTH) ** 2)
        sig = np.sqrt(probe_m ** 2 + (SEM_SPREAD * R * d ** 1.5) ** 2) / dx_m
        idx = index + into * np.arange(nd)
        if per_n:
            idx %= n
        else:
            ok = (idx >= 0) & (idx < n)
            if not ok.all():
                notes[e] = f"generation volume truncated at {ok.sum() / nd:.0%} of its depth by the end of the block"
            idx, w, sig = idx[ok], w[ok], sig[ok]
        modes = ["wrap" if p else "nearest" for p in per2]
        acc = np.zeros(f.shape[1:])
        groups = np.array_split(np.arange(len(idx)), min(n_bins, len(idx)))
        for g in groups:
            if len(g) == 0:
                continue
            layer = np.tensordot(w[g], f[idx[g]], axes=1)
            sg = float(np.sqrt(np.sum(w[g] * sig[g] ** 2) / np.sum(w[g])))
            acc += ndimage.gaussian_filter(layer, sg, mode=modes) if sg > 0.05 else layer
        maps[e] = acc / w.sum()
        lines[e] = line
        res[e] = R                                                  # 90 % of the signal within this diameter
    f = 1 if pixel_m is None else max(int(round(pixel_m / dx_m)), 1)
    maps = {e: _bin(a, f) for e, a in maps.items()}
    if total_counts is not None:
        maps = _noise(maps, total_counts, background, rng)
    axes = {0: ("y", "x"), 1: ("z", "x"), 2: ("z", "y")}[axis]
    return EDSMap(maps, lines, res, dx_m * f, "SEM", beam_kV, axes,
                  dict(density=density, total_counts=total_counts, background=background, notes=notes))


def stem_eds(fields: dict, dx_m: float, index: int, axis: int = 0, foil_m: float = 100e-9, beam_kV: float = 200.0,
             probe_m: float = 1e-9, density: float = 8.44, into: int = -1, total_counts: Optional[float] = None,
             background: float = 0.1, pixel_m: Optional[float] = None, seed: int = 0,
             elements: Optional[Sequence[str]] = None) -> EDSMap:
    """STEM-EDS maps of a foil of thickness ``foil_m`` cut at ``index`` normal to ``axis`` and
    extending to the ``into`` side: the signal is the average through the foil, blurred by the
    probe and the beam broadening of the mean composition."""
    from scipy import ndimage

    elements = list(fields) if elements is None else list(elements)
    rng = np.random.default_rng(seed)
    nt = max(int(round(foil_m / dx_m)), 1)
    # beam broadening with the mass-weighted mean Z and A of the block
    tot = {e: float(np.mean(fields[e])) for e in fields if e in ATOMS}
    s = sum(tot.values())
    Z = sum(ATOMS[e][0] * v for e, v in tot.items()) / s
    A = sum(ATOMS[e][1] * v for e, v in tot.items()) / s
    b = beam_broadening_m(foil_m, beam_kV, Z, A, density)
    fwhm = np.hypot(probe_m, 0.5 * b)
    sig = fwhm / 2.355 / dx_m
    maps, lines, res = {}, {}, {}
    for e in elements:
        f, per2, per_n = _move(np.asarray(fields[e], dtype=float), axis)
        n = f.shape[0]
        idx = [index + into * d for d in range(nt)]
        idx = [i % n for i in idx] if per_n else [i for i in idx if 0 <= i < n]
        layer = f[idx].mean(axis=0)
        modes = ["wrap" if p else "nearest" for p in per2]
        maps[e] = ndimage.gaussian_filter(layer, sig, mode=modes) if sig > 0.05 else layer
        lines[e] = select_line(e, beam_kV)
        res[e] = 1.822 * fwhm                                     # d90 of a 2-D Gaussian
    f = 1 if pixel_m is None else max(int(round(pixel_m / dx_m)), 1)
    maps = {e: _bin(a, f) for e, a in maps.items()}
    if total_counts is not None:
        maps = _noise(maps, total_counts, background, rng)
    axes = {0: ("y", "x"), 1: ("z", "x"), 2: ("z", "y")}[axis]
    return EDSMap(maps, lines, res, dx_m * f, "STEM", beam_kV, axes,
                  dict(foil_m=foil_m, beam_broadening_m=b, total_counts=total_counts, background=background))


def line_profile(m: EDSMap, element: str, row: Optional[int] = None):
    """(distance [m], signal) along the middle row (or ``row``) of a map."""
    a = m.maps[element]
    r = a.shape[0] // 2 if row is None else row
    return np.arange(a.shape[1]) * m.pixel_m, a[r].copy()
