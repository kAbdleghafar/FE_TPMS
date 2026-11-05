"""Parametric hip-implant geometries with TPMS regions.

These are deliberately simple, fully parametric stand-ins for real device CAD:
they exercise the same mechanics (thin TPMS walls, solid/porous transitions,
stress concentrations at the neck) and can be swapped for a patient- or
vendor-specific surface (STL -> tet mesh, see ``tpmsfem.mesh.adapters``).

Coordinate conventions (mm):

* Femoral stem: stem axis = +z, distal tip at z = 0, neck/head towards -x
  (medial), anterior = +y.
* Acetabular cup: centre of the hemisphere at the origin, dome towards +z,
  rim on the plane z = 0 (the natural LPBF build orientation: rim on plate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fields as F
from .tpms import TPMS


@dataclass
class FemoralStem:
    stem_length: float = 130.0     # distal tip -> shoulder
    r_distal: float = 5.0
    r_proximal: float = 10.0
    neck_angle_deg: float = 135.0  # CCD angle between neck and stem axes
    neck_length: float = 35.0      # shoulder axis point -> head centre
    neck_radius: float = 6.0
    neck_origin_z: float = 115.0   # where the neck axis meets the stem axis
    head_radius: float = 14.0      # 0 => no head (trunnion end only)
    # lattice design
    design: str = "lattice_core"   # "lattice_core" | "porous_coating" | "solid"
    lattice: TPMS = field(default_factory=lambda: TPMS("gyroid", cell_size=4.0, kind="sheet", density=0.35))
    skin: float = 1.5              # solid skin (lattice_core) or coating thickness
    lattice_z: tuple = (40.0, 105.0)

    @property
    def head_center(self) -> np.ndarray:
        a = np.deg2rad(180.0 - self.neck_angle_deg)  # angle of neck from +z
        d = np.array([-np.sin(a), 0.0, np.cos(a)])
        return np.array([0.0, 0.0, self.neck_origin_z]) + self.neck_length * d

    def body(self) -> F.Field:
        stem = F.round_cone((0, 0, self.r_distal), (0, 0, self.stem_length),
                            self.r_distal, self.r_proximal)
        neck = F.capsule((0, 0, self.neck_origin_z), tuple(self.head_center), self.neck_radius)
        g = stem | neck
        if self.head_radius > 0:
            g = g | F.sphere(tuple(self.head_center), self.head_radius)
        return g

    def field(self) -> F.Field:
        stem = F.round_cone((0, 0, self.r_distal), (0, 0, self.stem_length),
                            self.r_distal, self.r_proximal)
        body = self.body()
        if self.design == "solid":
            return body
        zone = F.zone_z(*self.lattice_z)
        if self.design == "lattice_core":
            core = stem.offset(-self.skin) & zone
            return (body - core) | (core & self.lattice)
        if self.design == "porous_coating":
            coat = (stem - stem.offset(-self.skin)) & zone
            return (body - coat) | (coat & self.lattice)
        raise ValueError(self.design)

    def bounds(self, pad: float = 1.0):
        hc = self.head_center
        rh = max(self.head_radius, self.neck_radius)
        lo = np.array([min(-self.r_proximal, hc[0] - rh), -max(self.r_proximal, rh), 0.0]) - pad
        hi = np.array([self.r_proximal, max(self.r_proximal, rh), max(self.stem_length, hc[2] + rh)]) + pad
        return lo, hi


@dataclass
class AcetabularCup:
    outer_radius: float = 27.0     # 54 mm cup
    inner_radius: float = 23.0     # liner seat
    porous_thickness: float = 1.5  # outer TPMS layer for bone ingrowth
    rim_height: float = 2.0        # dense band above the rim
    lattice: TPMS = field(default_factory=lambda: TPMS("gyroid", cell_size=1.5, kind="sheet", density=0.3))

    def field(self) -> F.Field:
        Ro, Ri, tp = self.outer_radius, self.inner_radius, self.porous_thickness
        upper = F.half_space((0, 0, 0), (0, 0, -1))  # z >= 0
        dense = (F.sphere(radius=Ro - tp) - F.sphere(radius=Ri)) & upper
        layer = (F.sphere(radius=Ro) - F.sphere(radius=Ro - tp)) & upper
        rim = layer & F.zone_z(-1.0, self.rim_height)
        porous = layer - F.zone_z(-1.0, self.rim_height)
        return dense | rim | (porous & self.lattice)

    def bounds(self, pad: float = 0.5):
        R = self.outer_radius
        return np.array([-R - pad, -R - pad, 0.0]), np.array([R + pad, R + pad, R + pad])


def lattice_cube(lattice: TPMS, size: float, skin_plates: bool = True, plate: float = 0.5):
    """A cube of lattice, optionally with thin solid plates top/bottom.

    The plates mimic the platens of a compression test and make the
    effective-stiffness boundary conditions unambiguous.
    """
    cube = F.box((0, 0, 0), (size, size, size))
    g = cube & lattice
    if skin_plates:
        g = g | F.box((0, 0, 0), (size, size, plate)) | F.box((0, 0, size - plate), (size, size, size))
    return g
