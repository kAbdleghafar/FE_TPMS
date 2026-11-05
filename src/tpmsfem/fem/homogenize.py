"""Numerical homogenisation of periodic voxel cells.

The displacement is split into a macroscopic part and a periodic fluctuation,
``u = E x + u~``. With ``sigma = C : (E + eps(u~))`` the fluctuation problem is
exactly an eigenstrain problem with ``eps* = -E`` applied to every element.
Periodicity is enforced by node identification in the voxel mesh
(``voxelize(..., periodic=...)``), so no constraint equations are needed and
the matrix-free GPU kernel is used unchanged.

Effective stresses are volume averages over the *whole* cell (solid + pores):
``<sigma> = (1/V_cell) sum_e sum_q w sigma``.

Non-periodic axes have traction-free faces (e.g. a plate that is periodic in
x, y and free in z, as in graded "sandwich" cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..backend import asnumpy
from .model import SolidModel

# Voigt order used here: xx, yy, zz, yz, xz, xy with ENGINEERING shear strains
VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass
class HomogenizationResult:
    components: tuple          # Voigt indices that were loaded
    C: np.ndarray              # effective stiffness on those components [MPa]
    volume_fraction: float
    infos: list

    @property
    def S(self) -> np.ndarray:
        return np.linalg.inv(self.C)

    def young(self, axis: str) -> float:
        """Uniaxial modulus along 'xx'/'yy'/'zz' (other loaded components stress-free)."""
        k = self.components.index(VOIGT.index(axis))
        return 1.0 / self.S[k, k]

    def shear(self, plane: str) -> float:
        k = self.components.index(VOIGT.index(plane))
        return 1.0 / self.S[k, k]

    def poisson(self, load: str, lateral: str) -> float:
        i = self.components.index(VOIGT.index(load))
        j = self.components.index(VOIGT.index(lateral))
        return -self.S[j, i] / self.S[i, i]

    def zener(self) -> float:
        """Zener anisotropy 2 C44 / (C11 - C12) for a full cubic C."""
        C = self.C
        return 2 * C[3, 3] / (C[0, 0] - C[0, 1])


def cell_volume(mesh) -> float:
    shape = np.asarray(mesh.meta["grid_shape"])
    return float(np.prod(shape) * mesh.meta["voxel_size"] ** 3)


def average_stress(model: SolidModel, V_cell: float) -> np.ndarray:
    """Volume-averaged stress (6,) [xx,yy,zz,yz,xz,xy] over the cell volume."""
    xp = model.xp
    w = model.wq[None, :] if model.shared else model.wq
    s = (model.sig * w[..., None] * model.active[:, None, None]).sum(axis=(0, 1))
    return asnumpy(s) / V_cell


def homogenize(model: SolidModel, components: Sequence[int] = (0, 1, 2, 3, 4, 5),
               solver: str = "auto", tol: float = 1e-8) -> HomogenizationResult:
    """Effective stiffness of a periodic voxel cell for unit macro strains.

    ``components``: Voigt indices to load (0..5 = xx,yy,zz,yz,xz,xy). Use
    ``(0, 1, 5)`` for a cell periodic in x, y and free in z (in-plane stiffness).
    The model's constraints are replaced by a single pinned node (rigid
    translation). Periodicity along at least two axes excludes every rigid
    rotation (each would need a displacement growing linearly along a
    periodic axis), so no further constraint is needed -- adding one would
    restrain the fluctuation field.
    """
    mesh = model.mesh
    if not mesh.meta.get("periodic") or not any(mesh.meta["periodic"]):
        raise ValueError("homogenize() needs a periodic voxel mesh (voxelize(..., periodic=...))")
    per = mesh.meta["periodic"]
    for c in components:
        if c < 3 and not per[c]:
            raise ValueError(f"normal strain {VOIGT[c]} needs a periodic {'xyz'[c]} axis")
    V = cell_volume(mesh)
    model.fixed_host[:] = False
    node = int(mesh.cells[0, 0])
    model.fix([node], (0, 1, 2), 0.0)
    if sum(per) < 2:
        raise ValueError("need at least two periodic axes to exclude rigid rotations")
    rows, infos = [], []
    for c in components:
        model.reset_state()
        eps = np.zeros(6)
        eps[c] = 1.0 if c < 3 else 0.5          # engineering shear 1 -> tensor 1/2
        model.set_eigenstrain(np.arange(model.ne), -eps * 1e-3, scale=1.0)
        infos.append(model.solve(solver=solver, tol=tol, name=f"E_{VOIGT[c]}"))
        rows.append(average_stress(model, V)[list(components)] / 1e-3)
    C = np.array(rows).T                           # column c = response to unit strain c
    C = 0.5 * (C + C.T)
    vf = model.ne * mesh.meta["voxel_size"] ** 3 / V
    return HomogenizationResult(tuple(components), C, vf, infos)
