"""Additive-manufacturing (LPBF) residual stress via the inherent-strain method.

Process model
-------------
1. The part is split into *super-layers* of thickness ``layer_height`` along
   the build direction (+z). Elements are inactive until their layer is built.
2. For each layer: activate its elements *strain-free* on the current deformed
   configuration, apply the inherent strain ``eps*`` to the layer, and solve for
   equilibrium (elastic or elasto-plastic). The bottom of the part is welded to
   the (rigid) build plate.
3. Optionally remove the part from the plate: replace the weld by a statically
   determinate 3-2-1 support and re-equilibrate -> spring-back distortion and
   the final residual stress state.

``mode="element"`` applies ``eps*`` uniformly per element (the classic method).
``mode="nodal"`` applies it through a nodal activation field interpolated to
the integration points, which is how a commercial code applies it with a
"temperature" field and orthotropic expansion coefficients -- use it for
code-to-code benchmarks of the layer-by-layer process (see
:func:`tpmsfem.io.abaqus.write_inp` with ``am=``).

The inherent strain tensor must be calibrated for the machine/material/scan
strategy (e.g. from cantilever or bridge distortion experiments, see
:func:`calibrate_scale`). The default is illustrative only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..fem.model import SolidModel

DEFAULT_INHERENT_STRAIN = (-0.004, -0.004, -0.002, 0.0, 0.0, 0.0)   # illustrative


@dataclass
class LayerRecord:
    layer: int
    z_top: float
    n_active: int
    max_disp: float
    max_vm: float
    newton: list
    krylov: int
    seconds: float


@dataclass
class AMProcess:
    model: SolidModel
    layer_height: float
    inherent_strain: tuple = DEFAULT_INHERENT_STRAIN
    mode: str = "element"                 # "element" | "nodal"
    z0: Optional[float] = None            # build-plate plane (default: min z of mesh)
    plate_tol: Optional[float] = None
    n_inc: int = 1                        # increments per layer (use >1 with plasticity)
    solver: str = "auto"
    birth_reference: str = "qp"           # "qp" | "element_mean" (CalculiX semantics, nodal mode)
    substrate: Optional[np.ndarray] = None  # cells of an elastic build plate (present from the start)
    records: list = field(default_factory=list)

    def __post_init__(self):
        mesh = self.model.mesh
        self.layer_of_cell, self.z0, self.plate_nodes, self.is_substrate, self.plate_tol = layer_partition(
            mesh, self.layer_height, self.substrate, self.z0, self.plate_tol)
        self.n_layers = int(self.layer_of_cell.max()) + 1
        self.removed = False

    # ------------------------------------------------------------------
    def _layer_nodes(self, k):
        cells = np.flatnonzero(self.layer_of_cell == k)
        return cells, np.unique(self.model.mesh.cells[cells])

    def build(self, layers: Optional[int] = None, verbose: bool = True,
              callback: Optional[Callable] = None):
        """Simulate the layer-by-layer build (all layers by default)."""
        m = self.model
        eps = np.asarray(self.inherent_strain, float)
        part = np.flatnonzero(~self.is_substrate)
        m.deactivate(part)
        if self.is_substrate.any():
            m.activate(np.flatnonzero(self.is_substrate), strain_free=False)
        m.fix(self.plate_nodes)                        # welded to / bolted on the build plate
        m.set_eigenstrain(part, eps, scale=0.0)
        theta = np.zeros(m.n_nodes)
        n = self.n_layers if layers is None else min(layers, self.n_layers)
        for k in range(n):
            t0 = time.perf_counter()
            cells, nodes = self._layer_nodes(k)
            if len(cells) == 0:
                continue
            m.activate(cells, strain_free=True, eig_birth=self.birth_reference)
            if self.mode == "element":
                m.set_eigenstrain_scale(cells, 1.0)
            elif self.mode == "nodal":
                theta[nodes] = 1.0
                m.set_eigenstrain_nodal(theta)
            else:
                raise ValueError(self.mode)
            info = m.solve(n_inc=self.n_inc, solver=self.solver, name=f"layer{k}")
            rec = self._record(k, info, time.perf_counter() - t0)
            self.records.append(rec)
            if verbose:
                print(f"layer {k + 1:4d}/{n}: z<={rec.z_top:7.3f}  active={rec.n_active:8d}  "
                      f"|u|max={rec.max_disp:.4e}  vm_max={rec.max_vm:8.1f}  "
                      f"newton={rec.newton} krylov={rec.krylov}  {rec.seconds:.2f}s")
            if callback is not None:
                callback(self, k)
        return self.records

    def remove_from_plate(self, verbose: bool = True):
        """Cut the part off the plate: release the weld and hold it with a
        statically determinate 3-2-1 support (adds no stress)."""
        m = self.model
        m.release(self.plate_nodes)
        for node, comps in self.three_two_one():
            m.fix([node], comps, 0.0)   # support nodes were welded => u = 0 already
        t0 = time.perf_counter()
        info = m.solve(solver=self.solver, name="removal")
        self.removed = True
        rec = self._record(-1, info, time.perf_counter() - t0)
        self.records.append(rec)
        if verbose:
            print(f"removal: |u|max={rec.max_disp:.4e} vm_max={rec.max_vm:.1f} newton={rec.newton} "
                  f"krylov={rec.krylov} {rec.seconds:.2f}s")
        return rec

    def three_two_one(self):
        """[(node, comps), ...] of a 3-2-1 support on the plate nodes."""
        return three_two_one(self.model.mesh.points, self.plate_nodes)

    def _record(self, k, info, seconds):
        m = self.model
        from ..fem.constitutive import von_mises

        u = m.displacement()
        act_nodes = np.unique(m.mesh.cells[m.active_host])
        vm = np.asarray(von_mises(m.stress().mean(axis=1)))[m.active_host]
        z_top = self.z0 + (k + 1) * self.layer_height if k >= 0 else float("nan")
        return LayerRecord(k, z_top, int(m.active_host.sum()),
                           float(np.linalg.norm(u[act_nodes], axis=1).max()),
                           float(vm.max()) if len(vm) else 0.0,
                           list(info.newton_iterations), info.krylov_iterations, seconds)


def three_two_one(points, nodes):
    """[(node, comps), ...] of a statically determinate 3-2-1 support on ``nodes``.

    A (x,y,z) and B (2 comps) span the longest in-plane chord, C (z) is the
    node farthest from line AB, so all six rigid-body modes are removed
    without over-constraining the part."""
    p = points[nodes]
    c = p.mean(axis=0)
    ia = int(np.argmax(np.linalg.norm(p - c, axis=1)))
    ib = int(np.argmax(np.linalg.norm(p - p[ia], axis=1)))
    ab = (p[ib] - p[ia]) / np.linalg.norm(p[ib] - p[ia])
    rel = p - p[ia]
    ic = int(np.argmax(np.linalg.norm(rel - np.outer(rel @ ab, ab), axis=1)))
    A, B, Cn = (int(nodes[i]) for i in (ia, ib, ic))
    comps_b = (1, 2) if abs(ab[0]) >= abs(ab[1]) else (0, 2)
    return [(A, (0, 1, 2)), (B, comps_b), (Cn, (2,))]


def layer_partition(mesh, layer_height, substrate=None, z0=None, plate_tol=None):
    """Super-layer index of every cell (-1 for the substrate), the build-plate
    plane z0, the support nodes (bottom of the substrate, or the plate plane
    for a rigid plate) and the substrate mask."""
    z = mesh.points[:, 2]
    if plate_tol is None:
        plate_tol = 1e-6 * float(np.ptp(mesh.points))
    zc = mesh.centroids()[:, 2]
    is_sub = np.zeros(mesh.n_cells, bool)
    if substrate is not None:
        is_sub[np.asarray(substrate)] = True
    if z0 is None:
        z0 = float(mesh.points[np.unique(mesh.cells[is_sub])][:, 2].max()) if is_sub.any() else float(z.min())
    layer = np.floor((zc - z0) / layer_height + 1e-9).astype(np.int64)
    layer[is_sub] = -1
    if is_sub.any():
        zb = float(mesh.points[np.unique(mesh.cells[is_sub])][:, 2].min())
        plate = mesh.nodes_where(lambda x, y, zz: zz <= zb + plate_tol)
    else:
        plate = mesh.nodes_where(lambda x, y, zz: zz <= z0 + plate_tol)
    if len(plate) == 0:
        raise ValueError("no nodes on the build plate; check z0")
    return layer, z0, plate, is_sub, plate_tol


def calibrate_scale(run: Callable[[float], float], measured: float) -> float:
    """Calibrate an isotropic multiplier ``s`` of the inherent strain so that a
    simulated scalar response (e.g. cantilever tip deflection after cutting)
    matches ``measured``. Exact for the elastic method (response is linear in
    ``s``): one run with ``s=1`` suffices."""
    r1 = run(1.0)
    if r1 == 0:
        raise ValueError("response is insensitive to the inherent strain")
    return measured / r1
