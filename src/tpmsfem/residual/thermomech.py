"""Sequentially coupled thermo-mechanical simulation of a layer-wise AM build.

Layer-lumped ("flash") process model, the part-scale approach of commercial
AM simulators and of LLNL's Diablo AM-Bench models:

1. the part is split into super-layers of thickness ``layer_height``
   (each lumping ``layer_height / physical layer`` real layers);
2. a super-layer is born at the deposition temperature ``T_deposit``
   (typically the liquidus) on the current, deformed part; its new nodes take
   that temperature and its elements are mechanically born stress-free;
3. the heat model (:class:`tpmsfem.thermal.HeatModel`) cools the part for the
   time the super-layer takes to print (``layer_time``) in ``n_steps``
   geometrically growing steps, with conduction into the part and the build
   plate, latent heat, surface losses and the plate bottom held at
   ``T_plate``;
4. after each heat step the mechanical model
   (:class:`tpmsfem.fem.SolidModel` with temperature-dependent elasto-plastic
   materials) is re-equilibrated under the new temperature field: thermal
   strains, E(T), sigma_y(T, eps_p), annealing;
5. finally the part cools to room temperature and can be cut from the plate.

Both models must share the same mesh (one temperature per node).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..fem.model import SolidModel
from ..thermal.heat import HeatModel, geometric_steps
from .am import layer_partition, three_two_one


@dataclass
class TMLayerRecord:
    layer: int
    z_top: float
    T_max_after_first_step: float
    T_max_end: float
    newton: list
    krylov: int
    heat_iterations: int
    max_vm: float
    seconds: float


@dataclass
class ThermoMechanicalAM:
    model: SolidModel
    heat: HeatModel
    layer_height: float
    T_deposit: float
    layer_time: float                     # time to print one super-layer [s]
    n_steps: int = 8                      # heat steps per super-layer
    first_step: float = 0.05              # first heat step after deposition [s]
    mech_every: int = 1                   # mechanical equilibrium every k-th heat step (and the last)
    T_plate: float = 20.0                 # prescribed at the bottom of the plate during the build
    substrate: Optional[np.ndarray] = None
    z0: Optional[float] = None
    plate_tol: Optional[float] = None
    solver: str = "auto"
    n_inc: int = 1
    records: list = field(default_factory=list)

    def __post_init__(self):
        if self.heat.mesh is not self.model.mesh and self.heat.n_nodes != self.model.n_nodes:
            raise ValueError("heat and mechanical models must share the mesh")
        if not self.model.thermal:
            raise ValueError("the mechanical model needs temperature-dependent materials")
        mesh = self.model.mesh
        self.layer_of_cell, self.z0, self.plate_nodes, self.is_substrate, self.plate_tol = layer_partition(
            mesh, self.layer_height, self.substrate, self.z0, self.plate_tol)
        self.n_layers = int(self.layer_of_cell.max()) + 1
        self.steps = geometric_steps(self.layer_time, self.first_step, self.n_steps)

    def _mech(self, name):
        self.model.set_temperature(self.heat.T)
        return self.model.solve(n_inc=self.n_inc, solver=self.solver, name=name)

    def build(self, layers: Optional[int] = None, verbose: bool = True, callback: Optional[Callable] = None):
        m, hm = self.model, self.heat
        part = np.flatnonzero(~self.is_substrate)
        m.deactivate(part)
        hm.deactivate(part)
        if self.is_substrate.any():
            m.activate(np.flatnonzero(self.is_substrate), strain_free=False)
        hm.T[:] = self.T_plate
        hm.fix(self.plate_nodes, self.T_plate)
        m.fix(self.plate_nodes)
        m.set_initial_temperature(hm.T)               # stress-free plate at the build temperature
        n = self.n_layers if layers is None else min(layers, self.n_layers)
        for k in range(n):
            t0 = time.perf_counter()
            cells = np.flatnonzero(self.layer_of_cell == k)
            if len(cells) == 0:
                continue
            hm.activate(cells, T_new=self.T_deposit)
            m.set_temperature(hm.T, ramp=False)        # changes only not-yet-active elements
            m.activate(cells, strain_free=True)        # born stress-free at the deposition temperature
            newton, krylov, hit, Tfirst = [], 0, 0, None
            for i, dt in enumerate(self.steps):
                hi = hm.step(dt)
                hit += hi.iterations
                if Tfirst is None:
                    Tfirst = float(hm.T.max())
                if (i + 1) % self.mech_every == 0 or i == len(self.steps) - 1:
                    info = self._mech(f"layer{k}.{i}")
                    newton += info.newton_iterations
                    krylov += info.krylov_iterations
            rec = TMLayerRecord(k, self.z0 + (k + 1) * self.layer_height, Tfirst, float(hm.T.max()),
                                newton, krylov, hit, self._max_vm(), time.perf_counter() - t0)
            self.records.append(rec)
            if verbose:
                print(f"layer {k + 1:3d}/{n}: z<={rec.z_top:6.2f}  T_max {rec.T_max_after_first_step:6.0f} -> "
                      f"{rec.T_max_end:5.0f} C  newton={rec.newton} krylov={rec.krylov} heat_its={rec.heat_iterations} "
                      f"vm_max={rec.max_vm:6.0f}  {rec.seconds:.1f}s", flush=True)
            if callback is not None:
                callback(self, k)
        return self.records

    def cool_down(self, T_room: float = 20.0, time_total: float = 3600.0, n_steps: int = 6,
                  verbose: bool = True):
        """Cool to ``T_room`` (plate bottom lowered to it), then impose the uniform
        room temperature exactly and re-equilibrate."""
        m, hm = self.model, self.heat
        t0 = time.perf_counter()
        hm.fix(self.plate_nodes, T_room)
        for i, dt in enumerate(geometric_steps(time_total, min(10.0, time_total / n_steps), n_steps)):
            hm.step(dt)
            self._mech(f"cool{i}")
        hm.T[:] = T_room
        m.set_temperature(T_room)
        info = m.solve(n_inc=self.n_inc, solver=self.solver, name="room temperature")
        if verbose:
            print(f"cool-down to {T_room} C: vm_max={self._max_vm():.0f}  newton={info.newton_iterations} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def remove_from_plate(self, verbose: bool = True):
        """Release the plate bottom and hold the part on a 3-2-1 support."""
        m = self.model
        m.release(self.plate_nodes)
        for node, comps in three_two_one(m.mesh.points, self.plate_nodes):
            m.fix([node], comps, 0.0)
        info = m.solve(n_inc=self.n_inc, solver=self.solver, name="removal")
        if verbose:
            print(f"removal: vm_max={self._max_vm():.0f} newton={info.newton_iterations}", flush=True)
        return info

    def _max_vm(self):
        from ..fem.constitutive import von_mises

        vm = np.asarray(von_mises(self.model.stress().mean(axis=1)))[self.model.active_host]
        return float(vm.max()) if len(vm) else 0.0
