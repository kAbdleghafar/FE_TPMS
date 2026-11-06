"""Transient nonlinear heat conduction on the solid-mechanics meshes.

    sum_m C_m dH_m(T)/dt + K(T) T + q_s(T) = Q

* enthalpy formulation with a nodal (row-sum lumped) capacity, so latent heat
  is conserved exactly however large the time step: ``H_m(T)`` is the
  volumetric enthalpy of material ``m`` (``rho int cp dT`` + latent heat
  released linearly between solidus and liquidus);
* conductivity ``k(T)`` per element at the element-mean temperature;
* surface losses ``q_s = h (T - T_amb) + eps sigma (T^4 - T_amb^4)`` on the
  free faces of the *active* elements (they move with element birth),
  optionally restricted with a face filter (e.g. powder-covered faces);
* prescribed temperatures (e.g. the bottom of the build plate);
* element birth with a deposition temperature for the new nodes;
* backward Euler; each step is solved with secant-capacity Newton iterations
  (Morgan-type apparent capacity ``(H(T)-H(T_n))/(T-T_n)``, robust across the
  latent-heat kink) and AMG-preconditioned CG.

Units are the code's (mm, s, t, N): energy mJ, power mW, k in mW/(mm K)
(= W/(m K)), see :class:`tpmsfem.fem.materials.ThermalProperties`.
The heat problem is scalar and much cheaper than the mechanics, so it runs on
the host with SciPy/pyamg whatever the mechanical backend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np
import scipy.sparse as sp

from ..fem.elements import element_geometry, reference, voxel_geometry
from ..fem.materials import Material
from ..mesh.mesh import FACE_CORNERS, FACE_TYPE, FACES, Mesh

SIGMA_SB = 5.670374419e-11          # Stefan-Boltzmann [mW/(mm^2 K^4)]
KELVIN = 273.15


@dataclass
class HeatStepInfo:
    dt: float
    iterations: int
    krylov: int
    dT_max: float
    seconds: float
    residuals: list = field(default_factory=list)


class HeatModel:
    """Transient heat conduction with element birth (temperatures in degC by default;
    set ``kelvin_offset=0`` if the tables are in K)."""

    def __init__(self, mesh: Mesh, materials: Union[Material, Dict[int, Material]], T_initial: float = 20.0,
                 kelvin_offset: float = KELVIN):
        if isinstance(materials, Material):
            materials = {int(m): materials for m in np.unique(mesh.cell_data["material"])}
        self.mesh = mesh
        self.materials = dict(materials)
        for k, m in self.materials.items():
            if m.thermal is None:
                raise ValueError(f"material {k} ({m.name}) has no ThermalProperties")
        self.kelvin = float(kelvin_offset)
        self.ne, self.nn = mesh.cells.shape
        self.n_nodes = mesh.n_points
        keys = sorted(self.materials)
        self._keys = keys
        self.mat_index = np.searchsorted(np.array(keys), np.asarray(mesh.cell_data["material"]))

        # element matrices with k = 1 and lumped capacities (volume shares)
        _, w, N, _ = reference(mesh.cell_type)
        if mesh.is_voxel:
            dN, wq = voxel_geometry(mesh.meta["voxel_size"])
            self._Kref = np.einsum("q,qai,qbi->ab", wq, dN, dN)                 # (nn, nn) shared
            self._Ke = None
            self._cap = np.broadcast_to(np.einsum("q,qa->a", wq, N), (self.ne, self.nn))
        else:
            dN, wq = element_geometry(mesh.points, mesh.cells, mesh.cell_type)
            self._Kref = None
            self._Ke = np.einsum("eq,eqai,eqbi->eab", wq, dN, dN)
            self._cap = np.einsum("eq,qa->ea", wq, N)
        self._csr_pattern()

        self._H = [m.thermal.enthalpy_table() for m in (self.materials[k] for k in keys)]
        self.T = np.full(self.n_nodes, float(T_initial))
        self.time = 0.0
        self.active = np.ones(self.ne, bool)
        self.fixed = np.zeros(self.n_nodes, bool)
        self.T_fixed = np.zeros(self.n_nodes)
        self.q_vol = np.zeros(self.ne)                     # volumetric heat source [mW/mm^3]
        self.h_conv, self.emissivity, self.T_amb = 0.0, 0.0, float(T_initial)
        self.face_filter: Optional[Callable] = None
        self._version = 0
        self._bnd_cache = None
        self.history = []

    # ------------------------------------------------------------------ setup
    def _csr_pattern(self):
        """CSR structure of the full mesh and the map element entry -> CSR slot,
        so each assembly is a single bincount."""
        c = self.mesh.cells
        rows = np.repeat(c, self.nn, axis=1).ravel()
        cols = np.tile(c, (1, self.nn)).ravel()
        key = rows.astype(np.int64) * self.n_nodes + cols
        uk, inv = np.unique(key, return_inverse=True)
        self._slot = inv.ravel()
        r, cc = uk // self.n_nodes, uk % self.n_nodes
        self._indptr = np.r_[0, np.cumsum(np.bincount(r, minlength=self.n_nodes))]
        self._indices = cc.astype(np.int32)
        self._nnz = len(uk)

    def fix(self, nodes, T):
        nodes = np.atleast_1d(np.asarray(nodes, np.int64))
        self.fixed[nodes] = True
        self.T_fixed[nodes] = T
        self.T[nodes] = self.T_fixed[nodes]

    def release(self, nodes):
        self.fixed[np.atleast_1d(np.asarray(nodes, np.int64))] = False

    def set_surface_losses(self, h: float = 0.0, emissivity: float = 0.0, T_amb: Optional[float] = None,
                           face_filter: Optional[Callable] = None):
        """Convection ``h`` [mW/(mm^2 K)] (= W/(m^2 K) x 1e-3) and radiation on the
        free faces of active elements. ``face_filter(centres, normals) -> bool``
        selects the faces that lose heat (default: all free faces)."""
        self.h_conv, self.emissivity = float(h), float(emissivity)
        if T_amb is not None:
            self.T_amb = float(T_amb)
        self.face_filter = face_filter
        self._bnd_cache = None

    def deactivate(self, cells):
        self.active[np.atleast_1d(cells)] = False
        self._version += 1

    def activate(self, cells, T_new: Optional[float] = None):
        """Element birth. Nodes that did not belong to an active element take
        ``T_new`` (deposition temperature); shared nodes keep theirs."""
        cells = np.atleast_1d(np.asarray(cells))
        if T_new is not None:
            old = np.zeros(self.n_nodes, bool)
            old[np.unique(self.mesh.cells[self.active])] = True
            new = np.unique(self.mesh.cells[cells])
            new = new[~old[new] & ~self.fixed[new]]
            self.T[new] = T_new
        self.active[cells] = True
        self._version += 1

    # ------------------------------------------------------------------ pieces
    def _active_nodes(self):
        a = np.zeros(self.n_nodes, bool)
        a[np.unique(self.mesh.cells[self.active])] = True
        return a

    def _boundary(self):
        """Lumped nodal areas (n_nodes,) of the heat-losing free faces."""
        if self._bnd_cache is not None and self._bnd_cache[0] == self._version:
            return self._bnd_cache[1]
        area = np.zeros(self.n_nodes)
        if self.h_conv > 0 or self.emissivity > 0:
            act = np.flatnonzero(self.active)
            lf = FACES[self.mesh.cell_type]
            nc = FACE_CORNERS[FACE_TYPE[self.mesh.cell_type]]
            allf = self.mesh.cells[act][:, lf].reshape(-1, lf.shape[1])[:, :nc]
            owner = np.repeat(act, lf.shape[0])
            _, inv, cnt = np.unique(np.sort(allf, axis=1), axis=0, return_inverse=True, return_counts=True)
            free = cnt[inv.ravel()] == 1
            faces, owner = allf[free], owner[free]
            P = self.mesh.points[faces]
            if nc == 4:
                n = 0.5 * np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])
            else:
                n = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            A = np.linalg.norm(n, axis=1)
            ctr = P.mean(axis=1)
            out = self.mesh.points[self.mesh.cells[owner]].mean(axis=1)
            n *= np.sign(np.einsum("ij,ij->i", n, ctr - out))[:, None]        # outward
            keep = np.ones(len(faces), bool)
            if self.face_filter is not None:
                keep = np.asarray(self.face_filter(ctr, n / np.maximum(A, 1e-300)[:, None]), bool)
            np.add.at(area, faces[keep].ravel(), np.repeat(A[keep] / nc, nc))
        self._bnd_cache = (self._version, area)
        return area

    def _capacities(self):
        """Nodal lumped volume per material (list of (n_nodes,))."""
        out = []
        cap = self._cap
        for i in range(len(self._keys)):
            sel = self.active & (self.mat_index == i)
            v = np.zeros(self.n_nodes)
            if sel.any():
                np.add.at(v, self.mesh.cells[sel].ravel(), np.asarray(cap[sel]).ravel())
            out.append(v)
        return out

    def _conductivity_matrix(self, T):
        Te = T[self.mesh.cells].mean(axis=1)
        k = np.zeros(self.ne)
        for i, key in enumerate(self._keys):
            sel = self.mat_index == i
            k[sel] = self.materials[key].thermal.conductivity(Te[sel])
        k *= self.active
        if self._Ke is None:
            vals = (k[:, None, None] * self._Kref[None]).ravel()
        else:
            vals = (k[:, None, None] * self._Ke).ravel()
        data = np.bincount(self._slot, weights=vals, minlength=self._nnz)
        return sp.csr_matrix((data, self._indices, self._indptr), shape=(self.n_nodes,) * 2)

    def enthalpy(self, T=None) -> float:
        """Total enthalpy of the active material [mJ] (energy balance checks)."""
        T = self.T if T is None else T
        return float(sum((c * np.interp(T, *self._H[i])).sum() for i, c in enumerate(self._capacities())))

    # ------------------------------------------------------------------ solve
    def _nodal_enthalpy(self, caps):
        """Per-node enthalpy functions h(T), dh/dT and the inverse T(h) for the
        capacity-weighted mix of the materials meeting at each node."""
        Ctot = sum(caps)
        w = [c / np.where(Ctot > 0, Ctot, 1.0) for c in caps]
        slopes = [(H[0][1:] - 0.5 * (H[0][1] - H[0][0]), np.diff(H[1]) / np.diff(H[0])) for H in self._H]
        pure = [np.isclose(wi, 1.0) for wi in w]
        mixed = ~np.any(pure, axis=0)

        def h(T):
            return sum(wi * np.interp(T, *H) for wi, H in zip(w, self._H))

        def dh(T):
            return sum(wi * np.interp(T, *sl) for wi, sl in zip(w, slopes))

        def inv(hv, idx):
            T = np.empty(len(idx))
            for i, H in enumerate(self._H):
                sel = pure[i][idx]
                T[sel] = np.interp(hv[sel], H[1], H[0])
            mx = mixed[idx]
            if mx.any():                           # monotone: bisection
                lo = np.full(mx.sum(), self._H[0][0][0])
                hi = np.full(mx.sum(), self._H[0][0][-1])
                ww = [wi[idx][mx] for wi in w]
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    up = sum(wi * np.interp(mid, *H) for wi, H in zip(ww, self._H)) > hv[mx]
                    hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
                T[mx] = 0.5 * (lo + hi)
            return T

        return Ctot, h, dh, inv

    def step(self, dt: float, tol: float = 1e-4, max_iter: int = 60, rtol: float = 1e-10) -> HeatStepInfo:
        """Advance by ``dt`` (backward Euler). Newton iterations whose update is
        applied to the nodal *enthalpy* and mapped back through the monotone
        T(h) (Nedjar 2002): robust across the solidus/liquidus kinks for any
        step size. ``tol``: max temperature change of the last iteration [K]."""
        from scipy.sparse.linalg import cg

        t0 = time.perf_counter()
        act = self._active_nodes()
        free = act & ~self.fixed
        idx = np.flatnonzero(free)
        Tn = self.T.copy()
        T = Tn.copy()
        T[self.fixed] = self.T_fixed[self.fixed]
        caps = self._capacities()
        Ctot, hfun, dhfun, hinv = self._nodal_enthalpy(caps)
        hN = hfun(Tn)
        A = self._boundary()
        Q = np.zeros(self.n_nodes)
        if np.any(self.q_vol):
            np.add.at(Q, self.mesh.cells.ravel(), (np.asarray(self._cap) * (self.q_vol * self.active)[:, None]).ravel())
        Ta = self.T_amb + self.kelvin
        info = HeatStepInfo(dt, 0, 0, 0.0, 0.0)
        key = (self._version, float(dt))
        M = self._M[1] if getattr(self, "_M", (None,))[0] == key else None
        step_max = np.inf
        for it in range(max_iter):
            K = self._conductivity_matrix(T)
            Tk = T + self.kelvin
            qs = A * (self.h_conv * (T - self.T_amb) + self.emissivity * SIGMA_SB * (Tk**4 - Ta**4))
            dqs = A * (self.h_conv + 4.0 * self.emissivity * SIGMA_SB * Tk**3)
            h = hfun(T)
            c = dhfun(T)
            R = Ctot * (h - hN) / dt + K @ T + qs - Q
            J = (K[idx][:, idx] + sp.diags(Ctot[idx] * c[idx] / dt + dqs[idx])).tocsr()
            if M is None:
                M = _preconditioner(J)
                self._M = (key, M)
            its = [0]

            def cb(_):
                its[0] += 1

            d, _ = cg(J, -R[idx], rtol=rtol, atol=0.0, M=M, maxiter=2000, callback=cb)
            info.krylov += its[0]
            Tnew = hinv(h[idx] + c[idx] * d, idx)          # update in enthalpy space
            step_max = float(np.abs(Tnew - T[idx]).max()) if len(idx) else 0.0
            T[idx] = Tnew
            info.iterations = it + 1
            info.residuals.append(step_max)
            if step_max < tol:
                break
        else:
            raise RuntimeError(f"heat step did not converge: last |dT| = {step_max:.3e} K")
        self.T = T
        self.time += dt
        info.dT_max = float(np.abs(T - Tn)[act].max()) if act.any() else 0.0
        info.seconds = time.perf_counter() - t0
        self.history.append(info)
        return info


def _preconditioner(J):
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(J, symmetry="symmetric", max_coarse=500)
        return ml.aspreconditioner(cycle="V")
    except ImportError:  # pragma: no cover
        d = J.diagonal()
        return sp.diags(1.0 / d)


def geometric_steps(total: float, first: float, n: int) -> np.ndarray:
    """``n`` time steps growing geometrically from ``first`` and summing to ``total``."""
    if n == 1:
        return np.array([total])
    if first * n >= total:
        return np.full(n, total / n)
    from scipy.optimize import brentq

    r = brentq(lambda q: first * (q**n - 1) / (q - 1) - total, 1.0 + 1e-9, 1e3)
    return first * r ** np.arange(n)
