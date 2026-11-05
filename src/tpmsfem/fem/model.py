"""Matrix-free, GPU-ready small-strain solid mechanics model.

Design
------
* Element data (shape-function gradients, weights, dof maps) and all state live
  on the active backend (NumPy or CuPy).
* The stiffness matrix is never assembled for the default ``pcg`` solver: the
  operator ``K v`` is evaluated element-by-element as gather -> batched GEMM ->
  scatter-add. For voxel meshes every element shares one reference gradient
  matrix, so the kernel is a single dense GEMM per chunk.
* Total stress: ``sigma = C : (eps(u) - eps_ref - eps_eig - eps_p)`` where
  ``eps_ref`` makes elements born *strain-free* (layer-by-layer activation),
  ``eps_eig`` is an eigenstrain (thermal / inherent strain) and ``eps_p`` the
  J2 plastic strain.
* Loads, prescribed displacements and eigenstrains are *targets* that a call to
  :meth:`SolidModel.solve` ramps to in ``n_inc`` increments (Abaqus-like step
  semantics); Newton-Raphson with the consistent tangent handles plasticity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np

from ..backend import asnumpy, backend_name, get_xp, on_gpu, scatter_add, sparse_module
from ..mesh.mesh import Mesh
from . import constitutive as C
from .elements import element_geometry, face_integrals, reference, voxel_geometry
from .materials import Material
from .solvers import AssembledSolver, KrylovInfo, pcg, rigid_body_modes


@dataclass
class StepInfo:
    converged: bool
    increments: int
    newton_iterations: list = field(default_factory=list)
    krylov_iterations: int = 0
    residuals: list = field(default_factory=list)
    seconds: float = 0.0
    solver: str = ""

    def __str__(self):
        return (f"converged={self.converged} inc={self.increments} newton={self.newton_iterations} "
                f"krylov_its={self.krylov_iterations} solver={self.solver} time={self.seconds:.2f}s")


class SolidModel:
    def __init__(self, mesh: Mesh, materials: Union[Material, Dict[int, Material]], *,
                 bbar: Optional[bool] = None, chunk_size: Optional[int] = None,
                 dtype=np.float64):
        xp = get_xp()
        self.xp = xp
        self.mesh = mesh
        self.dtype = dtype
        if isinstance(materials, Material):
            materials = {int(m): materials for m in np.unique(mesh.cell_data["material"])}
        self.materials = dict(materials)
        mat_ids = np.asarray(mesh.cell_data["material"])
        missing = set(np.unique(mat_ids).tolist()) - set(self.materials)
        if missing:
            raise KeyError(f"no Material for material ids {sorted(missing)}")

        ct = mesh.cell_type
        self.ne, self.nn = mesh.cells.shape
        self.n_nodes = mesh.n_points
        self.ndof = 3 * self.n_nodes
        self.plastic = any(m.is_plastic for m in self.materials.values())
        self.bbar = (ct == "hex8" and self.plastic) if bbar is None else bool(bbar)
        if self.bbar and ct != "hex8":
            raise ValueError("B-bar is only implemented for hex8")

        xi, w, N, _ = reference(ct)
        self.nq = len(w)
        self.Nq = N                                            # (nq, nn) host
        # chunk temporaries: ~2M qp-values on CPU (cache), ~32M on GPU (fewer kernel launches)
        budget = 32_000_000 if on_gpu() else 2_000_000
        self.chunk = chunk_size or max(1024, int(budget // (self.nq * self.nn)))

        # --- geometry ---------------------------------------------------------
        self.shared = mesh.is_voxel
        if self.shared:
            dN0, w0 = voxel_geometry(mesh.meta["voxel_size"])
            self.dN = xp.asarray(dN0, dtype=dtype)             # (nq, nn, 3)
            self.wq = xp.asarray(w0, dtype=dtype)              # (nq,)
            nd = 3 * self.nn
            Mg = np.zeros((self.nn, 3, self.nq, 3, 3))
            for i in range(3):
                Mg[:, i, :, i, :] = dN0.transpose(1, 0, 2)
            self.Mg = xp.asarray(Mg.reshape(nd, self.nq * 9), dtype=dtype)
            self.MgT = xp.ascontiguousarray(self.Mg.T)
        else:
            dNdx, wdet = element_geometry(mesh.points, mesh.cells, ct)
            self.dN = xp.asarray(dNdx, dtype=dtype)            # (ne, nq, nn, 3)
            self.wq = xp.asarray(wdet, dtype=dtype)            # (ne, nq)
        dofs = (mesh.cells[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(self.ne, -1)
        self.dofs = xp.asarray(dofs)

        # --- materials --------------------------------------------------------
        lut = {k: m for k, m in self.materials.items()}
        self.mat_ids = mat_ids
        self.lam = xp.asarray(np.array([lut[i].lam for i in mat_ids]), dtype=dtype)
        self.mu = xp.asarray(np.array([lut[i].mu for i in mat_ids]), dtype=dtype)
        self.sy0 = xp.asarray(np.array([lut[i].sy0 for i in mat_ids]), dtype=dtype)
        self.H = xp.asarray(np.array([lut[i].hardening for i in mat_ids]), dtype=dtype)

        # --- state ------------------------------------------------------------
        z = lambda *s: xp.zeros(s, dtype=dtype)
        self.u = z(self.ndof)
        self.sig = z(self.ne, self.nq, 6)
        self.eps_ref = None
        self.eig_dir = z(self.ne, 6)
        self.eig_target = z(self.ne, self.nq)
        self.eig_applied = z(self.ne, self.nq)
        self.eig_cur = self.eig_applied
        self._has_eig = False
        if self.plastic:
            self.eps_p, self.alpha = z(self.ne, self.nq, 6), z(self.ne, self.nq)
            self.eps_p_t, self.alpha_t = z(self.ne, self.nq, 6), z(self.ne, self.nq)
            self.theta = xp.ones((self.ne, self.nq), dtype=dtype)
            self.theta_bar = z(self.ne, self.nq)
            self.nrm = z(self.ne, self.nq, 6)
        self.active_host = np.ones(self.ne, bool)
        self.active = xp.ones(self.ne, dtype=dtype)

        # --- loads / BCs (host masks, device values) -------------------------
        self.fixed_host = np.zeros(self.ndof, bool)
        self.u_bc_target = z(self.ndof)
        self.f_ext_target = z(self.ndof)
        self.f_ext_applied = z(self.ndof)
        self.f_int = z(self.ndof)

        self._version = 0
        self._force_scale = 0.0
        self._cache = {}
        self.history = []
        self._precompute_reference_stiffness()

    # ======================================================================
    # element kernels (``sel`` is a slice or an index array of elements)
    # ======================================================================
    def _w(self, sel):
        return self.wq[None, :] if self.shared else self.wq[sel]

    def _grad(self, ue, sel):
        nc = ue.shape[0]
        if self.shared:
            return (ue @ self.Mg).reshape(nc, self.nq, 3, 3)
        U = ue.reshape(nc, 1, self.nn, 3).transpose(0, 1, 3, 2)      # (nc,1,3,nn)
        return U @ self.dN[sel]                                       # (nc,nq,3,3)

    def _force(self, S, sel):
        """Element nodal forces from weighted full stresses S (nc, nq, 3, 3)."""
        nc = S.shape[0]
        if self.shared:
            return S.reshape(nc, self.nq * 9) @ self.MgT
        F = S @ self.dN[sel].transpose(0, 1, 3, 2)                     # (nc,nq,3,nn)
        return F.sum(axis=1).transpose(0, 2, 1).reshape(nc, 3 * self.nn)

    def _bbar_strain(self, eps, w):
        th = C.trace(eps)
        thbar = (th * w).sum(axis=1) / w.sum(axis=1)
        return C.add_iso(eps, (thbar[:, None] - th) / 3.0)

    def _bbar_stress(self, sig, w):
        p = C.trace(sig) / 3.0
        pbar = (p * w).sum(axis=1) / w.sum(axis=1)
        return C.add_iso(sig, pbar[:, None] - p)

    def _elem_strain(self, ue, sel):
        eps = C.sym_from_grad(self._grad(ue, sel))
        if self.bbar:
            eps = self._bbar_strain(eps, self._w(sel))
        return eps

    def _elem_force_from_stress(self, sig, sel):
        w = self._w(sel)
        if self.bbar:
            sig = self._bbar_stress(sig, w)
        return self._force(C.full_from_sym(sig) * w[..., None, None], sel)

    def _eig(self, sel):
        return self.eig_cur[sel][..., None] * self.eig_dir[sel][:, None, :]

    def _elem_tangent(self, ve, sel):
        lam = self.lam[sel][:, None]
        mu = self.mu[sel][:, None]
        deps = self._elem_strain(ve, sel)
        if self.plastic:
            ds = C.tangent_stress(deps, lam, mu, self.theta[sel], self.theta_bar[sel], self.nrm[sel])
        else:
            ds = C.elastic_stress(deps, lam, mu)
        return self._elem_force_from_stress(ds, sel)

    def _precompute_reference_stiffness(self):
        """Reference element matrices K_lambda, K_mu for voxel meshes (fast elastic GEMM path)."""
        self.K_lam = self.K_mu = None
        if not self.shared:
            return
        xp = self.xp
        nd = 3 * self.nn
        I = xp.eye(nd, dtype=self.dtype)
        sel = slice(0, nd)
        eps = self._elem_strain(I, sel)
        one, zero = xp.ones((nd, 1), self.dtype), xp.zeros((nd, 1), self.dtype)
        self.K_lam = self._elem_force_from_stress(C.elastic_stress(eps, one, zero), sel)
        self.K_mu = self._elem_force_from_stress(C.elastic_stress(eps, zero, one), sel)

    def _chunks(self):
        for s in range(0, self.ne, self.chunk):
            e = min(self.ne, s + self.chunk)
            if self.active_host[s:e].any():
                yield slice(s, e)

    # ======================================================================
    # global operators
    # ======================================================================
    def internal_force(self, u):
        """Assemble f_int(u) and update the trial material state."""
        xp = self.xp
        f = xp.zeros(self.ndof, dtype=self.dtype)
        fabs = xp.zeros(self.ndof, dtype=self.dtype)
        for sl in self._chunks():
            eps = self._elem_strain(u[self.dofs[sl]], sl)
            if self.eps_ref is not None:
                eps = eps - self.eps_ref[sl]
            if self._has_eig:
                eps = eps - self._eig(sl)
            lam = self.lam[sl][:, None]
            mu = self.mu[sl][:, None]
            if self.plastic:
                sig, ep, al, th, thb, nrm, _ = C.j2_return(
                    eps - self.eps_p[sl], lam, mu, self.sy0[sl][:, None], self.H[sl][:, None],
                    self.alpha[sl], self.eps_p[sl])
                self.eps_p_t[sl], self.alpha_t[sl] = ep, al
                self.theta[sl], self.theta_bar[sl], self.nrm[sl] = th, thb, nrm
            else:
                sig = C.elastic_stress(eps, lam, mu)
            act = self.active[sl]
            sig = sig * act[:, None, None]
            self.sig[sl] = sig
            fe = self._elem_force_from_stress(sig, sl)
            f += scatter_add(self.ndof, self.dofs[sl], fe)
            fabs += scatter_add(self.ndof, self.dofs[sl], xp.abs(fe))
        # force scale that does not cancel (cf. Abaqus' time-averaged force):
        # used as convergence reference when f_int and f_ext vanish
        self._force_scale = float(xp.linalg.norm(fabs))
        return f

    def tangent(self, v):
        """Matrix-free product K_t v (consistent tangent; elastic if no plasticity)."""
        xp = self.xp
        f = xp.zeros(self.ndof, dtype=self.dtype)
        fast = self.shared and not self.plastic
        for sl in self._chunks():
            ve = v[self.dofs[sl]]
            if fast:
                fe = self.lam[sl][:, None] * (ve @ self.K_lam) + self.mu[sl][:, None] * (ve @ self.K_mu)
            else:
                fe = self._elem_tangent(ve, sl)
            fe *= self.active[sl][:, None]
            f += scatter_add(self.ndof, self.dofs[sl], fe)
        return f

    def diagonal(self):
        """Diagonal of the elastic stiffness (Jacobi preconditioner)."""
        xp = self.xp
        d = xp.zeros(self.ndof, dtype=self.dtype)
        for sl in self._chunks():
            lam, mu = self.lam[sl][:, None], self.mu[sl][:, None]
            if self.shared:
                de = lam * xp.diag(self.K_lam)[None, :] + mu * xp.diag(self.K_mu)[None, :]
            else:
                dN = self.dN[sl]
                w = self.wq[sl][..., None, None]
                de = (w * ((lam + mu)[..., None, None] * dN**2
                           + mu[..., None, None] * (dN**2).sum(-1, keepdims=True))).sum(axis=1)
                de = de.reshape(de.shape[0], -1)
            de *= self.active[sl][:, None]
            d += scatter_add(self.ndof, self.dofs[sl], de)
        return d

    def assemble(self, elastic: bool = False):
        """Assembled global (tangent) stiffness as a CSR matrix on the backend."""
        xp = self.xp
        sp = sparse_module()
        nd = 3 * self.nn
        K = None
        for sl in self._chunks():
            dofs = self.dofs[sl]
            nc = dofs.shape[0]
            if self.shared and (elastic or not self.plastic):
                Ke = (self.lam[sl][:, None, None] * self.K_lam[None] + self.mu[sl][:, None, None] * self.K_mu[None])
            else:
                Ke = xp.empty((nc, nd, nd), dtype=self.dtype)
                saved = None
                if elastic and self.plastic:
                    saved = (self.theta[sl].copy(), self.theta_bar[sl].copy())
                    self.theta[sl], self.theta_bar[sl] = 1.0, 0.0
                for k in range(nd):
                    ve = xp.zeros((nc, nd), dtype=self.dtype)
                    ve[:, k] = 1.0
                    Ke[:, k, :] = self._elem_tangent(ve, sl)
                if saved is not None:
                    self.theta[sl], self.theta_bar[sl] = saved
            Ke *= self.active[sl][:, None, None]
            rows = xp.broadcast_to(dofs[:, :, None], (nc, nd, nd)).ravel()
            cols = xp.broadcast_to(dofs[:, None, :], (nc, nd, nd)).ravel()
            # sum chunk by chunk: peak memory ~ one chunk of COO triplets + the CSR
            Kc = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()
            K = Kc if K is None else K + Kc
        return K

    # ======================================================================
    # boundary conditions, loads, eigenstrains, activation
    # ======================================================================
    def fix(self, nodes, comps=(0, 1, 2), value=0.0):
        """Prescribe displacement components of ``nodes`` (value: scalar or array)."""
        nodes = np.atleast_1d(np.asarray(nodes, np.int64))
        comps = np.atleast_1d(comps)
        dofs = (nodes[:, None] * 3 + comps[None, :])
        value = np.asarray(value, float)
        if value.ndim == 1 and len(value) == len(nodes) and len(nodes) != len(comps):
            value = value[:, None]                      # one value per node
        vals = np.broadcast_to(value, dofs.shape)       # scalar, per-comp, or (n, ncomp)
        self.fixed_host[dofs.ravel()] = True
        self.u_bc_target[self.xp.asarray(dofs.ravel())] = self.xp.asarray(vals.ravel(), dtype=self.dtype)
        self._version += 1

    def release(self, nodes, comps=(0, 1, 2)):
        nodes = np.atleast_1d(np.asarray(nodes, np.int64))
        dofs = (nodes[:, None] * 3 + np.atleast_1d(comps)[None, :]).ravel()
        self.fixed_host[dofs] = False
        self._version += 1

    def add_nodal_force(self, nodes, force):
        """Add force(s) (3,) or (n, 3) at nodes."""
        nodes = np.atleast_1d(np.asarray(nodes, np.int64))
        F = np.broadcast_to(np.asarray(force, float), (len(nodes), 3))
        f = np.zeros(self.ndof)
        np.add.at(f, (nodes[:, None] * 3 + np.arange(3)).ravel(), F.ravel())
        self.f_ext_target += self.xp.asarray(f, dtype=self.dtype)
        return f

    def add_surface_load(self, faces, face_type: str, *, total_force=None, traction=None,
                         pressure=None):
        """Consistent nodal loads from a surface load on boundary ``faces``.

        Exactly one of: ``total_force`` (3,) distributed as a uniform traction
        (like an Abaqus distributing coupling), ``traction`` (3,) [MPa], or
        ``pressure`` [MPa] (positive = pushing into the surface).
        Returns the host nodal force vector that was added.
        """
        N, ndA = face_integrals(self.mesh.points, faces, face_type)     # (nq,nfn), (nf,nq,3)
        dA = np.linalg.norm(ndA, axis=2)
        if pressure is not None:
            tdA = -float(pressure) * ndA
        else:
            if total_force is not None:
                traction = np.asarray(total_force, float) / dA.sum()
            tdA = dA[..., None] * np.asarray(traction, float)[None, None, :]
        fn = np.einsum("qa,fqi->fai", N, tdA)
        f = np.zeros(self.ndof)
        np.add.at(f, (faces[:, :, None] * 3 + np.arange(3)).ravel(), fn.ravel())
        self.f_ext_target += self.xp.asarray(f, dtype=self.dtype)
        return f

    def clear_loads(self):
        self.f_ext_target[:] = 0.0

    def reset_state(self):
        """Return to the unloaded, stress-free initial state (keeps mesh,
        materials, constraints and the cached preconditioner) -- e.g. to run
        several load cases on the same model."""
        self.u[:] = 0.0
        self.sig[:] = 0.0
        self.f_int[:] = 0.0
        self.f_ext_target[:] = 0.0
        self.f_ext_applied[:] = 0.0
        self.eig_dir[:] = 0.0
        self.eig_target[:] = 0.0
        self.eig_applied = self.xp.zeros_like(self.eig_target)
        self.eig_cur = self.eig_applied
        self.eps_ref = None
        if self.plastic:
            for a in (self.eps_p, self.alpha, self.eps_p_t, self.alpha_t, self.theta_bar, self.nrm):
                a[:] = 0.0
            self.theta[:] = 1.0

    def set_eigenstrain(self, cells, eps6, scale=1.0):
        """Eigenstrain (thermal/inherent strain) tensor(s) [xx,yy,zz,yz,xz,xy] on
        ``cells``; the target magnitude ``scale`` is reached in the next solve."""
        xp = self.xp
        cells = xp.asarray(np.atleast_1d(cells))
        self.eig_dir[cells] = xp.asarray(np.asarray(eps6, float), dtype=self.dtype)
        self.eig_target[cells] = scale
        self._has_eig = True

    def set_eigenstrain_scale(self, cells, scale):
        self.eig_target[self.xp.asarray(np.atleast_1d(cells))] = scale
        self._has_eig = True

    def set_eigenstrain_nodal(self, theta_nodes):
        """Eigenstrain magnitude from a nodal field interpolated to the
        integration points -- mirrors the "temperature + orthotropic expansion"
        way commercial codes apply eigenstrains (exact cross-code benchmarks)."""
        xp = self.xp
        th = xp.asarray(np.asarray(theta_nodes, float), dtype=self.dtype)
        Nq = xp.asarray(self.Nq, dtype=self.dtype)
        cells = xp.asarray(self.mesh.cells)
        for s in range(0, self.ne, self.chunk):
            self.eig_target[s:s + self.chunk] = th[cells[s:s + self.chunk]] @ Nq.T
        self._has_eig = True

    def deactivate(self, cells):
        cells = np.atleast_1d(np.asarray(cells))
        self.active_host[cells] = False
        self.active[self.xp.asarray(cells)] = 0.0
        self._version += 1

    def activate(self, cells, strain_free: bool = True, eig_birth: str = "qp"):
        """Element birth. ``strain_free`` elements carry no stress in the
        configuration in which they are born (``eps_ref`` = current strain).

        ``eig_birth`` sets which eigenstrain already present at birth is
        absorbed: ``"qp"`` (point-wise, exact strain-free state) or
        ``"element_mean"`` (element-average, reproduces CalculiX' behaviour for
        nodal temperature fields; verified against ccx 2.21)."""
        xp = self.xp
        cells = np.atleast_1d(np.asarray(cells))
        if len(cells) == 0:
            return
        idx = xp.asarray(cells)
        if strain_free:
            if self.eps_ref is None:
                self.eps_ref = xp.zeros((self.ne, self.nq, 6), dtype=self.dtype)
            for s in range(0, len(cells), self.chunk):
                sel = idx[s:s + self.chunk]
                eps = self._elem_strain(self.u[self.dofs[sel]], sel)
                if self._has_eig:
                    sc = self.eig_applied[sel]
                    if eig_birth == "element_mean":
                        w = self.wq[None, :] if self.shared else self.wq[sel]
                        sc = xp.broadcast_to(((sc * w).sum(axis=1) / w.sum(axis=1))[:, None], sc.shape)
                    elif eig_birth != "qp":
                        raise ValueError(eig_birth)
                    eps = eps - sc[..., None] * self.eig_dir[sel][:, None, :]
                self.eps_ref[sel] = eps
        if self.plastic:
            self.eps_p[idx] = 0.0
            self.alpha[idx] = 0.0
        self.active_host[cells] = True
        self.active[idx] = 1.0
        self._version += 1

    # ======================================================================
    # solution
    # ======================================================================
    def _free_mask(self):
        key = ("free", self._version)
        if key not in self._cache:
            node_act = np.zeros(self.n_nodes, bool)
            node_act[np.unique(self.mesh.cells[self.active_host])] = True
            free = np.repeat(node_act, 3) & ~self.fixed_host
            self._cache = {k: v for k, v in self._cache.items() if k[1] == self._version}
            self._cache[key] = (free, self.xp.asarray(free.astype(self.dtype)))
        return self._cache[key]

    def _linear_solver(self, kind, free_host, free_dev):
        key = (kind, self._version)
        if key in self._cache:
            return self._cache[key]
        if kind == "pcg":
            d = self.diagonal()
            Minv = self.xp.where(free_dev > 0, 1.0 / self.xp.where(d > 0, d, 1.0), 0.0)
            obj = Minv
        else:
            K = self.assemble(elastic=True)
            K = K.get() if on_gpu() else K
            idx = np.flatnonzero(free_host)
            K_ff = K[idx][:, idx]
            B = rigid_body_modes(self.mesh.points)[idx] if kind == "amg" else None
            obj = AssembledSolver(K_ff, kind, B=B)
        self._cache[key] = obj
        return obj

    def _solve_linear(self, R, kind, free_host, free_dev, rtol):
        xp = self.xp
        if kind == "pcg":
            Minv = self._linear_solver(kind, free_host, free_dev)
            du, info = pcg(lambda v: self.tangent(v) * free_dev, R * free_dev,
                           M=lambda r: Minv * r, rtol=rtol, maxiter=max(20000, self.ndof // 10))
            return du, info
        solver = self._linear_solver(kind, free_host, free_dev)
        idx = np.flatnonzero(free_host)
        idx_dev = xp.asarray(idx)

        def A(v):  # true tangent restricted to free dofs (host vectors)
            full = xp.zeros(self.ndof, dtype=self.dtype)
            full[idx_dev] = xp.asarray(v)
            return asnumpy(self.tangent(full)[idx_dev])

        tangent_is_elastic = not self.plastic
        x, info = solver.solve(asnumpy(R)[idx], A=None if tangent_is_elastic else A, rtol=rtol)
        du = xp.zeros(self.ndof, dtype=self.dtype)
        du[idx_dev] = xp.asarray(x, dtype=self.dtype)
        return du, info

    def solve(self, n_inc: int = 1, tol: float = 1e-6, max_iter: int = 30,
              solver: str = "auto", lin_rtol: Optional[float] = None, verbose: bool = False,
              name: str = "") -> StepInfo:
        """Ramp loads/BCs/eigenstrains from their applied to their target values
        in ``n_inc`` increments and find equilibrium (Newton-Raphson).

        solver: ``pcg`` (matrix-free Jacobi-PCG, GPU), ``amg`` (pyamg-
        preconditioned CG, CPU), ``direct`` (sparse LU, CPU) or ``auto``.
        """
        xp = self.xp
        t_start = time.perf_counter()
        if solver == "auto":
            if on_gpu():
                solver = "pcg"
            elif self.ndof <= 15_000:
                solver = "direct"
            else:
                try:
                    import pyamg  # noqa: F401
                    solver = "amg"
                except ImportError:
                    solver = "pcg"
        if lin_rtol is None:
            lin_rtol = 1e-9 if not self.plastic else 1e-7
        free_host, free_dev = self._free_mask()
        fixed_dev = xp.asarray(self.fixed_host)
        f0, f1 = self.f_ext_applied.copy(), self.f_ext_target.copy()
        e0, e1 = self.eig_applied.copy(), self.eig_target.copy()
        ub0 = self.u.copy()
        ub1 = xp.where(fixed_dev, self.u_bc_target, self.u)
        scale = float(self.mu.max()) * float(np.mean(np.ptp(self.mesh.points, axis=0))) ** 2
        info = StepInfo(False, n_inc, solver=solver)
        for inc in range(1, n_inc + 1):
            t = inc / n_inc
            fext = f0 + t * (f1 - f0)
            self.eig_cur = e0 + t * (e1 - e0)
            self.u = xp.where(fixed_dev, ub0 + t * (ub1 - ub0), self.u)
            converged = False
            R0 = 0.0
            for it in range(max_iter + 1):
                fint = self.internal_force(self.u)
                R = (fext - fint) * free_dev
                nR = float(xp.linalg.norm(R))
                if it == 0:
                    R0 = nR   # e.g. released reactions: f_int -> 0 at equilibrium after removal
                ref = max(float(xp.linalg.norm(fext)), float(xp.linalg.norm(fint)), R0,
                          self._force_scale, 1e-14 * scale)
                info.residuals.append(nR / ref)
                if verbose:
                    print(f"  [{name}] inc {inc}/{n_inc} it {it}: |R|/ref = {nR / ref:.3e}")
                if nR <= tol * ref:
                    converged = True
                    break
                if it == max_iter:
                    break
                du, kinfo = self._solve_linear(R, solver, free_host, free_dev, lin_rtol)
                info.krylov_iterations += kinfo.iterations
                if not kinfo.converged and verbose:
                    print(f"  [{name}] warning: linear solver stopped at rel. residual {kinfo.rel_residual:.2e}")
                self.u = self.u + du
            info.newton_iterations.append(it)
            if not converged:
                info.seconds = time.perf_counter() - t_start
                raise RuntimeError(f"Newton did not converge in step {name!r}: {info}")
            self._commit()
            self.f_int = fint
        self.f_ext_applied = f1
        self.eig_applied = e1
        self.eig_cur = self.eig_applied
        info.converged = True
        info.seconds = time.perf_counter() - t_start
        self.history.append((name, info))
        return info

    def _commit(self):
        if self.plastic:
            self.eps_p, self.eps_p_t = self.eps_p_t, self.eps_p
            self.alpha, self.alpha_t = self.alpha_t, self.alpha
            self.eps_p_t[...] = self.eps_p
            self.alpha_t[...] = self.alpha

    # ======================================================================
    # results (host arrays)
    # ======================================================================
    def displacement(self) -> np.ndarray:
        return asnumpy(self.u).reshape(-1, 3)

    def stress(self) -> np.ndarray:
        """Integration-point stresses (ne, nq, 6) [xx,yy,zz,yz,xz,xy]."""
        return asnumpy(self.sig)

    def plastic_strain(self):
        if not self.plastic:
            return None, None
        return asnumpy(self.eps_p), asnumpy(self.alpha)

    def reactions(self) -> np.ndarray:
        """Reaction forces at constrained dofs (n_nodes, 3); zero elsewhere."""
        r = asnumpy(self.f_int - self.f_ext_applied)
        r[~self.fixed_host] = 0.0
        return r.reshape(-1, 3)

    def strain_energy(self) -> float:
        """Elastic strain energy 0.5 * int sigma : C^-1 sigma (exact for linear elasticity)."""
        xp = self.xp
        lam, mu = self.lam[:, None], self.mu[:, None]
        E = mu * (3 * lam + 2 * mu) / (lam + mu)
        nu = lam / (2 * (lam + mu))
        s = self.sig
        tr = C.trace(s)
        ss = C.ddot(s, s)
        dens = 0.5 * ((1 + nu) * ss - nu * tr**2) / E
        w = self.wq[None, :] if self.shared else self.wq
        return float((dens * w * self.active[:, None]).sum())

    def summary(self) -> str:
        return (f"SolidModel: {self.ne:,} {self.mesh.cell_type} elements ({'B-bar' if self.bbar else 'standard'}), "
                f"{self.ndof:,} dofs, nq={self.nq}, plastic={self.plastic}, backend={backend_name()}, "
                f"{'shared voxel kernel' if self.shared else 'per-element gradients'}")
