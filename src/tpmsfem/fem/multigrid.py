"""Geometric multigrid preconditioner for voxel (hex8) meshes.

Hierarchy: every level merges 2 x 2 x 2 voxels into one coarse voxel. The
prolongation P is trilinear interpolation on the nested grids, and the coarse
operators are exact Galerkin products ``A_{l+1} = P^T A_l P``. They are formed
element by element: a fine voxel at child position c of its parent contributes
``P_c^T K_e P_c``, with ``P_c`` one of eight constant 24 x 24 matrices. Coarse
operators are therefore stored as per-element 24 x 24 matrices and applied
matrix-free (gather, batched GEMV, scatter-add), just like the fine level, with no
assembled sparse matrix except on the coarsest level.

Dirichlet dofs are handled exactly. The fine element matrices are masked
(``D_e K_e D_e``) before the Galerkin product, prolongation is followed by the
fine mask, and coarse dofs left without stiffness become constrained.

Smoother: Chebyshev polynomial of the Jacobi-preconditioned operator (degree
``degree``, spectrum bound from power iteration). It is symmetric, so the V-cycle is a
valid preconditioner for CG. The coarsest level is solved with a sparse LU
(host) or, if still large, pyamg.

Works on NumPy and CuPy. The fine level uses the model's own matrix-free
tangent, so it also preconditions plasticity. The hierarchy is built from the
elastic stiffness, which is spectrally equivalent to the tangent for hardening plasticity.
"""
from __future__ import annotations

import time

import numpy as np

from ..backend import asnumpy, scatter_add, sparse_module
from ..mesh.voxel import HEX_CORNERS


def child_prolongators() -> np.ndarray:
    """P_c (8, 24, 24): coarse-element dofs -> fine-element dofs for child position c."""
    Pc = np.zeros((8, 24, 24))
    for c in range(8):
        off = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1], float)
        Pn = np.zeros((8, 8))
        for a in range(8):
            q = (off + HEX_CORNERS[a]) / 2.0
            for b in range(8):
                Pn[a, b] = np.prod(np.where(HEX_CORNERS[b] == 1, q, 1.0 - q))
        Pc[c] = np.kron(Pn, np.eye(3))
    return Pc


class _Level:
    pass


class VoxelMultigrid:
    def __init__(self, model, coarse_dofs: int = 20_000, degree: int = 2, ratio: float = 8.0,
                 max_levels: int = 12, chunk: int = 20_000, verbose: bool = False):
        if not model.shared:
            raise ValueError("VoxelMultigrid needs a voxel mesh")
        t0 = time.perf_counter()
        self.xp = xp = model.xp
        self.model = model
        self.degree, self.ratio = degree, ratio
        mesh = model.mesh
        dt = model.dtype
        self.sp = sparse_module()
        free_host, free_dev = model._free_mask()
        act = np.flatnonzero(model.active_host)
        h = float(mesh.meta["voxel_size"])
        periodic = tuple(mesh.meta.get("periodic", (False, False, False)))
        shape = np.array(mesh.meta["grid_shape"], np.int64)

        # ---------------- level 0: the model itself (matrix-free tangent)
        L0 = _Level()
        L0.ndof = model.ndof
        L0.free = free_dev
        L0.A = lambda v: model.tangent(v) * free_dev
        d0 = model.diagonal()
        L0.dinv = xp.where(free_dev > 0, 1.0 / xp.where(d0 > 0, d0, 1.0), 0.0)
        self.levels = [L0]

        # element data of level 0 used to build level 1
        ijk = np.asarray(mesh.cell_data["ijk"])[act]
        conn = mesh.cells[act]
        G = np.rint((mesh.points - mesh.meta["origin"]) / h).astype(np.int64)   # node lattice coords
        node_shape = np.where(periodic, shape, shape + 1)
        G %= node_shape
        Pc = xp.asarray(child_prolongators(), dtype=dt)
        free_e = free_dev
        lam, mu = model.lam[xp.asarray(act)], model.mu[xp.asarray(act)]
        Klam, Kmu = model.K_lam, model.K_mu
        Ke_prev = None                          # explicit element matrices of the previous level (l >= 1)

        while True:
            ne = len(ijk)
            ndof_l = self.levels[-1].ndof
            can = all((not p) or (s % 2 == 0) for p, s in zip(periodic, shape)) and (shape >= 2).any()
            if ndof_l <= coarse_dofs or not can or len(self.levels) >= max_levels:
                break
            # ---- coarse cells, child positions, coarse nodes
            pijk = ijk // 2
            cshape = np.where(periodic, shape // 2, (shape + 1) // 2)
            cnode_shape = np.where(periodic, cshape, cshape + 1)
            plin = (pijk[:, 0] * cshape[1] + pijk[:, 1]) * cshape[2] + pijk[:, 2]
            ucell, parent = np.unique(plin, return_inverse=True)
            parent = parent.ravel()
            cijk = np.stack([ucell // (cshape[1] * cshape[2]), (ucell // cshape[2]) % cshape[1], ucell % cshape[2]], 1)
            child = (ijk[:, 0] % 2) + 2 * (ijk[:, 1] % 2) + 4 * (ijk[:, 2] % 2)
            corner = (cijk[:, None, :] + HEX_CORNERS[None]) % cnode_shape
            clin = (corner[..., 0] * cnode_shape[1] + corner[..., 1]) * cnode_shape[2] + corner[..., 2]
            unode, cconn = np.unique(clin.ravel(), return_inverse=True)
            cconn = cconn.reshape(-1, 8)
            nnc = len(unode)
            # ---- prolongation on nodes (fine node lattice coords -> coarse nodes)
            nnf = len(G)
            used = np.zeros(nnf, bool)
            used[conn.ravel()] = True               # nodes of inactive elements get empty rows
            rows, cols, vals = [], [], []
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = np.ones(nnf)
                        cc = np.empty((nnf, 3), np.int64)
                        for d, dd in enumerate((dx, dy, dz)):
                            odd = G[:, d] % 2 == 1
                            cc[:, d] = np.where(odd, (G[:, d] - 1) // 2 + dd, G[:, d] // 2)
                            w *= np.where(odd, 0.5, 1.0 if dd == 0 else 0.0)
                        keep = (w > 0) & used
                        cc %= cnode_shape
                        lin = (cc[:, 0] * cnode_shape[1] + cc[:, 1]) * cnode_shape[2] + cc[:, 2]
                        pos = np.searchsorted(unode, lin[keep])
                        if np.any(pos >= nnc) or np.any(unode[np.minimum(pos, nnc - 1)] != lin[keep]):
                            raise RuntimeError("multigrid: prolongation stencil outside the coarse mesh")
                        rows.append(np.flatnonzero(keep))
                        cols.append(pos)
                        vals.append(w[keep])
            P = self.sp.csr_matrix((xp.asarray(np.concatenate(vals), dtype=dt),
                                    (xp.asarray(np.concatenate(rows)), xp.asarray(np.concatenate(cols)))),
                                   shape=(nnf, nnc))
            # ---- Galerkin coarse element matrices
            nec = len(ucell)
            Kc = xp.zeros((nec, 24, 24), dtype=dt)
            fdofs = conn[:, :, None] * 3 + np.arange(3)
            fdofs = fdofs.reshape(ne, 24)
            parent_d, child_d = xp.asarray(parent), xp.asarray(child)
            if Ke_prev is None:
                # level 0: P_c^T (lam K_lam + mu K_mu) P_c = lam A_c + mu B_c for elements
                # without constrained dofs (no matrix products needed)
                Ac = xp.stack([Pc[c].T @ Klam @ Pc[c] for c in range(8)])
                Bc = xp.stack([Pc[c].T @ Kmu @ Pc[c] for c in range(8)])
            for s in range(0, ne, chunk):
                e = slice(s, min(ne, s + chunk))
                m = free_e[xp.asarray(fdofs[e])]
                masked = (m < 1).any(axis=1)
                for c in range(8):
                    sel = xp.flatnonzero((child_d[e] == c) & ~masked) if Ke_prev is None else \
                        xp.flatnonzero(child_d[e] == c)
                    if sel.size:
                        if Ke_prev is None:
                            T = lam[e][sel][:, None, None] * Ac[c][None] + mu[e][sel][:, None, None] * Bc[c][None]
                        else:
                            K = Ke_prev[e][sel] * m[sel][:, :, None] * m[sel][:, None, :]
                            T = Pc[c].T @ K @ Pc[c]
                        Kc[parent_d[e][sel]] += T             # unique parents per child position
                    if Ke_prev is None:
                        selm = xp.flatnonzero((child_d[e] == c) & masked)
                        if selm.size:
                            K = lam[e][selm][:, None, None] * Klam[None] + mu[e][selm][:, None, None] * Kmu[None]
                            K = K * m[selm][:, :, None] * m[selm][:, None, :]
                            Kc[parent_d[e][selm]] += Pc[c].T @ K @ Pc[c]
            cdofs = (cconn[:, :, None] * 3 + np.arange(3)).reshape(nec, 24)
            cdofs_d = xp.asarray(cdofs)
            ndc = 3 * nnc
            diag = scatter_add(ndc, cdofs_d, xp.diagonal(Kc, axis1=1, axis2=2))
            freec = (diag > 1e-12 * float(diag.max())).astype(dt)
            Lc = _Level()
            Lc.ndof, Lc.free, Lc.P, Lc.K, Lc.dofs = ndc, freec, P, Kc, cdofs_d
            Lc.dinv = xp.where(freec > 0, 1.0 / xp.where(diag > 0, diag, 1.0), 0.0)
            Lc.A = self._make_op(Lc)
            self.levels.append(Lc)
            # next iteration works on this level
            ijk, conn, shape, free_e, Ke_prev = cijk, cconn, cshape, freec, Kc
            G = np.empty((nnc, 3), np.int64)              # lattice coords of the coarse nodes
            G[cconn.ravel()] = corner.reshape(-1, 3)
            if verbose:
                print(f"  gmg level {len(self.levels) - 1}: {nec:,} cells, {ndc:,} dofs", flush=True)

        self._coarse_setup(self.levels[-1])
        for L in self.levels[:-1]:
            L.lmax = self._lmax(L)
        self.setup_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------
    def _make_op(self, L):
        xp = self.xp

        def A(v):
            ve = v[L.dofs]
            fe = (L.K @ ve[:, :, None])[:, :, 0]                 # batched GEMV (BLAS / cuBLAS)
            return scatter_add(L.ndof, L.dofs, fe) * L.free
        return A

    def _lmax(self, L, its: int = 15):
        xp = self.xp
        rng = np.random.default_rng(0)
        x = xp.asarray(rng.standard_normal(L.ndof)) * L.free
        lam = 1.0
        for _ in range(its):
            y = L.dinv * L.A(x)
            lam = float(xp.linalg.norm(y)) / max(float(xp.linalg.norm(x)), 1e-300)
            x = y / max(float(xp.linalg.norm(y)), 1e-300)
        return 1.1 * lam

    def _coarse_setup(self, L):
        import scipy.sparse as ssp

        if not hasattr(L, "K"):                     # single level: fall back to Jacobi
            L.solve = lambda r: L.dinv * r
            return
        K = asnumpy(L.K)
        dofs = asnumpy(L.dofs)
        n = L.ndof
        r = np.broadcast_to(dofs[:, :, None], K.shape).ravel()
        c = np.broadcast_to(dofs[:, None, :], K.shape).ravel()
        A = ssp.coo_matrix((K.ravel(), (r, c)), shape=(n, n)).tocsr()
        free = np.flatnonzero(asnumpy(L.free) > 0)
        A = A[free][:, free].tocsc()
        if len(free) <= 80_000:
            from scipy.sparse.linalg import splu

            lu = splu(A)
            solve = lu.solve
        else:
            import pyamg

            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), symmetry="symmetric")
            solve = lambda b: ml.solve(b, tol=1e-6, maxiter=50)
        xp = self.xp
        free_d = xp.asarray(free)

        def coarse(rhs):
            out = xp.zeros(n, dtype=rhs.dtype)
            out[free_d] = xp.asarray(solve(asnumpy(rhs)[free]))
            return out
        L.solve = coarse

    def _smooth(self, L, x, b):
        """Chebyshev iteration on D^-1 A over [lmax/ratio, lmax], starting at x."""
        xp = self.xp
        lmax = L.lmax
        lmin = lmax / self.ratio
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        r = b - L.A(x) if x is not None else b.copy()
        x = xp.zeros_like(b) if x is None else x.copy()
        d = L.dinv * r / theta
        for k in range(self.degree):
            x += d
            if k == self.degree - 1:
                break
            r -= L.A(d)
            rho_new = 1.0 / (2.0 * sigma - rho)
            d = rho_new * rho * d + (2.0 * rho_new / delta) * (L.dinv * r)
            rho = rho_new
        return x

    def vcycle(self, r, l: int = 0):
        L = self.levels[l]
        if l == len(self.levels) - 1:
            return L.solve(r)
        C = self.levels[l + 1]
        x = self._smooth(L, None, r)
        res = (r - L.A(x)).reshape(-1, 3)
        rc = (C.P.T @ res).reshape(-1) * C.free
        ec = self.vcycle(rc, l + 1)
        x = x + (C.P @ ec.reshape(-1, 3)).reshape(-1) * L.free
        return self._smooth(L, x, r)

    def summary(self) -> str:
        return (f"GMG: {len(self.levels)} levels, dofs " + " -> ".join(f"{L.ndof:,}" for L in self.levels)
                + f", setup {self.setup_seconds:.1f}s")
