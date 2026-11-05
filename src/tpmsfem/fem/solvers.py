"""Krylov solvers and preconditioners (backend agnostic)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..backend import array_module, asnumpy


@dataclass
class KrylovInfo:
    iterations: int
    rel_residual: float
    converged: bool
    seconds: float


def pcg(A, b, M=None, x0=None, rtol: float = 1e-8, maxiter: int = 20000) -> tuple:
    """Preconditioned conjugate gradients for SPD ``A``.

    ``A`` and ``M`` are callables ``v -> A v`` / ``r -> M^{-1} r``. Works for
    NumPy and CuPy vectors; only two scalars per iteration cross to the host.
    """
    xp = array_module(b)
    t0 = time.perf_counter()
    bnorm = float(xp.linalg.norm(b))
    if bnorm == 0.0:
        return xp.zeros_like(b), KrylovInfo(0, 0.0, True, 0.0)
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    r = b - A(x) if x0 is not None else b.copy()
    z = M(r) if M is not None else r
    p = z.copy()
    rz = float(xp.dot(r, z))
    rel = float(xp.linalg.norm(r)) / bnorm
    it = 0
    while rel > rtol and it < maxiter:
        Ap = A(p)
        pAp = float(xp.dot(p, Ap))
        if pAp <= 0:
            raise FloatingPointError("PCG: operator not positive definite "
                                     "(missing boundary conditions or a disconnected part?)")
        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap
        it += 1
        rel = float(xp.linalg.norm(r)) / bnorm
        if rel <= rtol:
            break
        z = M(r) if M is not None else r
        rz_new = float(xp.dot(r, z))
        p *= rz_new / rz
        p += z
        rz = rz_new
    return x, KrylovInfo(it, rel, rel <= rtol, time.perf_counter() - t0)


def rigid_body_modes(points: np.ndarray) -> np.ndarray:
    """(3n, 6) near-null space of 3D elasticity (for algebraic multigrid)."""
    x = points - points.mean(axis=0)
    n = len(x)
    B = np.zeros((n, 3, 6))
    B[:, 0, 0] = B[:, 1, 1] = B[:, 2, 2] = 1.0
    B[:, 1, 3], B[:, 2, 3] = -x[:, 2], x[:, 1]
    B[:, 0, 4], B[:, 2, 4] = x[:, 2], -x[:, 0]
    B[:, 0, 5], B[:, 1, 5] = -x[:, 1], x[:, 0]
    return B.reshape(3 * n, 6)


class AssembledSolver:
    """Host-side solver on an assembled free-dof matrix: ``direct`` (sparse LU)
    or ``amg`` (smoothed-aggregation AMG preconditioned CG via pyamg)."""

    def __init__(self, K_ff, kind: str, B=None, rtol: float = 1e-9):
        self.kind = kind
        self.rtol = rtol
        self.K = K_ff.tocsr()
        if kind == "direct":
            from scipy.sparse.linalg import factorized

            self._solve = factorized(self.K.tocsc())
        elif kind == "amg":
            import pyamg

            self.ml = pyamg.smoothed_aggregation_solver(
                self.K, B=B, symmetry="symmetric", strength=("symmetric", {"theta": 0.0}),
                max_coarse=1000)
            self.M = self.ml.aspreconditioner(cycle="V")
        else:
            raise ValueError(kind)

    def solve(self, b, A=None, rtol=None):
        """Solve with the assembled matrix, or -- if ``A`` is given -- use the
        factorisation/AMG of this matrix as a preconditioner for ``A``."""
        rtol = self.rtol if rtol is None else rtol
        b = asnumpy(b)
        if self.kind == "direct" and A is None:
            t0 = time.perf_counter()
            x = self._solve(b)
            return x, KrylovInfo(1, 0.0, True, time.perf_counter() - t0)
        Aop = (lambda v: self.K @ v) if A is None else A
        Mop = self.M if self.kind == "amg" else self._solve
        return pcg(Aop, b, M=lambda r: Mop(r), rtol=rtol, maxiter=5000)
