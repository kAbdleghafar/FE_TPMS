"""Fused CPU kernels (numba) for the free-surface lattice Boltzmann step.

One pass per cell does what the array version (``FreeSurfaceLBM.step`` with NumPy/CuPy) does
in ~600 whole-array operations: pull streaming, wall bounce-back, gas-side reconstruction,
interface mass exchange, the link fluxes that carry the enthalpy, macroscopic moments, Darcy
damping, the speed limiter and the TRT collision with Guo forcing. The loop structure is that
of a CUDA kernel (one thread per cell). Neighbour sums used by the interface bookkeeping are
provided as stencils too. Results agree with the array version to round-off (tested).
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:                                  # pragma: no cover
    HAVE_NUMBA = False

GAS, INTERFACE, LIQUID, WALL = 0, 1, 2, 3


def available():
    return HAVE_NUMBA


if HAVE_NUMBA:
    @nb.njit(cache=True)
    def _tables(n):
        t = np.empty((3, n), dtype=np.int64)
        for i in range(n):
            t[0, i] = (i - 1) % n
            t[1, i] = i
            t[2, i] = (i + 1) % n
        return t

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def stream_collide(f, flag, mass, eps, rho_g, u_prev, h, solid, Fext, K, g, wp, wm, u_limit, carry,
                       C, W, OPP, f_out, rho_out, u_out, mass_out, E_in, M_in, M_out):
        nx, ny, nz = flag.shape
        Q = C.shape[0]
        limited = 0
        # periodic neighbour index tables (as np.roll): T[d + 1, i] = (i + d) mod n
        TX, TY, TZ = _tables(nx), _tables(ny), _tables(nz)
        for i in nb.prange(nx):
            fs = np.empty(Q)
            feq = np.empty(Q)
            for j in range(ny):
                for k in range(nz):
                    fl = flag[i, j, k]
                    if fl != LIQUID and fl != INTERFACE:
                        for q in range(Q):
                            f_out[q, i, j, k] = f[q, i, j, k]
                        rho_out[i, j, k] = 1.0
                        u_out[0, i, j, k] = 0.0
                        u_out[1, i, j, k] = 0.0
                        u_out[2, i, j, k] = 0.0
                        mass_out[i, j, k] = mass[i, j, k]
                        if carry:
                            E_in[i, j, k] = 0.0
                            M_in[i, j, k] = 0.0
                            M_out[i, j, k] = 0.0
                        continue
                    itf = fl == INTERFACE
                    rg = rho_g[i, j, k]
                    ux0, uy0, uz0 = u_prev[0, i, j, k], u_prev[1, i, j, k], u_prev[2, i, j, k]
                    usq0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0
                    dm = 0.0
                    ein = 0.0
                    min_ = 0.0
                    mout = 0.0
                    ex_ = eps[i, j, k]
                    sol = solid[i, j, k]
                    for q in range(Q):
                        if q == 0:
                            fs[0] = f[0, i, j, k]
                            continue
                        o = OPP[q]
                        ii = TX[1 - C[q, 0], i]
                        jj = TY[1 - C[q, 1], j]
                        kk = TZ[1 - C[q, 2], k]
                        sf = flag[ii, jj, kk]
                        if sf == WALL:
                            fi = f[o, i, j, k]
                        else:
                            fi = f[q, ii, jj, kk]
                        ex = fi - f[o, i, j, k]
                        J = 0.0
                        if itf:
                            if sf == GAS:
                                cu = C[q, 0] * ux0 + C[q, 1] * uy0 + C[q, 2] * uz0
                                fi = (W[q] * rg * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq0)
                                      + W[o] * rg * (1.0 - 3.0 * cu + 4.5 * cu * cu - 1.5 * usq0) - f[o, i, j, k])
                            if sf == LIQUID:
                                wgt = 1.0
                            elif sf == INTERFACE:
                                wgt = 0.5 * (ex_ + eps[ii, jj, kk])
                                if sol and solid[ii, jj, kk]:
                                    wgt = 0.0
                            else:
                                wgt = 0.0
                            dm += wgt * ex
                            J = wgt * ex
                        else:
                            if sf == LIQUID or sf == INTERFACE:
                                J = ex
                        if carry:
                            if J > 0.0:
                                min_ += J
                                ein += J * h[ii, jj, kk]
                            else:
                                mout -= J
                        fs[q] = fi
                    # moments
                    rho = 0.0
                    jx = 0.0
                    jy = 0.0
                    jz = 0.0
                    for q in range(Q):
                        rho += fs[q]
                        jx += C[q, 0] * fs[q]
                        jy += C[q, 1] * fs[q]
                        jz += C[q, 2] * fs[q]
                    Fx = rho * g[0] + Fext[0, i, j, k]
                    Fy = rho * g[1] + Fext[1, i, j, k]
                    Fz = rho * g[2] + Fext[2, i, j, k]
                    Kd = K[i, j, k]
                    den = rho * (1.0 + 0.5 * Kd)
                    ux = (jx + 0.5 * Fx) / den
                    uy = (jy + 0.5 * Fy) / den
                    uz = (jz + 0.5 * Fz) / den
                    if u_limit > 0.0:
                        sp = np.sqrt(ux * ux + uy * uy + uz * uz)
                        if sp > u_limit:
                            s = u_limit / sp
                            ux *= s
                            uy *= s
                            uz *= s
                            limited += 1
                    Fx -= Kd * rho * ux
                    Fy -= Kd * rho * uy
                    Fz -= Kd * rho * uz
                    usq = ux * ux + uy * uy + uz * uz
                    uF = ux * Fx + uy * Fy + uz * Fz
                    for q in range(Q):
                        cu = C[q, 0] * ux + C[q, 1] * uy + C[q, 2] * uz
                        feq[q] = W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
                    for q in range(Q):
                        o = OPP[q]
                        fp = 0.5 * (fs[q] + fs[o])
                        fm = 0.5 * (fs[q] - fs[o])
                        ep = 0.5 * (feq[q] + feq[o])
                        em = 0.5 * (feq[q] - feq[o])
                        cF = C[q, 0] * Fx + C[q, 1] * Fy + C[q, 2] * Fz
                        cu = C[q, 0] * ux + C[q, 1] * uy + C[q, 2] * uz
                        Sp = W[q] * (9.0 * cu * cF - 3.0 * uF)
                        Sm = W[q] * 3.0 * cF
                        f_out[q, i, j, k] = (fs[q] - wp * (fp - ep) - wm * (fm - em)
                                             + (1.0 - 0.5 * wp) * Sp + (1.0 - 0.5 * wm) * Sm)
                    rho_out[i, j, k] = rho
                    u_out[0, i, j, k] = ux
                    u_out[1, i, j, k] = uy
                    u_out[2, i, j, k] = uz
                    if itf:
                        mass_out[i, j, k] = mass[i, j, k] + dm
                    else:
                        mass_out[i, j, k] = rho
                    if carry:
                        E_in[i, j, k] = ein
                        M_in[i, j, k] = min_
                        M_out[i, j, k] = mout
        return limited

    # the neighbour stencils are small and called often: serial (no thread start-up per call)
    @nb.njit(cache=True)
    def nb_sum(a, C):
        """Sum of ``a`` over the 18 neighbours (periodic indices, as np.roll)."""
        nx, ny, nz = a.shape
        TX, TY, TZ = _tables(nx), _tables(ny), _tables(nz)
        out = np.zeros(a.shape)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    s = 0.0
                    for q in range(1, C.shape[0]):
                        s += a[TX[1 + C[q, 0], i], TY[1 + C[q, 1], j], TZ[1 + C[q, 2], k]]
                    out[i, j, k] = s
        return out

    @nb.njit(cache=True)
    def nb_sum_stack(A, C):
        """Neighbour sums of several fields in one pass: A (k, nx, ny, nz) -> (k, nx, ny, nz)."""
        nf, nx, ny, nz = A.shape
        TX, TY, TZ = _tables(nx), _tables(ny), _tables(nz)
        out = np.zeros(A.shape)
        acc = np.empty(nf)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for f in range(nf):
                        acc[f] = 0.0
                    for q in range(1, C.shape[0]):
                        a, b, c = TX[1 + C[q, 0], i], TY[1 + C[q, 1], j], TZ[1 + C[q, 2], k]
                        for f in range(nf):
                            acc[f] += A[f, a, b, c]
                    for f in range(nf):
                        out[f, i, j, k] = acc[f]
        return out

    @nb.njit(cache=True)
    def nb_any(m, C):
        """True where any of the 18 neighbours is True."""
        nx, ny, nz = m.shape
        TX, TY, TZ = _tables(nx), _tables(ny), _tables(nz)
        out = np.zeros(m.shape, dtype=np.bool_)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for q in range(1, C.shape[0]):
                        if m[TX[1 + C[q, 0], i], TY[1 + C[q, 1], j], TZ[1 + C[q, 2], k]]:
                            out[i, j, k] = True
                            break
        return out
