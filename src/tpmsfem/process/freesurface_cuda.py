"""Fused CUDA kernels (CuPy) for the free-surface lattice Boltzmann step.

The GPU counterpart of :mod:`tpmsfem.process.freesurface_kernels` (numba): one thread per
cell does the pull streaming, wall bounce-back, gas-side reconstruction, interface mass
exchange with the one-way rules, the link fluxes that carry the enthalpy, the moments, Darcy
damping, the speed limiter, the Smagorinsky relaxation time and the TRT collision with Guo
forcing, in the same order of operations as the numba kernel. The neighbour stencils used by
the interface bookkeeping (any / sum / several sums over the 18 neighbours) are kernels too.
Periodic neighbour indices, as ``np.roll``. Templated on the floating-point type.
"""
from __future__ import annotations

import numpy as np

_SRC = r"""
#define GAS 0
#define INTERFACE 1
#define LIQUID 2
#define WALL 3
__constant__ int CX[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int CY[19] = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int CZ[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
__constant__ int OPP[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
__constant__ double WD[19] = {1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

__device__ __forceinline__ int wrap(int a, int n) { return a < 0 ? a + n : (a >= n ? a - n : a); }

template<typename REAL>
__global__ void fs_stream_collide(
    const REAL* __restrict__ f, const signed char* __restrict__ flag, const REAL* __restrict__ mass,
    const REAL* __restrict__ eps, const REAL* __restrict__ rho_g, const REAL* __restrict__ u_prev,
    const REAL* __restrict__ h, const unsigned char* __restrict__ solid, const REAL* __restrict__ Fext,
    const REAL* __restrict__ Karr, const REAL Kconst, const REAL gx, const REAL gy, const REAL gz,
    const REAL wp, const REAL wm, const REAL u_limit, const int carry, const int has_solid,
    const int has_force, const int has_K, const signed char* __restrict__ itype,
    const REAL tau0, const REAL magic, const REAL cs,
    REAL* __restrict__ f_out, REAL* __restrict__ rho_out, REAL* __restrict__ u_out,
    REAL* __restrict__ mass_out, REAL* __restrict__ E_in, REAL* __restrict__ M_in, REAL* __restrict__ M_out,
    unsigned long long* limited, const int nx, const int ny, const int nz)
{
    const long long N = (long long)nx * ny * nz;
    const long long c = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= N) return;
    const int k = (int)(c % nz);
    const int j = (int)((c / nz) % ny);
    const int i = (int)(c / ((long long)ny * nz));
    const signed char fl = flag[c];
    if (fl != LIQUID && fl != INTERFACE) {
        for (int q = 0; q < 19; ++q) f_out[q * N + c] = f[q * N + c];
        rho_out[c] = (REAL)1.0;
        u_out[c] = (REAL)0.0; u_out[N + c] = (REAL)0.0; u_out[2 * N + c] = (REAL)0.0;
        mass_out[c] = mass[c];
        if (carry) { E_in[c] = (REAL)0.0; M_in[c] = (REAL)0.0; M_out[c] = (REAL)0.0; }
        return;
    }
    const bool itf = fl == INTERFACE;
    const REAL rg = rho_g[c];
    const REAL ux0 = u_prev[c], uy0 = u_prev[N + c], uz0 = u_prev[2 * N + c];
    const REAL usq0 = ux0 * ux0 + uy0 * uy0 + uz0 * uz0;
    REAL dm = 0.0, ein = 0.0, min_ = 0.0, mout = 0.0;
    const REAL ex_ = eps[c];
    const bool sol = has_solid ? (solid[c] != 0) : false;
    const signed char a_t = itype[c];
    REAL fs[19];
    fs[0] = f[c];
    for (int q = 1; q < 19; ++q) {
        const int o = OPP[q];
        const int ii = wrap(i - CX[q], nx), jj = wrap(j - CY[q], ny), kk = wrap(k - CZ[q], nz);
        const long long s = ((long long)ii * ny + jj) * nz + kk;
        const signed char sf = flag[s];
        const REAL fo = f[o * N + c];
        REAL fi = (sf == WALL) ? fo : f[q * N + s];
        REAL ex = fi - fo;
        REAL J = 0.0;
        if (itf) {
            REAL wgt;
            if (sf == GAS) {
                const REAL cu = CX[q] * ux0 + CY[q] * uy0 + CZ[q] * uz0;
                fi = ((REAL)WD[q] * rg * ((REAL)1.0 + (REAL)3.0 * cu + (REAL)4.5 * cu * cu - (REAL)1.5 * usq0)
                      + (REAL)WD[o] * rg * ((REAL)1.0 - (REAL)3.0 * cu + (REAL)4.5 * cu * cu - (REAL)1.5 * usq0) - fo);
            }
            if (sf == LIQUID) {
                wgt = 1.0;
            } else if (sf == INTERFACE) {
                wgt = (REAL)0.5 * (ex_ + eps[s]);
                if (sol && solid[s]) wgt = 0.0;
                const signed char b_t = itype[s];
                if (a_t != b_t) {
                    if (a_t == 1 || b_t == 2) ex = ex < (REAL)0.0 ? ex : (REAL)0.0;
                    else ex = ex > (REAL)0.0 ? ex : (REAL)0.0;
                }
            } else {
                wgt = 0.0;
            }
            dm += wgt * ex;
            J = wgt * ex;
        } else {
            if (sf == LIQUID || sf == INTERFACE) J = ex;
        }
        if (carry) {
            if (J > (REAL)0.0) { min_ += J; ein += J * h[s]; }
            else { mout -= J; }
        }
        fs[q] = fi;
    }
    REAL rho = 0.0, jx = 0.0, jy = 0.0, jz = 0.0;
    for (int q = 0; q < 19; ++q) {
        rho += fs[q];
        jx += CX[q] * fs[q];
        jy += CY[q] * fs[q];
        jz += CZ[q] * fs[q];
    }
    REAL Fx = rho * gx + (has_force ? Fext[c] : (REAL)0.0);
    REAL Fy = rho * gy + (has_force ? Fext[N + c] : (REAL)0.0);
    REAL Fz = rho * gz + (has_force ? Fext[2 * N + c] : (REAL)0.0);
    const REAL Kd = has_K ? Karr[c] : Kconst;
    const REAL den = rho * ((REAL)1.0 + (REAL)0.5 * Kd);
    REAL ux = (jx + (REAL)0.5 * Fx) / den;
    REAL uy = (jy + (REAL)0.5 * Fy) / den;
    REAL uz = (jz + (REAL)0.5 * Fz) / den;
    if (u_limit > (REAL)0.0) {
        const REAL sp = sqrt(ux * ux + uy * uy + uz * uz);
        if (sp > u_limit) {
            const REAL sc = u_limit / sp;
            ux *= sc; uy *= sc; uz *= sc;
            atomicAdd(limited, 1ULL);
        }
    }
    Fx -= Kd * rho * ux;
    Fy -= Kd * rho * uy;
    Fz -= Kd * rho * uz;
    const REAL usq = ux * ux + uy * uy + uz * uz;
    const REAL uF = ux * Fx + uy * Fy + uz * Fz;
    REAL feq[19];
    for (int q = 0; q < 19; ++q) {
        const REAL cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        feq[q] = (REAL)WD[q] * rho * ((REAL)1.0 + (REAL)3.0 * cu + (REAL)4.5 * cu * cu - (REAL)1.5 * usq);
    }
    REAL wpc = wp, wmc = wm;
    if (cs > (REAL)0.0) {
        REAL pxx = 0.0, pyy = 0.0, pzz = 0.0, pxy = 0.0, pxz = 0.0, pyz = 0.0;
        for (int q = 0; q < 19; ++q) {
            const REAL nq = fs[q] - feq[q];
            pxx += CX[q] * CX[q] * nq;
            pyy += CY[q] * CY[q] * nq;
            pzz += CZ[q] * CZ[q] * nq;
            pxy += CX[q] * CY[q] * nq;
            pxz += CX[q] * CZ[q] * nq;
            pyz += CY[q] * CZ[q] * nq;
        }
        const REAL pn = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + (REAL)2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
        const REAL te = (REAL)0.5 * (tau0 + sqrt(tau0 * tau0 + (REAL)(18.0 * 1.4142135623730951) * cs * cs * pn / rho));
        wpc = (REAL)1.0 / te;
        wmc = (REAL)1.0 / (magic / (te - (REAL)0.5) + (REAL)0.5);
    }
    for (int q = 0; q < 19; ++q) {
        const int o = OPP[q];
        const REAL fp = (REAL)0.5 * (fs[q] + fs[o]);
        const REAL fm = (REAL)0.5 * (fs[q] - fs[o]);
        const REAL ep = (REAL)0.5 * (feq[q] + feq[o]);
        const REAL em = (REAL)0.5 * (feq[q] - feq[o]);
        const REAL cF = CX[q] * Fx + CY[q] * Fy + CZ[q] * Fz;
        const REAL cu = CX[q] * ux + CY[q] * uy + CZ[q] * uz;
        const REAL Sp = (REAL)WD[q] * ((REAL)9.0 * cu * cF - (REAL)3.0 * uF);
        const REAL Sm = (REAL)WD[q] * (REAL)3.0 * cF;
        f_out[q * N + c] = (fs[q] - wpc * (fp - ep) - wmc * (fm - em)
                            + ((REAL)1.0 - (REAL)0.5 * wpc) * Sp + ((REAL)1.0 - (REAL)0.5 * wmc) * Sm);
    }
    rho_out[c] = rho;
    u_out[c] = ux; u_out[N + c] = uy; u_out[2 * N + c] = uz;
    mass_out[c] = itf ? mass[c] + dm : rho;
    if (carry) { E_in[c] = ein; M_in[c] = min_; M_out[c] = mout; }
}

// sums over the 18 neighbours x + c of nf stacked fields A (nf, N)
template<typename REAL>
__global__ void fs_nb_sum(const REAL* __restrict__ A, REAL* __restrict__ out, const int nf,
                          const int nx, const int ny, const int nz)
{
    const long long N = (long long)nx * ny * nz;
    const long long c = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= N) return;
    const int k = (int)(c % nz);
    const int j = (int)((c / nz) % ny);
    const int i = (int)(c / ((long long)ny * nz));
    REAL acc[8];
    for (int f = 0; f < nf; ++f) acc[f] = 0.0;
    for (int q = 1; q < 19; ++q) {
        const long long s = ((long long)wrap(i + CX[q], nx) * ny + wrap(j + CY[q], ny)) * nz + wrap(k + CZ[q], nz);
        for (int f = 0; f < nf; ++f) acc[f] += A[f * N + s];
    }
    for (int f = 0; f < nf; ++f) out[f * N + c] = acc[f];
}

// value of the neighbour along axis ax (d = +1 / -1); across a wall the cell's own value
template<typename REAL>
__device__ __forceinline__ REAL mirror(const REAL* __restrict__ a, const signed char* __restrict__ flag,
                                       long long c, int i, int j, int k, int ax, int d, int nx, int ny, int nz)
{
    int ii = i, jj = j, kk = k;
    if (ax == 0) ii = wrap(i + d, nx); else if (ax == 1) jj = wrap(j + d, ny); else kk = wrap(k + d, nz);
    const long long s = ((long long)ii * ny + jj) * nz + kk;
    return flag[s] == WALL ? a[c] : a[s];
}

#define CELL_IJK \
    const long long N = (long long)nx * ny * nz; \
    const long long c = (long long)blockDim.x * blockIdx.x + threadIdx.x; \
    if (c >= N) return; \
    const int k = (int)(c % nz); \
    const int j = (int)((c / nz) % ny); \
    const int i = (int)(c / ((long long)ny * nz));

// one [1 2 1] smoothing pass along axis ax, walls as mirrors
template<typename REAL>
__global__ void fs_smooth_axis(const REAL* __restrict__ e, const signed char* __restrict__ flag, REAL* __restrict__ out,
                               const int ax, const int nx, const int ny, const int nz)
{
    CELL_IJK
    out[c] = (REAL)0.25 * (mirror(e, flag, c, i, j, k, ax, -1, nx, ny, nz) + (REAL)2.0 * e[c]
                           + mirror(e, flag, c, i, j, k, ax, 1, nx, ny, nz));
}

// unit normal from the central-difference gradient of the smoothed fill
template<typename REAL>
__global__ void fs_normal(const REAL* __restrict__ e, const signed char* __restrict__ flag, REAL* __restrict__ n,
                          const int nx, const int ny, const int nz)
{
    CELL_IJK
    REAL g[3];
    for (int ax = 0; ax < 3; ++ax)
        g[ax] = (REAL)0.5 * (mirror(e, flag, c, i, j, k, ax, 1, nx, ny, nz) - mirror(e, flag, c, i, j, k, ax, -1, nx, ny, nz));
    const REAL norm = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) + (REAL)1e-12;
    n[c] = g[0] / norm; n[N + c] = g[1] / norm; n[2 * N + c] = g[2] / norm;
}

// curvature: minus the divergence of the normal
template<typename REAL>
__global__ void fs_curvature(const REAL* __restrict__ n, const signed char* __restrict__ flag, REAL* __restrict__ kap,
                             const int nx, const int ny, const int nz)
{
    CELL_IJK
    REAL div = 0.0;
    for (int ax = 0; ax < 3; ++ax) {
        const REAL* na = n + ax * N;
        div += (REAL)0.5 * (mirror(na, flag, c, i, j, k, ax, 1, nx, ny, nz) - mirror(na, flag, c, i, j, k, ax, -1, nx, ny, nz));
    }
    kap[c] = -div;
}

// linear interpolation on a uniform table x0 + i dx (clamped at the ends, as np.interp)
template<typename REAL>
__device__ __forceinline__ REAL interp_uniform(REAL x, const REAL* __restrict__ fp, REAL x0, REAL dx, int n)
{
    const REAL s = (x - x0) / dx;
    if (!(s > (REAL)0.0)) return fp[0];
    if (s >= (REAL)(n - 1)) return fp[n - 1];
    int m = (int)s;
    if (m > n - 2) m = n - 2;
    const REAL xm = x0 + m * dx;
    return (fp[m + 1] - fp[m]) / dx * (x - xm) + fp[m];
}

// linear interpolation on a monotone table xp (binary search, clamped)
template<typename REAL>
__device__ __forceinline__ REAL interp_sorted(REAL x, const REAL* __restrict__ xp, const REAL* __restrict__ fp, int n)
{
    if (!(x > xp[0])) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    return (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]) * (x - xp[lo]) + fp[lo];
}

// conductivity k(T) of every cell (uniform temperature table)
template<typename REAL>
__global__ void fs_conductivity(const REAL* __restrict__ T, const REAL* __restrict__ kT, REAL* __restrict__ kc,
                                const REAL t0, const REAL dtab, const int ntab, const long long N)
{
    const long long c = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= N) return;
    kc[c] = interp_uniform(T[c], kT, t0, dtab, ntab);
}

// one explicit enthalpy step: conduction over the six faces (harmonic-mean conductance,
// scaled by min(s_a, s_b)), optional upwind advection, sources with a floored capacity, and
// T(H). Writes H, T and the absorbed / lost energy density weights.
template<typename REAL>
__global__ void fs_heat(const REAL* __restrict__ T, const REAL* __restrict__ H, const REAL* __restrict__ kc,
                        const REAL* __restrict__ w, const signed char* __restrict__ flag, const REAL* __restrict__ u,
                        const REAL* __restrict__ q, const REAL* __restrict__ loss, const int has_q, const int has_loss,
                        const int advect, const REAL dx, const REAL dt, const REAL udx, const REAL eps_min,
                        const REAL* __restrict__ HH, const REAL* __restrict__ tt, const int ntab,
                        REAL* __restrict__ H_out, REAL* __restrict__ T_out, REAL* __restrict__ e_in, REAL* __restrict__ e_out,
                        const int nx, const int ny, const int nz)
{
    CELL_IJK
    const bool m = flag[c] == INTERFACE || flag[c] == LIQUID;
    if (!m) { H_out[c] = H[c]; T_out[c] = T[c]; e_in[c] = 0.0; e_out[c] = 0.0; return; }
    const REAL mass = w[c] > (REAL)0.0 ? w[c] : (REAL)0.0;
    REAL sa = mass / eps_min;
    sa = sa < (REAL)0.0 ? (REAL)0.0 : (sa > (REAL)1.0 ? (REAL)1.0 : sa);
    const REAL Ta = T[c], ka = kc[c];
    REAL div = 0.0;
    const int idx[3] = {i, j, k};
    const int nn[3] = {nx, ny, nz};
    for (int ax = 0; ax < 3; ++ax) {
        for (int side = 0; side < 2; ++side) {              // +1 face (cell is "a"), then -1 face (cell is "b")
            const int d = side == 0 ? 1 : -1;
            const int p = idx[ax] + d;
            if (p < 0 || p >= nn[ax]) continue;                // the array version uses slices: no wrap
            int ii = i, jj = j, kk = k;
            if (ax == 0) ii = p; else if (ax == 1) jj = p; else kk = p;
            const long long s = ((long long)ii * ny + jj) * nz + kk;
            const signed char fs_ = flag[s];
            if (fs_ != INTERFACE && fs_ != LIQUID) continue;
            const REAL kb = kc[s];
            const REAL lo = ka < kb ? ka : kb, hi = ka < kb ? kb : ka;
            const REAL kf = (REAL)2.0 * lo * hi / (lo + hi);
            const REAL mb = w[s] > (REAL)0.0 ? w[s] : (REAL)0.0;
            REAL sb = mb / eps_min;
            sb = sb < (REAL)0.0 ? (REAL)0.0 : (sb > (REAL)1.0 ? (REAL)1.0 : sb);
            // flux into the lower cell of the face, as in the array version
            const long long lo_c = side == 0 ? c : s, hi_c = side == 0 ? s : c;
            REAL flux = kf * (T[hi_c] - T[lo_c]) / dx;
            if (advect) {
                const REAL uf = (REAL)0.5 * (u[ax * N + lo_c] + u[ax * N + hi_c]) * udx;
                flux = flux - uf * (uf > (REAL)0.0 ? H[lo_c] : H[hi_c]);
            }
            flux = flux * (sa < sb ? sa : sb);
            if (side == 0) div += flux / dx; else div -= flux / dx;
        }
    }
    const REAL cap = mass > eps_min ? mass : eps_min;
    const bool live = mass > (REAL)1e-9;
    const REAL qc = has_q ? q[c] : (REAL)0.0;
    const REAL lc = has_loss ? loss[c] : (REAL)0.0;
    REAL Hn = H[c];
    if (live) {
        const REAL dH = dt * div / (mass > (REAL)1e-9 ? mass : (REAL)1e-9) + dt * (qc - lc) / cap;
        Hn = Hn + dH;
    }
    H_out[c] = Hn;
    T_out[c] = interp_sorted(Hn, HH, tt, ntab);
    const REAL frac = live ? mass / cap : (REAL)0.0;
    e_in[c] = frac * qc;
    e_out[c] = frac * lc;
}

extern "C" __global__ void fs_nb_any(const unsigned char* __restrict__ m, unsigned char* __restrict__ out,
                                     const int nx, const int ny, const int nz)
{
    const long long N = (long long)nx * ny * nz;
    const long long c = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= N) return;
    const int k = (int)(c % nz);
    const int j = (int)((c / nz) % ny);
    const int i = (int)(c / ((long long)ny * nz));
    unsigned char r = 0;
    for (int q = 1; q < 19; ++q) {
        const long long s = ((long long)wrap(i + CX[q], nx) * ny + wrap(j + CY[q], ny)) * nz + wrap(k + CZ[q], nz);
        if (m[s]) { r = 1; break; }
    }
    out[c] = r;
}
"""

_MODULES = {}
BLOCK = 128


def _module(dtype):
    import cupy as cp
    dtype = np.dtype(dtype)
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"CUDA free-surface kernels need float64 or float32, not {dtype}")
    if dtype not in _MODULES:
        T = "double" if dtype == np.float64 else "float"
        names = tuple(f"{b}<{T}>" for b in ("fs_stream_collide", "fs_nb_sum", "fs_smooth_axis", "fs_normal",
                                             "fs_curvature", "fs_conductivity", "fs_heat"))
        mod = cp.RawModule(code=_SRC, options=("-std=c++14",), name_expressions=names)
        fns = {n.split("<")[0]: mod.get_function(n) for n in names}
        fns["fs_nb_any"] = mod.get_function("fs_nb_any")
        _MODULES[dtype] = fns
    return _MODULES[dtype]


def _grid(n):
    return ((int(n) + BLOCK - 1) // BLOCK,), (BLOCK,)


def stream_collide(lbm, flag, itype, rho_g, eps):
    """One fused stream-collide pass of ``lbm`` (a FreeSurfaceLBM on CuPy). Returns
    (f, rho, u, mass, E_in, M_in, M_out, n_limited); the carried-flux arrays are None when
    nothing is carried."""
    import cupy as cp
    dt = np.dtype(lbm.dtype)
    R = dt.type
    k = _module(dt)["fs_stream_collide"]
    sh = lbm.shape
    N = int(np.prod(sh))
    carry = lbm.h is not None
    f_out = cp.empty_like(lbm.f)
    rho, mass = cp.empty(sh, dt), cp.empty(sh, dt)
    u = cp.empty((3,) + sh, dt)
    if carry:
        E_in, M_in, M_out = cp.empty(sh, dt), cp.empty(sh, dt), cp.empty(sh, dt)
        h = cp.ascontiguousarray(lbm.h, dtype=dt)
    else:
        E_in = M_in = M_out = h = cp.empty(1, dt)
    solid = lbm.solid
    has_solid = solid is not None
    solid = cp.ascontiguousarray(solid, dtype=cp.uint8) if has_solid else cp.empty(1, cp.uint8)
    has_force = lbm.force is not None
    force = cp.ascontiguousarray(lbm.force, dtype=dt) if has_force else cp.empty(1, dt)
    damp = lbm.damping
    has_K = damp is not None and not np.isscalar(damp)
    Karr = cp.ascontiguousarray(damp, dtype=dt) if has_K else cp.empty(1, dt)
    Kc = 0.0 if (damp is None or has_K) else float(damp)
    lim = cp.zeros(1, cp.uint64)
    g = lbm.g
    args = (cp.ascontiguousarray(lbm.f, dtype=dt), cp.ascontiguousarray(flag, dtype=cp.int8),
            cp.ascontiguousarray(lbm.mass, dtype=dt), cp.ascontiguousarray(eps, dtype=dt),
            cp.ascontiguousarray(rho_g, dtype=dt), cp.ascontiguousarray(lbm.u, dtype=dt), h, solid, force, Karr,
            R(Kc), R(g[0]), R(g[1]), R(g[2]), R(lbm.w_plus), R(lbm.w_minus), R(float(lbm.u_limit or 0.0)),
            np.int32(carry), np.int32(has_solid), np.int32(has_force), np.int32(has_K),
            cp.ascontiguousarray(itype, dtype=cp.int8), R(lbm.tau), R(lbm.magic), R(float(lbm.smagorinsky)),
            f_out, rho, u, mass, E_in, M_in, M_out, lim, np.int32(sh[0]), np.int32(sh[1]), np.int32(sh[2]))
    k(*_grid(N), args)
    if not carry:
        E_in = M_in = M_out = None
    return f_out, rho, u, mass, E_in, M_in, M_out, int(lim.get()[0])


def nb_sums(arrays, dtype):
    """Sums over the 18 neighbours of each array (as ``np.roll`` to x + c)."""
    import cupy as cp
    dt = np.dtype(dtype)
    out = []
    for s in range(0, len(arrays), 8):
        chunk = arrays[s:s + 8]
        A = cp.ascontiguousarray(cp.stack([cp.asarray(a, dtype=dt) for a in chunk]))
        sh = A.shape[1:]
        O = cp.empty_like(A)
        _module(dt)["fs_nb_sum"](*_grid(np.prod(sh)), (A, O, np.int32(len(chunk)), np.int32(sh[0]),
                                                      np.int32(sh[1]), np.int32(sh[2])))
        out.extend(O[f] for f in range(len(chunk)))
    return out


def nb_any(mask, dtype=np.float64):
    """True where any of the 18 neighbours is True."""
    import cupy as cp
    m = cp.ascontiguousarray(mask, dtype=cp.uint8)
    sh = m.shape
    out = cp.empty(sh, cp.uint8)
    _module(dtype)["fs_nb_any"](*_grid(m.size), (m, out, np.int32(sh[0]), np.int32(sh[1]), np.int32(sh[2])))
    return out.view(cp.bool_)


def smoothed_fill(flag, eps, smooth, dtype):
    """The fill level smoothed ``smooth`` times with a [1 2 1] filter per axis (walls mirror)."""
    import cupy as cp
    dt = np.dtype(dtype)
    k = _module(dt)["fs_smooth_axis"]
    flag = cp.ascontiguousarray(flag, dtype=cp.int8)
    e = cp.ascontiguousarray(eps, dtype=dt)
    sh = e.shape
    g = _grid(e.size)
    for _ in range(smooth):
        for ax in range(3):
            out = cp.empty_like(e)
            k(*g, (e, flag, out, np.int32(ax), np.int32(sh[0]), np.int32(sh[1]), np.int32(sh[2])))
            e = out
    return e


def normal_curvature(flag, eps, smooth, dtype):
    """Unit normal (3, ...) and mean curvature of the smoothed fill, as
    ``FreeSurfaceLBM.normal_curvature``."""
    import cupy as cp
    dt = np.dtype(dtype)
    K = _module(dt)
    flag = cp.ascontiguousarray(flag, dtype=cp.int8)
    e = smoothed_fill(flag, eps, smooth, dt)
    sh = e.shape
    dims = (np.int32(sh[0]), np.int32(sh[1]), np.int32(sh[2]))
    g = _grid(e.size)
    n = cp.empty((3,) + sh, dt)
    K["fs_normal"](*g, (e, flag, n) + dims)
    kap = cp.empty(sh, dt)
    K["fs_curvature"](*g, (n, flag, kap) + dims)
    return n, kap


def heat_step(flag, T, H, weight, u, q, loss, advect, dx, dt, eps_min, tt, kT, HH, dtype):
    """One explicit enthalpy step of ``FreeSurfaceMeltPool._heat_step`` in two kernels.
    ``tt`` must be uniform. Returns (H, T, e_in, e_out) with the energies [J] as floats."""
    import cupy as cp
    d = np.dtype(dtype)
    R = d.type
    K = _module(d)
    sh = T.shape
    N = int(np.prod(sh))
    T = cp.ascontiguousarray(T, dtype=d)
    H = cp.ascontiguousarray(H, dtype=d)
    kc = cp.empty(sh, d)
    t0, dtab = float(tt[0]), float(tt[1] - tt[0])
    K["fs_conductivity"](*_grid(N), (T, kT, kc, R(t0), R(dtab), np.int32(len(tt)), np.int64(N)))
    has_q = q is not None and not np.isscalar(q)
    has_loss = loss is not None
    qa = cp.ascontiguousarray(q, dtype=d) if has_q else cp.empty(1, d)
    la = cp.ascontiguousarray(loss, dtype=d) if has_loss else cp.empty(1, d)
    ua = cp.ascontiguousarray(u, dtype=d) if advect else cp.empty(1, d)
    H_out, T_out, e_in, e_out = cp.empty(sh, d), cp.empty(sh, d), cp.empty(sh, d), cp.empty(sh, d)
    K["fs_heat"](*_grid(N), (T, H, kc, cp.ascontiguousarray(weight, dtype=d), cp.ascontiguousarray(flag, dtype=cp.int8),
                             ua, qa, la, np.int32(has_q), np.int32(has_loss), np.int32(advect), R(dx), R(dt), R(dx / dt),
                             R(eps_min), HH, tt, np.int32(len(tt)), H_out, T_out, e_in, e_out,
                             np.int32(sh[0]), np.int32(sh[1]), np.int32(sh[2])))
    vol = dx ** 3 * dt
    ein = float(e_in.sum()) * vol if has_q else 0.0
    eout = float(e_out.sum()) * vol if has_loss else 0.0
    return H_out, T_out, ein, eout
