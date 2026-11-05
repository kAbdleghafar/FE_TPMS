"""Vectorised small-strain constitutive updates (backend agnostic).

Symmetric tensors are stored as 6-vectors ``[xx, yy, zz, yz, xz, xy]`` with
*tensor* (not engineering) shear components; the double contraction is
``a:b = a0 b0 + a1 b1 + a2 b2 + 2 (a3 b3 + a4 b4 + a5 b5)``.

Material model: isotropic linear elasticity + J2 (von Mises) plasticity with
linear isotropic hardening, radial-return mapping and the consistent
(algorithmic) tangent of Simo & Hughes (1998), Box 3.2.
"""
from __future__ import annotations

import numpy as np

from ..backend import array_module

SQ32 = np.sqrt(1.5)
NO_YIELD = 1.0e30


def sym_from_grad(G):
    """(..., 3, 3) displacement gradient -> (..., 6) small strain."""
    xp = array_module(G)
    return xp.stack([G[..., 0, 0], G[..., 1, 1], G[..., 2, 2],
                     0.5 * (G[..., 1, 2] + G[..., 2, 1]),
                     0.5 * (G[..., 0, 2] + G[..., 2, 0]),
                     0.5 * (G[..., 0, 1] + G[..., 1, 0])], axis=-1)


def full_from_sym(s):
    """(..., 6) -> (..., 3, 3)."""
    xp = array_module(s)
    out = xp.empty(s.shape[:-1] + (3, 3), dtype=s.dtype)
    out[..., 0, 0] = s[..., 0]
    out[..., 1, 1] = s[..., 1]
    out[..., 2, 2] = s[..., 2]
    out[..., 1, 2] = out[..., 2, 1] = s[..., 3]
    out[..., 0, 2] = out[..., 2, 0] = s[..., 4]
    out[..., 0, 1] = out[..., 1, 0] = s[..., 5]
    return out


def trace(s):
    return s[..., 0] + s[..., 1] + s[..., 2]


def dev(s):
    out = s.copy()
    m = trace(s) / 3.0
    out[..., 0] -= m
    out[..., 1] -= m
    out[..., 2] -= m
    return out


def ddot(a, b):
    return (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]
            + 2.0 * (a[..., 3] * b[..., 3] + a[..., 4] * b[..., 4] + a[..., 5] * b[..., 5]))


def add_iso(s, v):
    """s + v * I (returns new array)."""
    out = s.copy()
    out[..., 0] += v
    out[..., 1] += v
    out[..., 2] += v
    return out


def von_mises(s):
    xp = array_module(s)
    d = dev(s)
    return xp.sqrt(1.5 * ddot(d, d))


def elastic_stress(eps, lam, mu):
    """sigma = lam tr(eps) I + 2 mu eps; lam/mu broadcast against eps[..., 0]."""
    return add_iso(2.0 * mu[..., None] * eps, lam * trace(eps))


def j2_return(eps_e_trial, lam, mu, sy0, H, alpha_n, eps_p_n):
    """Radial return for J2 plasticity with linear isotropic hardening.

    Parameters broadcast as ``(nc, 1)`` against ``(nc, nq)``.

    Returns ``sig, eps_p, alpha, theta, theta_bar, nrm, plastic`` where
    ``theta, theta_bar, nrm`` define the consistent tangent::

        C_ep = kappa I(x)I + 2 mu theta I_dev - 2 mu theta_bar n (x) n
    """
    xp = array_module(eps_e_trial)
    sig_tr = elastic_stress(eps_e_trial, lam, mu)
    s = dev(sig_tr)
    snorm = xp.sqrt(xp.maximum(ddot(s, s), 0.0))
    q = SQ32 * snorm
    f = q - (sy0 + H * alpha_n)
    plastic = f > 1e-12 * sy0
    denom = 3.0 * mu + H
    dl = xp.where(plastic, f / denom, 0.0)                  # equivalent plastic strain increment
    nrm = s / xp.maximum(snorm, 1e-300)[..., None]
    deps_p = (SQ32 * dl)[..., None] * nrm
    sig = sig_tr - (2.0 * mu)[..., None] * deps_p
    eps_p = eps_p_n + deps_p
    alpha = alpha_n + dl
    qs = xp.maximum(q, 1e-300)
    theta = xp.where(plastic, 1.0 - 3.0 * mu * dl / qs, 1.0)
    theta_bar = xp.where(plastic, 3.0 * mu / denom - 3.0 * mu * dl / qs, 0.0)
    return sig, eps_p, alpha, theta, theta_bar, nrm, plastic


def tangent_stress(deps, lam, mu, theta=None, theta_bar=None, nrm=None):
    """d sigma = C_ep : d eps (elastic if theta is None)."""
    if theta is None:
        return elastic_stress(deps, lam, mu)
    kappa = lam + 2.0 * mu / 3.0
    out = (2.0 * mu * theta)[..., None] * dev(deps)
    out = add_iso(out, kappa * trace(deps))
    out -= (2.0 * mu * theta_bar * ddot(nrm, deps))[..., None] * nrm
    return out


def principal(s6):
    """Principal values (ascending) of host 6-vectors, shape (..., 3)."""
    s6 = np.asarray(s6)
    return np.linalg.eigvalsh(full_from_sym(s6))
