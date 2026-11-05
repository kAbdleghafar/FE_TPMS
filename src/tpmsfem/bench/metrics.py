"""Comparison metrics for verification and code-to-code benchmarks."""
from __future__ import annotations

from typing import Optional

import numpy as np


def rel_l2(a, b, mask: Optional[np.ndarray] = None) -> float:
    """||a - b|| / ||b|| over (optionally masked) rows, ignoring NaNs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mask is not None:
        a, b = a[mask], b[mask]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.linalg.norm((a - b)[ok]) / max(np.linalg.norm(b[ok]), 1e-300))


def rel_max(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a - b)[ok].max() / max(np.abs(b[ok]).max(), 1e-300))


def percentile_diff(a, b, q=(50, 95, 99, 100)) -> dict:
    """Relative difference of percentiles of two scalar fields (e.g. von Mises).
    Robust against single-node singularities, which dominate max values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return {f"p{p}": float((np.percentile(a, p) - np.percentile(b, p)) / np.percentile(b, p)) for p in q}


def gci(f1: float, f2: float, f3: float, r: float, Fs: float = 1.25) -> dict:
    """Roache's Grid Convergence Index from three systematically refined meshes.

    f1 fine, f2 medium, f3 coarse; ``r`` = constant refinement ratio (h2/h1).
    Returns observed order ``p``, Richardson extrapolate and GCI (fine) [fraction].
    """
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 / e21 <= 0:
        return dict(p=float("nan"), f_exact=float("nan"), gci_fine=float("nan"),
                    note="oscillatory or converged; GCI undefined")
    p = np.log(abs(e32 / e21)) / np.log(r)
    f_ex = f1 + (f1 - f2) / (r**p - 1)
    return dict(p=float(p), f_exact=float(f_ex), gci_fine=float(Fs * abs(e21 / f1) / (r**p - 1)))


def compare_fields(ours: dict, ref: dict, keys=("U", "S_nodal", "vonMises_nodal")) -> dict:
    out = {}
    for k in keys:
        if k in ours and k in ref:
            out[k] = dict(rel_l2=rel_l2(ours[k], ref[k]), rel_max=rel_max(ours[k], ref[k]))
    return out
