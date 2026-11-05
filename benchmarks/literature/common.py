"""Shared helpers for the literature benchmarks."""
import json
import os

import numpy as np


def richardson(h, f):
    """Grid-convergence analysis from three solutions on grids h1 < h2 < h3 (any ratios).

    Solves for the observed order p (Celik et al. 2008, J. Fluids Eng. 130:078001) and
    returns the extrapolated value and GCI of the finest grid (safety factor 1.25)."""
    h1, h2, h3 = h
    f1, f2, f3 = f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    spread = float((max(f) - min(f)) / abs(f1))
    if e21 == 0 or e32 / e21 <= 0 or abs(e21) >= abs(e32):
        # oscillatory, or changes not shrinking with refinement: not in the
        # asymptotic range -> no extrapolation; report the finest value and
        # the spread of the three grids as the uncertainty
        return dict(p=float("nan"), f_ext=float(f1), gci=float("nan"), monotone=False, spread=spread)
    s = np.sign(e32 / e21)
    p = 1.0
    for _ in range(200):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r21**p - 1)
    return dict(p=float(p), f_ext=float(f_ext), gci=float(gci), monotone=True, spread=spread)


def power_fit(rho, y):
    """Gibson-Ashby fit y = C rho^n (least squares in log space)."""
    n, logC = np.polyfit(np.log(rho), np.log(y), 1)
    return float(np.exp(logC)), float(n)


class ResultStore:
    """Append-only JSON cache so long benchmark runs can be resumed."""

    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(self.data, open(self.path, "w"), indent=1)
