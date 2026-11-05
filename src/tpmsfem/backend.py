"""Array backend selection.

All heavy numerics in tpmsfem are written against a NumPy-compatible array
module ``xp``. On a CUDA machine with CuPy installed ``xp`` is ``cupy`` and
everything (TPMS field evaluation, element kernels, the Krylov solver) runs on
the GPU; otherwise it falls back to NumPy on the CPU.

Select the backend with ``tpmsfem.set_backend("cupy" | "numpy" | "auto")`` or
the environment variable ``TPMSFEM_BACKEND``. Mesh objects always live on the
host (NumPy); only the solver data is moved to the device.
"""
from __future__ import annotations

import os

import numpy as np

_state = {"name": "numpy", "xp": np}


def _cupy_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def set_backend(name: str = "auto") -> str:
    """Select the array backend. Returns the name actually selected."""
    name = (name or "auto").lower()
    if name == "auto":
        name = "cupy" if _cupy_available() else "numpy"
    if name == "cupy":
        import cupy as cp

        _state.update(name="cupy", xp=cp)
    elif name == "numpy":
        _state.update(name="numpy", xp=np)
    else:
        raise ValueError(f"unknown backend {name!r}; use 'numpy', 'cupy' or 'auto'")
    return name


def get_xp():
    """The active array module (numpy or cupy)."""
    return _state["xp"]


def backend_name() -> str:
    return _state["name"]


def on_gpu() -> bool:
    return _state["name"] == "cupy"


def array_module(a):
    """Array module that owns ``a`` (works for numpy and cupy arrays)."""
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def asnumpy(a):
    """Copy any array to host NumPy."""
    if a is None:
        return None
    if hasattr(a, "get") and type(a).__module__.split(".")[0] == "cupy":
        return a.get()
    return np.asarray(a)


def to_device(a, dtype=None):
    """Copy a host array to the active backend."""
    if a is None:
        return None
    return get_xp().asarray(a, dtype=dtype)


def sparse_module():
    """scipy.sparse or cupyx.scipy.sparse, matching the backend."""
    if on_gpu():
        import cupyx.scipy.sparse as sp

        return sp
    import scipy.sparse as sp

    return sp


def synchronize():
    """Block until queued GPU work is done (no-op on CPU). Use for timing."""
    if on_gpu():
        import cupy as cp

        cp.cuda.Device().synchronize()


def scatter_add(n: int, idx, vals):
    """Return ``out`` of length ``n`` with ``out[idx[k]] += vals[k]``.

    ``bincount`` with weights is a fast atomic scatter on both NumPy and CuPy.
    """
    xp = array_module(vals)
    return xp.bincount(idx.ravel(), weights=vals.ravel(), minlength=n)


def device_info() -> str:
    if on_gpu():
        import cupy as cp

        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        free, total = cp.cuda.runtime.memGetInfo()
        return f"cupy {cp.__version__} on {props['name'].decode()} ({total / 2**30:.1f} GiB)"
    import platform

    return f"numpy {np.__version__} on CPU ({platform.processor() or platform.machine()})"


set_backend(os.environ.get("TPMSFEM_BACKEND", "auto"))
