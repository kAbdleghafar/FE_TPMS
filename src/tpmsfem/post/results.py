"""Post-processing: derived stress measures, nodal averaging, VTU export."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..fem import constitutive as C
from ..fem.elements import extrapolation_matrix, reference


def element_average(model, qp_values: np.ndarray) -> np.ndarray:
    """Volume average over integration points: (ne, nq, k) -> (ne, k)."""
    w = np.asarray(model.wq.get() if hasattr(model.wq, "get") else model.wq)
    w = np.broadcast_to(w if w.ndim == 2 else w[None, :], qp_values.shape[:2])
    return (qp_values * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None]


def nodal_average(mesh, qp_values: np.ndarray, cells: Optional[np.ndarray] = None,
                  extrapolate: bool = True) -> np.ndarray:
    """Extrapolate integration-point values to nodes and average over the
    elements sharing each node (the usual "averaged nodal" output of commercial
    codes). ``cells`` restricts the averaging (e.g. one material/section)."""
    qp_values = np.asarray(qp_values)
    if qp_values.ndim == 2:
        qp_values = qp_values[..., None]
    ne, nq, k = qp_values.shape
    if extrapolate:
        E = extrapolation_matrix(mesh.cell_type)                   # (nn, nq)
        nodal_e = np.einsum("aq,eqk->eak", E, qp_values)
    else:
        nodal_e = np.repeat(qp_values.mean(axis=1, keepdims=True), mesh.cells.shape[1], axis=1)
    conn = mesh.cells
    if cells is not None:
        conn, nodal_e = conn[cells], nodal_e[cells]
    out = np.zeros((mesh.n_points, k))
    cnt = np.bincount(conn.ravel(), minlength=mesh.n_points).astype(float)
    for j in range(k):
        out[:, j] = np.bincount(conn.ravel(), weights=nodal_e[..., j].ravel(), minlength=mesh.n_points)
    with np.errstate(invalid="ignore", divide="ignore"):
        out /= cnt[:, None]
    out[cnt == 0] = np.nan
    return out


def von_mises(s6: np.ndarray) -> np.ndarray:
    return np.asarray(C.von_mises(np.asarray(s6)))


def principal(s6: np.ndarray) -> np.ndarray:
    return C.principal(s6)


def to_engineering_voigt_abaqus(s6: np.ndarray) -> np.ndarray:
    """[xx,yy,zz,yz,xz,xy] -> Abaqus order [11,22,33,12,13,23] (stress)."""
    return np.asarray(s6)[..., [0, 1, 2, 5, 4, 3]]


def result_fields(model) -> dict:
    """Standard set of host result arrays."""
    mesh = model.mesh
    sig_qp = model.stress()
    act = model.active_host
    s_el = element_average(model, sig_qp)
    s_nod = nodal_average(mesh, sig_qp, cells=np.flatnonzero(act))
    out = dict(
        U=model.displacement(),
        S_nodal=s_nod,
        vonMises_nodal=von_mises(s_nod),
        S_elem=s_el,
        vonMises_elem=von_mises(s_el),
        S_principal_elem=principal(s_el),
        active=act.astype(np.int8),
        RF=model.reactions(),
    )
    if model.plastic:
        ep, alpha = model.plastic_strain()
        out["PEEQ_elem"] = element_average(model, alpha[..., None])[:, 0]
    return out


def save_vtu(model, path: str, extra_point: Optional[dict] = None, extra_cell: Optional[dict] = None):
    """Write displacement/stress fields to a VTU file for ParaView."""
    r = result_fields(model)
    pd = dict(U=r["U"], S=r["S_nodal"], vonMises=r["vonMises_nodal"], RF=r["RF"])
    cd = dict(S=r["S_elem"], vonMises=r["vonMises_elem"], S_principal=r["S_principal_elem"],
              active=r["active"])
    if "PEEQ_elem" in r:
        cd["PEEQ"] = r["PEEQ_elem"]
    pd.update(extra_point or {})
    cd.update(extra_cell or {})
    cd = {k: v for k, v in cd.items()}
    model.mesh.save(path, point_data=pd, cell_data=cd)
    return r


def summarize(values: np.ndarray, label: str = "", unit: str = "MPa") -> str:
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return (f"{label}: max {v.max():.4g} {unit}, p99 {np.percentile(v, 99):.4g}, "
            f"p95 {np.percentile(v, 95):.4g}, mean {v.mean():.4g}")
