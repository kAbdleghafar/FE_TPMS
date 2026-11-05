"""Reference elements, quadrature and geometric precomputation.

Supported volume elements (Abaqus equivalents in brackets):

* ``hex8``  -- trilinear brick, 2x2x2 Gauss [C3D8 in CalculiX]; with
  ``bbar=True`` the mean-dilatation (selectively reduced) variant [C3D8 in
  Abaqus/Standard], recommended for plasticity (no volumetric locking).
* ``tet4``  -- linear tet, 1 point [C3D4]
* ``tet10`` -- quadratic tet, 4 points [C3D10]
"""
from __future__ import annotations

import numpy as np

G2 = 1.0 / np.sqrt(3.0)
HEX_NAT = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


# ----------------------------------------------------------------- hex8
def hex8_N(xi):
    xi = np.atleast_2d(xi)
    return 0.125 * np.prod(1 + xi[:, None, :] * HEX_NAT[None, :, :], axis=2)


def hex8_dN(xi):
    xi = np.atleast_2d(xi)
    t = 1 + xi[:, None, :] * HEX_NAT[None, :, :]          # (nq, 8, 3)
    d = np.empty_like(t)
    d[..., 0] = HEX_NAT[None, :, 0] * t[..., 1] * t[..., 2]
    d[..., 1] = HEX_NAT[None, :, 1] * t[..., 0] * t[..., 2]
    d[..., 2] = HEX_NAT[None, :, 2] * t[..., 0] * t[..., 1]
    return 0.125 * d


def hex8_quad():
    return HEX_NAT * G2, np.ones(8)


# ----------------------------------------------------------------- tet4 / tet10
def _bary(xi):
    xi = np.atleast_2d(xi)
    return np.column_stack([1 - xi.sum(axis=1), xi])        # L0..L3


DBARY = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)  # dL/dxi
TET10_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])


def tet4_N(xi):
    return _bary(xi)


def tet4_dN(xi):
    return np.broadcast_to(DBARY, (np.atleast_2d(xi).shape[0], 4, 3)).copy()


def tet10_N(xi):
    L = _bary(xi)
    N = np.empty((L.shape[0], 10))
    N[:, :4] = L * (2 * L - 1)
    for k, (a, b) in enumerate(TET10_EDGES):
        N[:, 4 + k] = 4 * L[:, a] * L[:, b]
    return N


def tet10_dN(xi):
    L = _bary(xi)
    d = np.empty((L.shape[0], 10, 3))
    for a in range(4):
        d[:, a, :] = (4 * L[:, a] - 1)[:, None] * DBARY[a][None, :]
    for k, (a, b) in enumerate(TET10_EDGES):
        d[:, 4 + k, :] = 4 * (L[:, a][:, None] * DBARY[b][None, :] + L[:, b][:, None] * DBARY[a][None, :])
    return d


def tet_quad(n: int):
    if n == 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])
    if n == 4:
        a, b = 0.5854101966249685, 0.1381966011250105
        pts = np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
        return pts, np.full(4, 1.0 / 24.0)
    raise ValueError(n)


ELEMENTS = {
    "hex8": dict(N=hex8_N, dN=hex8_dN, quad=hex8_quad),
    "tet4": dict(N=tet4_N, dN=tet4_dN, quad=lambda: tet_quad(1)),
    "tet10": dict(N=tet10_N, dN=tet10_dN, quad=lambda: tet_quad(4)),
}


def reference(cell_type: str):
    """(qp natural coords, weights, N at qps (nq,nn), dN/dxi at qps (nq,nn,3))."""
    e = ELEMENTS[cell_type]
    xi, w = e["quad"]()
    return xi, w, e["N"](xi), e["dN"](xi)


def element_geometry(points, cells, cell_type: str, chunk: int = 200_000):
    """Physical shape-function gradients and integration weights.

    Returns ``dNdx`` (ne, nq, nn, 3) and ``wdet`` (ne, nq). Raises on inverted
    or degenerate elements.
    """
    xi, w, N, dNdxi = reference(cell_type)
    ne, nn = cells.shape
    nq = len(w)
    dNdx = np.empty((ne, nq, nn, 3))
    wdet = np.empty((ne, nq))
    for s in range(0, ne, chunk):
        X = points[cells[s:s + chunk]]                        # (nc, nn, 3)
        J = np.einsum("cai,qaj->cqij", X, dNdxi)              # dx_i/dxi_j
        det = np.linalg.det(J)
        if np.any(det <= 0):
            bad = s + np.flatnonzero((det <= 0).any(axis=1))[:5]
            raise ValueError(f"non-positive Jacobian in elements {bad.tolist()} (inverted/degenerate)")
        Jinv = np.linalg.inv(J)
        dNdx[s:s + chunk] = np.einsum("qaj,cqji->cqai", dNdxi, Jinv)
        wdet[s:s + chunk] = det * w[None, :]
    return dNdx, wdet


def voxel_geometry(h: float):
    """Shared dN/dx (nq, 8, 3) and weights (nq,) of an axis-aligned cube."""
    xi, w, N, dNdxi = reference("hex8")
    return dNdxi * (2.0 / h), w * (h / 2.0) ** 3


def element_volumes(mesh) -> np.ndarray:
    if mesh.is_voxel:
        return np.full(mesh.n_cells, mesh.meta["voxel_size"] ** 3)
    _, wdet = element_geometry(mesh.points, mesh.cells, mesh.cell_type)
    return wdet.sum(axis=1)


def extrapolation_matrix(cell_type: str) -> np.ndarray:
    """Matrix E (nn, nq) mapping integration-point values to nodes."""
    xi, w, N, _ = reference(cell_type)
    if cell_type == "hex8":
        return np.linalg.inv(N)
    if cell_type == "tet4":
        return np.ones((4, 1))
    if cell_type == "tet10":
        Lq = _bary(xi)                       # (4 qp, 4 corners): qp = Lq @ corner
        Ec = np.linalg.inv(Lq)               # corners from qps
        E = np.zeros((10, 4))
        E[:4] = Ec
        for k, (a, b) in enumerate(TET10_EDGES):
            E[4 + k] = 0.5 * (Ec[a] + Ec[b])
        return E
    raise ValueError(cell_type)


# ----------------------------------------------------------------- faces
def face_reference(face_type: str):
    """(N (nq, nfn), dN/dxi (nq, nfn, 2), weights) on the reference face."""
    if face_type == "quad4":
        g = np.array([-G2, G2])
        xi = np.array([[a, b] for b in g for a in g])
        nat = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
        t = 1 + xi[:, None, :] * nat[None]
        N = 0.25 * t[..., 0] * t[..., 1]
        dN = np.stack([0.25 * nat[None, :, 0] * t[..., 1], 0.25 * nat[None, :, 1] * t[..., 0]], -1)
        return N, dN, np.ones(4)
    if face_type == "tri3":
        xi = np.array([[1 / 3, 1 / 3]])
        N = np.array([[1 / 3, 1 / 3, 1 / 3]])
        dN = np.array([[[-1, -1], [1, 0], [0, 1]]], float)
        return N, dN, np.array([0.5])
    if face_type == "tri6":
        xi = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        r, s = xi[:, 0], xi[:, 1]
        L = np.column_stack([1 - r - s, r, s])
        dL = np.array([[-1, -1], [1, 0], [0, 1]], float)
        N = np.column_stack([L * (2 * L - 1), 4 * L[:, 0] * L[:, 1], 4 * L[:, 1] * L[:, 2], 4 * L[:, 2] * L[:, 0]])
        dN = np.zeros((3, 6, 2))
        for a in range(3):
            dN[:, a] = (4 * L[:, a] - 1)[:, None] * dL[a]
        for k, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)]):
            dN[:, 3 + k] = 4 * (L[:, a][:, None] * dL[b] + L[:, b][:, None] * dL[a])
        return N, dN, np.full(3, 1 / 6)
    raise ValueError(face_type)


def face_integrals(points, faces, face_type: str):
    """Per face: N at qps, area-weighted outward normal vectors ``n dA`` (nf, nq, 3)."""
    N, dN, w = face_reference(face_type)
    X = points[faces]                                         # (nf, nfn, 3)
    t1 = np.einsum("fai,qa->fqi", X, dN[..., 0])
    t2 = np.einsum("fai,qa->fqi", X, dN[..., 1])
    ndA = np.cross(t1, t2) * w[None, :, None]
    return N, ndA
