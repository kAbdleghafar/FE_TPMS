"""Host-side mesh container and topology utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Node orderings follow Abaqus/CalculiX (== VTK for these types).
CELL_INFO = {
    #           nodes  corners  meshio name   abaqus
    "hex8": dict(nn=8, nc=8, meshio="hexahedron", abaqus="C3D8"),
    "tet4": dict(nn=4, nc=4, meshio="tetra", abaqus="C3D4"),
    "tet10": dict(nn=10, nc=4, meshio="tetra10", abaqus="C3D10"),
}
MESHIO_TO_TYPE = {v["meshio"]: k for k, v in CELL_INFO.items()}

# Boundary faces (local node ids). Orientation is fixed afterwards so that the
# normal points out of the element.
FACES = {
    "hex8": np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]),
    "tet4": np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]),
    "tet10": np.array([[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7],
                       [1, 2, 3, 5, 9, 8], [0, 2, 3, 6, 9, 7]]),
}
FACE_TYPE = {"hex8": "quad4", "tet4": "tri3", "tet10": "tri6"}
FACE_FLIP = {"quad4": [0, 3, 2, 1], "tri3": [0, 2, 1], "tri6": [0, 2, 1, 5, 4, 3]}
FACE_CORNERS = {"quad4": 4, "tri3": 3, "tri6": 3}


@dataclass
class Mesh:
    points: np.ndarray                      # (n_nodes, 3) float64
    cells: np.ndarray                       # (n_cells, nodes_per_cell) int64
    cell_type: str                          # 'hex8' | 'tet4' | 'tet10'
    cell_data: dict = field(default_factory=dict)   # e.g. {'material': int array}
    point_sets: dict = field(default_factory=dict)
    cell_sets: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)        # e.g. {'voxel_size': h}

    def __post_init__(self):
        self.points = np.ascontiguousarray(self.points, dtype=np.float64)
        self.cells = np.ascontiguousarray(self.cells, dtype=np.int64)
        if self.cell_type not in CELL_INFO:
            raise ValueError(f"unsupported cell type {self.cell_type}")
        if self.cells.shape[1] != CELL_INFO[self.cell_type]["nn"]:
            raise ValueError("cells array does not match cell type")
        self.cell_data.setdefault("material", np.zeros(len(self.cells), dtype=np.int64))

    # basic properties -------------------------------------------------------
    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def is_voxel(self) -> bool:
        return self.cell_type == "hex8" and "voxel_size" in self.meta

    def centroids(self) -> np.ndarray:
        nc = CELL_INFO[self.cell_type]["nc"]
        return self.points[self.cells[:, :nc]].mean(axis=1)

    def bounds(self):
        return self.points.min(axis=0), self.points.max(axis=0)

    def summary(self) -> str:
        lo, hi = self.bounds()
        s = (f"Mesh[{self.cell_type}] {self.n_cells:,} cells, {self.n_points:,} nodes, "
             f"{3 * self.n_points:,} dofs, bbox {np.round(lo, 3)} .. {np.round(hi, 3)}")
        if self.is_voxel:
            s += f", voxel h={self.meta['voxel_size']:g}"
        return s

    # selections -----------------------------------------------------------------
    def nodes_where(self, pred: Callable) -> np.ndarray:
        """Node ids where ``pred(x, y, z)`` is True (only nodes used by cells)."""
        used = np.zeros(self.n_points, bool)
        used[self.cells.ravel()] = True
        x, y, z = self.points.T
        return np.flatnonzero(np.asarray(pred(x, y, z), bool) & used)

    def cells_where(self, pred: Callable) -> np.ndarray:
        c = self.centroids()
        return np.flatnonzero(np.asarray(pred(c[:, 0], c[:, 1], c[:, 2]), bool))

    def closest_node(self, p) -> int:
        used = np.unique(self.cells)
        d = np.linalg.norm(self.points[used] - np.asarray(p, float), axis=1)
        return int(used[np.argmin(d)])

    # topology -------------------------------------------------------------------
    def boundary_faces(self):
        """Exterior faces.

        Returns ``(faces, face_type, owner)`` where ``faces`` is (n_faces, n)
        with outward orientation and ``owner`` is the owning cell id.
        """
        lf = FACES[self.cell_type]
        ftype = FACE_TYPE[self.cell_type]
        nfc = FACE_CORNERS[ftype]
        all_faces = self.cells[:, lf].reshape(-1, lf.shape[1])
        owner = np.repeat(np.arange(self.n_cells), lf.shape[0])
        key = np.sort(all_faces[:, :nfc], axis=1)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        bnd = counts[inv] == 1
        faces, owner = all_faces[bnd], owner[bnd]
        # orient outward
        fc = self.points[faces[:, :nfc]].mean(axis=1)
        cc = self.centroids()[owner]
        p = self.points
        n = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, nfc - 1]] - p[faces[:, 0]])
        flip = np.einsum("ij,ij->i", n, fc - cc) < 0
        faces[flip] = faces[flip][:, FACE_FLIP[ftype]]
        return faces, ftype, owner

    def boundary_nodes(self) -> np.ndarray:
        faces, _, _ = self.boundary_faces()
        return np.unique(faces)

    def volume(self) -> float:
        from ..fem.elements import element_volumes

        return float(element_volumes(self).sum())

    # editing ----------------------------------------------------------------------
    def subset(self, cell_ids) -> "Mesh":
        """Mesh made of the given cells, with unused nodes removed."""
        cell_ids = np.asarray(cell_ids)
        cells = self.cells[cell_ids]
        used, inv = np.unique(cells, return_inverse=True)
        cd = {k: np.asarray(v)[cell_ids] for k, v in self.cell_data.items()}
        return Mesh(self.points[used], inv.reshape(cells.shape), self.cell_type, cd,
                    meta=dict(self.meta))

    def reorder_cells(self, order) -> "Mesh":
        order = np.asarray(order)
        cd = {k: np.asarray(v)[order] for k, v in self.cell_data.items()}
        return Mesh(self.points, self.cells[order], self.cell_type, cd,
                    dict(self.point_sets), {}, dict(self.meta))

    # I/O ------------------------------------------------------------------------
    def to_meshio(self, point_data: Optional[dict] = None, cell_data: Optional[dict] = None):
        import meshio

        cd = {k: [np.asarray(v)] for k, v in self.cell_data.items()}
        for k, v in (cell_data or {}).items():
            cd[k] = [np.asarray(v)]
        pd = {k: np.asarray(v) for k, v in (point_data or {}).items()}
        return meshio.Mesh(self.points, [(CELL_INFO[self.cell_type]["meshio"], self.cells)],
                           point_data=pd, cell_data=cd)

    def save(self, path: str, point_data: Optional[dict] = None, cell_data: Optional[dict] = None):
        """Write any meshio format (.vtu recommended for ParaView)."""
        self.to_meshio(point_data, cell_data).write(path)

    @classmethod
    def from_meshio(cls, m, cell_type: Optional[str] = None) -> "Mesh":
        blocks = {c.type: c.data for c in m.cells}
        if cell_type is None:
            for ct in ("tetra10", "hexahedron", "tetra"):
                if ct in blocks:
                    cell_type = MESHIO_TO_TYPE[ct]
                    break
            else:
                raise ValueError(f"no supported volume cells in {list(blocks)}")
        mname = CELL_INFO[cell_type]["meshio"]
        cells = np.concatenate([c.data for c in m.cells if c.type == mname])
        used, inv = np.unique(cells, return_inverse=True)
        mesh = cls(np.asarray(m.points[used], float)[:, :3], inv.reshape(cells.shape), cell_type)
        return mesh


def read_mesh(path: str, cell_type: Optional[str] = None) -> Mesh:
    """Read any volume mesh meshio understands (Gmsh .msh, Abaqus .inp, VTK,
    MED, Nastran, ...). Only one volume cell type is kept."""
    import meshio

    return Mesh.from_meshio(meshio.read(path), cell_type)


def check_tet10_midnodes(mesh: Mesh, tol: float = 1e-6) -> float:
    """Max relative distance of tet10 mid-side nodes from straight-edge midpoints.

    Large values usually mean a node-ordering mismatch between tools.
    """
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    p = mesh.points[mesh.cells]
    err = 0.0
    h = np.linalg.norm(p[:, 1] - p[:, 0], axis=1).mean()
    for k, (a, b) in enumerate(edges):
        mid = 0.5 * (p[:, a] + p[:, b])
        err = max(err, float(np.linalg.norm(p[:, 4 + k] - mid, axis=1).max() / h))
    return err
