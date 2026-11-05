"""Geometry, meshing, solver-consistency, export and cross-code tests."""
import shutil

import numpy as np
import pytest

import tpmsfem as tf
from tpmsfem.geometry import TPMS, box, lattice_cube, sphere
from tpmsfem.mesh import voxelize

TI = tf.Material("ti", 110_000.0, 0.32)


def _lattice_mesh(h=0.2, size=3.0):
    lat = TPMS("gyroid", cell_size=size, kind="sheet", density=0.35)
    return voxelize(lattice_cube(lat, size, plate=0.4), (0, 0, 0), (size,) * 3, h)


def test_tpms_density_calibration():
    for surf, kind in [("gyroid", "sheet"), ("diamond", "network"), ("schwarz_p", "sheet")]:
        lat = TPMS(surf, cell_size=2.0, kind=kind, density=0.3)
        n = 64
        s = (np.arange(n) + 0.5) / n * 2.0
        X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
        assert (lat(X, Y, Z) < 0).mean() == pytest.approx(0.3, abs=0.01)


def test_graded_density():
    lat = TPMS("gyroid", cell_size=2.0, kind="sheet", density=lambda x, y, z: 0.2 + 0.2 * (z / 10.0))
    s = (np.arange(40) + 0.5) / 40 * 2.0
    X, Y = np.meshgrid(s, s, indexing="ij")
    lo = np.mean([(lat(X, Y, np.full_like(X, z)) < 0).mean() for z in np.linspace(0, 2, 40)])
    hi = np.mean([(lat(X, Y, np.full_like(X, z)) < 0).mean() for z in np.linspace(8, 10, 40)])
    assert lo == pytest.approx(0.22, abs=0.03) and hi == pytest.approx(0.38, abs=0.03)


def test_voxel_mesh_is_connected_and_valid():
    m = _lattice_mesh()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # face-connectivity graph of voxels must be a single component
    faces, _, _ = m.boundary_faces()
    assert m.n_cells > 1000
    e = m.cells[:, [0, 6]]
    A = coo_matrix((np.ones(m.n_cells), (e[:, 0], e[:, 1])), shape=(m.n_points,) * 2)
    assert m.volume() == pytest.approx(m.n_cells * 0.2**3)


def test_solvers_agree_and_operator_symmetric():
    m = _lattice_mesh(h=0.25)
    us = []
    for s in ("direct", "pcg", "amg"):
        md = tf.SolidModel(m, TI)
        z = m.points[:, 2]
        md.fix(np.flatnonzero(z < 1e-9))
        md.fix(np.flatnonzero(z > 3 - 1e-9), (2,), -0.003)
        md.solve(solver=s)
        us.append(md.displacement())
    assert np.abs(us[1] - us[0]).max() < 1e-6 * np.abs(us[0]).max()
    assert np.abs(us[2] - us[0]).max() < 1e-6 * np.abs(us[0]).max()
    # matrix-free operator == assembled matrix, and symmetric
    rng = np.random.default_rng(1)
    v = rng.standard_normal(md.ndof)
    K = md.assemble()
    assert np.allclose(md.tangent(v), K @ v, rtol=1e-10, atol=1e-8)
    assert abs((K - K.T)).max() < 1e-8 * abs(K).max()
    assert np.allclose(md.diagonal(), K.diagonal())


def test_non_voxel_path_matches_voxel_path():
    m = _lattice_mesh(h=0.3)
    generic = tf.Mesh(m.points, m.cells, "hex8")   # no voxel flag -> per-element gradients
    out = []
    for mesh in (m, generic):
        md = tf.SolidModel(mesh, TI)
        md.fix(mesh.nodes_where(lambda x, y, z: z < 1e-9))
        md.add_nodal_force(mesh.nodes_where(lambda x, y, z: z > 3 - 1e-9), (1.0, 0.0, -2.0))
        md.solve(solver="direct")
        out.append(md.displacement())
    assert np.allclose(out[0], out[1], rtol=1e-9, atol=1e-14)


def test_surface_and_tet_meshers():
    pytest.importorskip("skimage")
    from tpmsfem.mesh.adapters import gmsh_tetmesh, tet_quality, tetgen_tetmesh
    from tpmsfem.mesh.surface import implicit_to_surface, surface_area_volume

    v, f = implicit_to_surface(sphere(radius=1.0), (-1.2,) * 3, (1.2,) * 3, 0.08)
    area, vol = surface_area_volume(v, f)
    assert vol == pytest.approx(4 / 3 * np.pi, rel=0.02)
    for mesher in (tetgen_tetmesh, gmsh_tetmesh):
        try:
            m = mesher(v, f)
        except ImportError:
            continue
        assert m.volume() == pytest.approx(vol, rel=1e-4)
        assert tet_quality(m)["min"] > 0


def test_abaqus_export_roundtrip(tmp_path):
    import meshio

    from tpmsfem.io.abaqus import write_inp

    m = voxelize(box((0, 0, 0), (2, 1, 1)), (0, 0, 0), (2, 1, 1), 0.5)
    md = tf.SolidModel(m, TI)
    md.fix(m.nodes_where(lambda x, y, z: x < 1e-9))
    md.add_nodal_force(m.nodes_where(lambda x, y, z: x > 2 - 1e-9), (0, 0, -1))
    p = write_inp(md, str(tmp_path / "a.inp"))
    mm = meshio.read(p)
    assert np.allclose(mm.points, m.points)
    assert np.array_equal(mm.cells_dict["hexahedron"], m.cells)


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")
def test_cross_code_calculix_lattice(tmp_path):
    from tpmsfem.bench.metrics import rel_l2
    from tpmsfem.io.abaqus import write_inp
    from tpmsfem.io.calculix import frd_field, read_dat_stress, read_frd, run_ccx

    m = _lattice_mesh(h=0.3)
    md = tf.SolidModel(m, TI)
    md.fix(m.nodes_where(lambda x, y, z: z < 1e-9))
    faces, ft, _ = m.boundary_faces()
    top = faces[m.points[faces].mean(1)[:, 2] > 3 - 1e-9]
    md.add_surface_load(top, ft, total_force=(20.0, 0.0, -100.0))
    md.set_eigenstrain(np.arange(m.n_cells), (-1e-3, -1e-3, -5e-4, 0, 0, 0))
    md.solve(solver="direct")
    r = run_ccx(write_inp(md, str(tmp_path / "lat.inp"), dialect="calculix"))
    U = frd_field(read_frd(r["frd"]), "DISP", m.n_points)
    S = read_dat_stress(r["dat"], m.n_cells, 8)
    assert rel_l2(md.displacement(), U) < 1e-5
    assert rel_l2(md.stress(), S) < 1e-5


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")
def test_cross_code_calculix_am_process(tmp_path):
    from tpmsfem.bench.metrics import rel_l2
    from tpmsfem.io.abaqus import write_inp
    from tpmsfem.io.calculix import frd_field, read_dat_stress, read_frd, run_ccx
    from tpmsfem.residual.am import AMProcess

    m = voxelize(box((0, 0, 0), (4, 2, 2)), (0, 0, 0), (4, 2, 2), 0.5)
    eps = (-0.004, -0.004, -0.002, 0, 0, 0)
    md = tf.SolidModel(m, TI)
    am = AMProcess(md, 0.5, eps, mode="nodal", birth_reference="element_mean", solver="direct")
    am.build(verbose=False)
    am.remove_from_plate(verbose=False)
    deck = write_inp(tf.SolidModel(m, TI), str(tmp_path / "am.inp"), dialect="calculix",
                     am=AMProcess(tf.SolidModel(m, TI), 0.5, eps, mode="nodal"))
    r = run_ccx(deck)
    U = frd_field(read_frd(r["frd"]), "DISP", m.n_points)
    S = read_dat_stress(r["dat"], m.n_cells, 8)
    assert rel_l2(md.displacement(), U) < 1e-5
    assert rel_l2(md.stress(), S) < 1e-5
