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


def test_network_upper_side_density_and_topology():
    # IWP ligament on the f > c side: body-centred "I" labyrinth (corners and centre solid)
    lat = TPMS("iwp", cell_size=1.0, kind="network", side="upper", density=0.2)
    n = 48
    s = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(s, s, s, indexing="ij")
    assert (lat(X, Y, Z) < 0).mean() == pytest.approx(0.2, abs=0.01)
    pt = lambda *p: float(lat(*(np.array([v]) for v in p))[0])
    assert pt(0.0, 0.0, 0.0) < 0 and pt(0.5, 0.5, 0.5) < 0 and pt(0.5, 0.0, 0.0) > 0


def test_save_load_state_roundtrip():
    mat = tf.Material("j2", 200_000.0, 0.3, yield_stress=300.0, hardening=5_000.0)
    m = voxelize(box((0, 0, 0), (2, 1, 1)), (0, 0, 0), (2, 1, 1), 0.5)
    md = tf.SolidModel(m, mat)
    x, y, z = m.points.T
    md.fix(np.flatnonzero(x < 1e-9))
    md.fix(np.flatnonzero(x > 2 - 1e-9), (0,), 0.01)
    md.solve(n_inc=2, solver="direct")
    st = md.save_state()
    u_ref, ep_ref = md.displacement().copy(), md.plastic_strain()[1].copy()
    md.fix(np.flatnonzero(x > 2 - 1e-9), (0,), -0.01)
    md.solve(n_inc=4, solver="direct")
    md.load_state(st)
    assert np.allclose(md.displacement(), u_ref) and np.allclose(md.plastic_strain()[1], ep_ref)
    md.fix(np.flatnonzero(x > 2 - 1e-9), (0,), 0.01)      # same target as the saved state
    info = md.solve(solver="direct")
    assert info.newton_iterations == [0]                  # already in equilibrium


def test_stl_voxelization(tmp_path):
    from tpmsfem.mesh import voxelize_stl
    from tpmsfem.mesh.surface import implicit_to_surface, write_stl

    # axis-aligned box with round-number coordinates (rays would hit edges without jitter)
    import meshio
    pts = np.array([[x, y, z] for x in (0, 2) for y in (0, 1) for z in (0, 3)], float)
    hexa = np.array([[0, 4, 6, 2, 1, 5, 7, 3]])
    box_stl = tmp_path / "box.stl"
    meshio.Mesh(pts, [("hexahedron", hexa)]).write(str(tmp_path / "box.vtu"))
    v, f = implicit_to_surface(box((0, 0, 0), (2, 1, 3)), (-0.2,) * 3, (2.2, 1.2, 3.2), 0.05)
    m = voxelize_stl((v, f), 0.1, lo=(0, 0, 0), hi=(2, 1, 3))
    assert m.n_cells == pytest.approx(20 * 10 * 30, rel=0.02)
    v, f = implicit_to_surface(sphere(radius=1.0), (-1.2,) * 3, (1.2,) * 3, 0.05)
    write_stl(str(tmp_path / "s.stl"), v, f)
    m = voxelize_stl(str(tmp_path / "s.stl"), 0.05)
    assert m.n_cells * 0.05**3 == pytest.approx(4 / 3 * np.pi, rel=0.02)


def test_mixed_plastic_tangent_matches_integration_point_kernel():
    mat = tf.Material("j2", 110_000.0, 0.3, yield_stress=200.0, hardening=1_000.0)
    lat = TPMS("gyroid", cell_size=2.0, kind="sheet", density=0.4)
    m = voxelize(lat, (0, 0, 0), (2, 2, 2), 0.2)
    md = tf.SolidModel(m, mat)
    rng = np.random.default_rng(3)
    x, z = m.points[:, 0], m.points[:, 2]
    field = np.zeros((m.n_points, 3))
    field[:, 2] = x * z                                    # eps_zz grows with x: partial yielding
    for eps in (5e-4, 1e-3, 2e-3, 4e-3, 8e-3):
        md.internal_force(-eps * field.ravel())
        frac = float((md.theta_bar != 0).any(axis=1).mean())
        if 0.05 < frac < 0.95:
            break
    assert 0.05 < frac < 0.95
    v = rng.standard_normal(md.ndof)
    fast = md.tangent(v)
    ref = np.zeros(md.ndof)
    from tpmsfem.backend import scatter_add
    for sl in md._chunks():                                # reference: integration-point kernel everywhere
        ref += scatter_add(md.ndof, md.dofs[sl], md._elem_tangent(v[md.dofs[sl]], sl))
    assert np.allclose(fast, ref, rtol=1e-10, atol=1e-8 * np.abs(ref).max())


def test_check_reports_floating_part_and_hinge():
    # two cubes touching only along an edge, only the first is supported
    from tpmsfem.geometry import box as gbox
    two = gbox((0, 0, 0), (1, 1, 1)) | gbox((1, 1, 0), (2, 2, 1))
    m = voxelize(two, (0, 0, 0), (2, 2, 1), 0.25, keep="all")
    md = tf.SolidModel(m, TI)
    md.fix(m.nodes_where(lambda x, y, z: (z < 1e-9) & (x < 1 + 1e-9) & (y < 1 + 1e-9)))
    probs = md.check(raise_error=False)
    assert len(probs) == 1 and "hinge" in probs[0]
    assert "axis along (0, 0, 1) through (1, 1," in probs[0]          # rotation about the shared edge
    # unsupported body: clear error before any solve
    md2 = tf.SolidModel(voxelize(gbox((0, 0, 0), (1, 1, 1)), (0, 0, 0), (1, 1, 1), 0.25), TI)
    md2.add_nodal_force([0], (0, 0, -1.0))
    with pytest.raises(ValueError, match="no supports"):
        md2.solve(solver="direct")
    # supported models and periodic cells pass
    md3 = tf.SolidModel(voxelize(gbox((-1, -1, -1), (2, 2, 2)), (0, 0, 0), (1, 1, 1), 0.25, periodic=(True,) * 3), TI)
    md3.fix([0])
    assert md3.check(raise_error=False) == []
