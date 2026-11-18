import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat
from scipy.spatial import ConvexHull

from tubular import Mesh, TubULAR
from tubular.curvature import estimate_curvature
from tubular.geometry import surface_area
from tubular.grid import cylinder_mesh, glue_rectilinear_seam, rectilinear_faces


def sphere(subdivisions=2, radius=1):
    vertices = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    faces = ConvexHull(vertices).simplices.copy()
    tri = vertices[faces]
    bad = np.einsum("ij,ij->i", np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), tri.mean(axis=1)) < 0
    faces[bad] = faces[bad][:, [0, 2, 1]]
    for _ in range(subdivisions):
        pts, out, mids = list(vertices), [], {}
        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in mids:
                point = vertices[a]+vertices[b]
                mids[key] = len(pts)
                pts.append(point/np.linalg.norm(point))
            return mids[key]
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            out.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        vertices, faces = np.array(pts), np.array(out)
    return Mesh(vertices*radius, faces, length_unit="um")


def patch():
    y, x = np.meshgrid([-.8, -.1, .7, 1.4], [-1, -.35, .2, 1.3], indexing="ij")
    x, y = x.ravel(), y.ravel()
    return Mesh(np.column_stack((x, y, .3*x*x+.1*x*y-.2*y*y)), rectilinear_faces(4, 4), length_unit="um")


@pytest.mark.parametrize("vertices", [[[0.,0,0], [2.,0,0], [.2,.3,0]],
                                      [[0.,0,0], [1.,0,0], [0,1.,0]],
                                      [[0.,0,0], [1.,0,0], [.4,.9,0]]])
def test_flat_triangles_all_corner_area_branches(vertices):
    mesh = Mesh(vertices, np.array([[0,1,2]]))
    result = estimate_curvature(mesh)
    assert_allclose(result.principal_curvatures, 0, atol=1e-14)
    assert_allclose(result.corner_areas.sum(), surface_area(mesh), rtol=1e-14)
    assert (result.corner_areas > 0).all()
    assert result.boundary_vertices.all()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_sphere_sign_magnitude_and_area_partition(level):
    mesh = sphere(level, radius=2.3)
    result = estimate_curvature(mesh)
    assert_allclose(result.normals, mesh.v/2.3, atol=1e-12)
    assert_allclose(result.principal_curvatures, 1/2.3, rtol=1e-11, atol=1e-12)
    assert_allclose(result.mean, -1/2.3, rtol=1e-11, atol=1e-12)
    assert_allclose(result.gaussian, 1/2.3**2, rtol=1e-11, atol=1e-12)
    assert not result.boundary_vertices.any()
    assert_allclose(result.vertex_areas.sum(), surface_area(mesh), rtol=1e-14)


def test_periodic_cylinder_interior_and_end_boundaries():
    cut = cylinder_mesh(11, 49, radius=2, length=8, length_unit="um")
    mesh, _ = glue_rectilinear_seam(cut, 11, 49)
    result = estimate_curvature(mesh)
    middle = (mesh.v[:,0] > 1.5) & (mesh.v[:,0] < 6.5)
    assert_allclose(result.principal_curvatures[middle, 0], .5, atol=1e-12)
    assert_allclose(result.principal_curvatures[middle, 1], 0, atol=1e-12)
    assert_allclose(result.mean[middle], -.25, atol=1e-12)
    assert_allclose(result.gaussian[middle], 0, atol=1e-12)
    expected_boundary = (mesh.v[:,0] == 0) | (mesh.v[:,0] == 8)
    assert_array_equal(result.boundary_vertices, expected_boundary)
    # No implicit seam gluing: the original cut retains its seam boundaries.
    cut_result = estimate_curvature(cut)
    assert cut_result.boundary_vertices[:11].all()


def test_rigid_motion_scaling_winding_and_face_order():
    mesh = patch()
    original = estimate_curvature(mesh)
    q, _ = np.linalg.qr(np.array([[1., 2, 3], [-2., 1, 1], [1., 3, -1]]))
    moved = estimate_curvature(mesh.with_vertices(mesh.v @ q.T+[2, -3, 5]))
    assert_allclose(moved.mean, original.mean, atol=1e-13)
    assert_allclose(moved.gaussian, original.gaussian, atol=1e-13)
    assert_allclose(moved.shape_operator, q @ original.shape_operator @ q.T, atol=1e-13)
    assert_allclose(moved.normals, original.normals @ q.T, atol=1e-14)
    for scale in (1e-6, 3, 1e6):
        result = estimate_curvature(mesh.with_vertices(mesh.v*scale))
        assert_allclose(result.mean*scale, original.mean, atol=1e-13)
        assert_allclose(result.gaussian*scale**2, original.gaussian, atol=1e-13)
        assert_allclose(result.vertex_areas/scale**2, original.vertex_areas, atol=1e-14)
    reversed_result = estimate_curvature(Mesh(mesh.v, mesh.f[:, [0,2,1]]))
    assert_allclose(reversed_result.mean, -original.mean, atol=1e-13)
    assert_allclose(reversed_result.gaussian, original.gaussian, atol=1e-13)
    assert_allclose(reversed_result.shape_operator, -original.shape_operator, atol=1e-13)
    reordered = estimate_curvature(Mesh(mesh.v, mesh.f[::-1]))
    assert_allclose(reordered.shape_operator, original.shape_operator, atol=1e-13)


def test_principal_eigenvectors_and_readonly_outputs():
    mesh = patch()
    result = estimate_curvature(mesh)
    directions, k = result.principal_directions, result.principal_curvatures
    assert_allclose(np.einsum("nij,nkj->nik", directions, directions), np.tile(np.eye(2), (len(mesh.v),1,1)), atol=1e-14)
    assert_allclose(np.einsum("nij,nj->ni", directions, result.normals), 0, atol=1e-14)
    assert_allclose(np.einsum("nij,nkj->nki", result.shape_operator, directions), directions*k[...,None], atol=1e-14)
    assert_allclose(np.cross(directions[:,0], directions[:,1]), result.normals, atol=1e-14)
    assert np.all(np.abs(k[:,0]) >= np.abs(k[:,1]))
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    assert result.length_unit == "um"


def test_rejects_inconsistent_winding_and_canceled_normals():
    mesh = patch()
    faces = mesh.f.copy(); faces[0] = faces[0, [0,2,1]]
    with pytest.raises(ValueError, match="oriented manifold"):
        estimate_curvature(Mesh(mesh.v, faces))
    folded = Mesh([[0.,0,0],[1.,0,0],[0.,1,0],[0.,1,0]], np.array([[0,1,2],[1,0,3]]))
    with pytest.raises(ValueError, match="vertex normal"):
        estimate_curvature(folded)


def test_series_variable_meshes_units_and_selection(tmp_path):
    first, second = sphere(0), sphere(1, radius=2)
    tubi = TubULAR(time_points=[2,5], opts={"timeInterval": 3, "meshDir": tmp_path},
                    mesh_sources={"spcutMeshSmRSC": {2: first, 5: second}})
    tubi.setTime(2)
    cached = tubi.getCurrentSPCutMeshSmRSC()
    result = tubi.measureCurvatures()
    assert_array_equal(result["times"], [0,9])
    assert_allclose(result["curvatures"][0].mean, -1, atol=1e-14)
    assert_allclose(result["curvatures"][1].mean, -.5, atol=1e-14)
    assert tubi.getCurrentSPCutMeshSmRSC() is cached and tubi.currentTime == 2
    assert not list(tmp_path.iterdir())
    tubi.register_mesh(5, Mesh(second.v, second.f, length_unit="mm"), kind="spcutMeshSmRSC")
    with pytest.raises(ValueError, match="units"):
        tubi.measureCurvatures()
    assert tubi.currentTime == 2 and tubi.getCurrentSPCutMeshSmRSC() is cached


def test_series_invalid_or_missing_inputs():
    tubi = TubULAR(time_points=[0,2])
    with pytest.raises(FileNotFoundError):
        tubi.measureCurvatures()
    with pytest.raises(ValueError, match="kind"):
        tubi.measureCurvatures(mesh_kind="invented")
    with pytest.raises(ValueError):
        tubi.measureCurvatures(time_points=[2,0])


def test_original_matlab_curvature_reference():
    root = Path(__file__).parent / "data"
    fixture = root / "matlab_curvature_reference.mat"
    provenance = json.loads((root / "curvature_provenance.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    for case in loadmat(fixture, simplify_cells=True)["cases"]:
        mesh = Mesh(case["vertices"], case["faces"].astype(int)-1)
        result = estimate_curvature(mesh)
        for actual, key in ((result.principal_curvatures,"principal"), (result.mean,"mean"),
                            (result.gaussian,"gaussian"), (result.normals,"normals"),
                            (result.vertex_areas,"vertexAreas"), (result.corner_areas,"cornerAreas")):
            assert_allclose(actual, case[key], rtol=1e-10, atol=1e-12)
        assert_allclose(result.shape_operator, np.moveaxis(case["shapeOperator"], -1, 0), rtol=1e-10, atol=1e-12)
        # Eigenvector signs are arbitrary; compare rank-one projectors.
        expected = case["d1"]
        actual = result.principal_directions[:,0]
        assert_allclose(actual[:,:,None]*actual[:,None,:], expected[:,:,None]*expected[:,None,:], rtol=1e-10, atol=1e-12)
