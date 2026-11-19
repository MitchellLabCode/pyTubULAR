import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat

from tubular import Mesh, TubULAR
from tubular.geometry import surface_area
from tubular.grid import cylinder_mesh, glue_rectilinear_seam
from tubular.mesh import topology
from tubular.parameterization import cut_annulus, flatten_annulus, _check_embedding


def annulus(nu=6, nv=13):
    mesh, _ = glue_rectilinear_seam(cylinder_mesh(nu, nv+1, length_unit="um"), nu, nv+1)
    return mesh


def curved_annulus():
    mesh = annulus()
    x = mesh.v[:, 0]
    theta = np.arctan2(-mesh.v[:, 2], mesh.v[:, 1])
    radius = 1+.17*np.sin(1.1*x)+.08*np.cos(3*theta+.4*x)
    angle = theta+.13*x+.04*np.sin(2*theta)*np.sin(x)
    return mesh.with_vertices(np.column_stack((x+.07*np.sin(theta)*np.sin(x),
        radius*np.cos(angle)+.12*x*x, -.75*radius*np.sin(angle)+.08*x)))


@pytest.mark.parametrize("path", [np.arange(6), np.arange(6)[::-1],
                                    np.arange(6)*7, [0,1,7,8,14,15,21,22,28,29],
                                    [0,1,7,13,19,20,14,8,9,10,11]])
def test_cut_preserves_geometry_faces_fields_and_topology(path):
    mesh = curved_annulus()
    source_v, source_f = mesh.v.copy(), mesh.f.copy()
    cut = cut_annulus(mesh, path)
    assert_array_equal(cut.cut_to_uncut[cut.mesh.f], mesh.f)
    assert_array_equal(cut.mesh.v, mesh.v[cut.cut_to_uncut])
    assert_array_equal(cut.path_pairs[:,0], path)
    assert_array_equal(cut.mesh.v[cut.path_pairs[:,0]], cut.mesh.v[cut.path_pairs[:,1]])
    topo = topology(cut.mesh)
    assert topo.euler_characteristic == 1 and topo.components == 1
    assert topo.manifold and topo.oriented
    assert topo.boundary_edges == topology(mesh).boundary_edges+2*(len(path)-1)
    assert_allclose(surface_area(cut.mesh), surface_area(mesh), rtol=1e-14)
    assert cut.mesh.length_unit == "um" and cut.mesh.uv is None
    vertex_field = np.arange(len(mesh.v))**2
    transferred = vertex_field[cut.cut_to_uncut]
    for original, copies in enumerate(cut.uncut_to_cut):
        assert_array_equal(cut.cut_to_uncut[copies], original)
        assert_array_equal(transferred[copies], vertex_field[original])
        assert not copies.flags.writeable
    for a in (cut.path_pairs, cut.cut_to_uncut, cut.mesh.v, cut.mesh.f):
        assert not a.flags.writeable
    assert_array_equal(mesh.v, source_v)
    assert_array_equal(mesh.f, source_f)
    # Includes a hairpin whose two sides share path-adjacent faces. Correct
    # fan splitting must still give a valid, seamless embedding.
    chart = flatten_annulus(cut.mesh, cut.path_pairs)
    assert_allclose(chart.mesh.uv[cut.path_pairs[:,0]]-chart.mesh.uv[cut.path_pairs[:,1]],
                    np.tile([0,1], (len(path),1)), atol=1e-12)


@pytest.mark.parametrize("nu,nv", [(2,3), (6,13), (17,32)])
def test_cylinder_analytic_uv_and_distortion(nu, nv):
    mesh = annulus(nu, nv)
    cut = cut_annulus(mesh, np.arange(nu))
    result = flatten_annulus(cut.mesh, cut.path_pairs)
    expected = np.column_stack((np.tile(np.linspace(0, 1, nu), nv),
                               np.repeat(1-np.arange(nv)/nv, nu)))
    expected = np.vstack((expected, np.column_stack((np.linspace(0,1,nu), np.zeros(nu)))))
    assert_allclose(result.mesh.uv, expected, rtol=1e-12, atol=1e-13)
    assert result.orientation == -1 and result.clamped_edges == 0
    assert result.linear_residual < 1e-12 and result.constraint_residual < 1e-13
    # A regular polygonal cylinder is intrinsically flat. Lateral scaling by
    # length / polygon circumference makes each triangle map a similarity.
    from tubular.quasiconformal import beltrami_coefficient
    width = 5/(nv*4*np.sin(np.pi/nv))
    conformal = flatten_annulus(cut.mesh, cut.path_pairs, aspect_ratio=width)
    distortion = beltrami_coefficient(conformal.mesh)
    assert_allclose(distortion.mu, 0, atol=1e-12)


def test_curved_seams_constraints_and_negative_weight_policy():
    mesh = curved_annulus()
    cut = cut_annulus(mesh, np.arange(6)*7)
    result = flatten_annulus(cut.mesh, cut.path_pairs)
    pairs, uv = cut.path_pairs, result.mesh.uv
    assert_allclose(uv[pairs[:,0]]-uv[pairs[:,1]], np.tile([0,1], (len(pairs),1)), atol=1e-13)
    assert result.clamped_edges > 0
    assert not np.allclose(uv[pairs[:,0],1], 1, atol=1e-12)  # The seam is free to curve.
    assert_allclose(uv[:,0].min(), 0, atol=1e-13)
    assert_allclose(uv[:,0].max(), 1, atol=1e-13)
    for end in (0, 5):
        ids = np.arange(end, len(mesh.v), 6)
        assert_allclose(uv[ids,0], end/5, atol=1e-13)
    with pytest.raises(ValueError, match="negative edge weights"):
        flatten_annulus(cut.mesh, pairs, negative_weights="error")
    with pytest.raises(ValueError, match="negative_weights"):
        flatten_annulus(cut.mesh, pairs, negative_weights="invented")


def test_intrinsic_invariance_and_vertex_face_permutations():
    mesh = curved_annulus()
    path = np.arange(6)*7
    cut = cut_annulus(mesh, path)
    expected = flatten_annulus(cut.mesh, cut.path_pairs).mesh.uv
    q, _ = np.linalg.qr(np.array([[1.,2,3], [-2.,1,1], [1.,3,-1]]))
    for scale in (1e-8, 1, 1e8):
        moved = mesh.with_vertices((mesh.v @ q.T+[2,-3,4])*scale)
        c = cut_annulus(moved, path)
        result = flatten_annulus(c.mesh, c.path_pairs)
        assert_allclose(result.mesh.uv, expected, atol=2e-12, rtol=1e-12)
    rng = np.random.default_rng(17)
    permutation = rng.permutation(len(mesh.v))
    inverse = np.argsort(permutation)
    face_order = rng.permutation(len(mesh.f))
    renamed = Mesh(mesh.v[permutation], inverse[mesh.f[face_order]])
    c = cut_annulus(renamed, inverse[path])
    result = flatten_annulus(c.mesh, c.path_pairs)
    assert_allclose(result.mesh.uv[:len(mesh.v)], expected[permutation], atol=1e-12)
    assert_allclose(result.mesh.uv[len(mesh.v):], expected[len(mesh.v):], atol=1e-12)


def test_boundary_arclength_and_explicit_seam_side_conventions():
    source = curved_annulus()
    cut = cut_annulus(source, np.arange(6))
    result = flatten_annulus(cut.mesh, cut.path_pairs)
    for endpoint in (0,5):
        ring = np.append(np.arange(endpoint,len(source.v),6), len(source.v)+endpoint)
        lengths = np.linalg.norm(np.diff(cut.mesh.v[ring],axis=0),axis=1)
        expected_v = 1-np.append(0,np.cumsum(lengths))/sum(lengths)
        assert_allclose(result.mesh.uv[ring,1],expected_v,atol=1e-13)
    swapped = flatten_annulus(cut.mesh, cut.path_pairs[:,::-1])
    assert_allclose(swapped.mesh.uv, result.mesh.uv*[1,-1]+[0,1],atol=1e-12)
    assert swapped.orientation == -result.orientation
    reversed_faces = Mesh(cut.mesh.v, cut.mesh.f[:,[0,2,1]])
    reversed_result = flatten_annulus(reversed_faces,cut.path_pairs)
    assert_allclose(reversed_result.mesh.uv,result.mesh.uv,atol=1e-12)
    assert reversed_result.orientation == -result.orientation
    # A pre-existing chart is not used as an initializer or silently reused.
    old_chart = Mesh(cut.mesh.v,cut.mesh.f,result.mesh.uv*2+[8,-7])
    assert_allclose(flatten_annulus(old_chart,cut.path_pairs).mesh.uv,result.mesh.uv,atol=1e-13)


@pytest.mark.parametrize("path", [[0], [0,1,1,2,3,4,5], [1,2,3,4,5],
                                    [0,1,2,3,4,6], [0,6,7,8,9,10,11],
                                    [0,2,3,4,5], [0.,1.,2.,3.,4.,5.],
                                    [True,False], [0,1,2,3,4,99], [[0,1,2,3,4,5]]])
def test_invalid_seams_fail(path):
    with pytest.raises(ValueError):
        cut_annulus(annulus(), path)


def test_invalid_topology_and_winding_fail():
    source = annulus()
    faces = source.f.copy(); faces[0] = faces[0,[0,2,1]]
    with pytest.raises(ValueError, match="oriented manifold"):
        cut_annulus(Mesh(source.v, faces), np.arange(6))
    disk = cylinder_mesh(6,14)
    with pytest.raises(ValueError, match="annulus"):
        cut_annulus(disk, np.arange(6))
    disconnected = Mesh(np.vstack((source.v,source.v+[10,0,0])),
                        np.vstack((source.f,source.f+len(source.v))))
    with pytest.raises(ValueError, match="annulus"):
        cut_annulus(disconnected, np.arange(6))
    with pytest.raises(ValueError, match="disk"):
        flatten_annulus(source, np.column_stack((np.arange(6), np.arange(6)+6)))


@pytest.mark.parametrize("mutation", ["reorder", "shorten", "float", "repeat", "columns", "interior"])
def test_invalid_seam_pairs_fail(mutation):
    cut = cut_annulus(annulus(), np.arange(6))
    pairs = cut.path_pairs.copy()
    if mutation == "reorder":
        pairs[[1,2]] = pairs[[2,1]]
    elif mutation == "shorten":
        pairs = pairs[[0,2,3,4,5]]
    elif mutation == "float":
        pairs = pairs.astype(float)
    elif mutation == "repeat":
        pairs[1] = pairs[0]
    elif mutation == "columns":
        pairs = pairs[:,:1]
    else:
        pairs[2,0] = 8
    with pytest.raises(ValueError):
        flatten_annulus(cut.mesh, pairs)


def test_mismatched_seam_coordinates_fail():
    cut = cut_annulus(annulus(), np.arange(6))
    vertices = cut.mesh.v.copy(); vertices[-1] += [0,0,1e-8]
    with pytest.raises(ValueError, match="identical positions"):
        flatten_annulus(Mesh(vertices, cut.mesh.f), cut.path_pairs)


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan, True, [1]])
def test_invalid_aspect_ratio_fails(width):
    cut = cut_annulus(annulus(), np.arange(6))
    with pytest.raises(ValueError):
        flatten_annulus(cut.mesh, cut.path_pairs, aspect_ratio=width)


def test_foldover_and_global_overlap_checks():
    # Two overlapping triangles with opposing chart winding.
    vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
    faces = np.array([[0,1,2], [0,2,3]])
    folded = Mesh(vertices, faces, [[0,0],[1,0],[1,1],[1,0]])
    with pytest.raises(ValueError, match="flipped"):
        _check_embedding(folded, np.arange(4))
    # A fan that winds twice: EVERY local determinant is positive, yet the
    # boundary self-intersects. This must not pass a local-only fold test.
    t = np.linspace(0, 4*np.pi, 10, endpoint=False)
    uv = np.vstack(([0,0], np.column_stack((np.cos(t),np.sin(t)))))
    faces = np.array([[0,i+1,(i+1)%10+1] for i in range(10)])
    double = Mesh(np.column_stack((uv,np.zeros(len(uv)))), faces, uv)
    with pytest.raises(ValueError, match="boundary"):
        _check_embedding(double, np.arange(1,11))


def test_class_preview_registration_failure_and_selection(tmp_path):
    source = curved_annulus()
    tubi = TubULAR(time_points=[2,5], opts={"meshDir":tmp_path, "flipy":True},
                  mesh_sources={"cylinderMeshClean":{2:source}})
    with pytest.raises(ValueError, match="setTime"):
        tubi.generateCurrentCutMesh(np.arange(6))
    tubi.setTime(2)
    cached = tubi.getCurrentCylinderMeshClean()
    preview = tubi.generateCurrentCutMesh(np.arange(6))
    with pytest.raises(FileNotFoundError):
        tubi.getCurrentCutMesh()
    result = tubi.generateCurrentCutMesh(np.arange(6), aspect_ratio=2, register=True)
    assert_allclose(result.mesh.uv, preview.mesh.uv*[2,1], atol=1e-13)
    assert tubi.getCurrentCutMesh() is result.mesh
    assert tubi.getCurrentCylinderMeshClean() is cached
    assert tubi.currentTime == 2
    with pytest.raises(ValueError):
        tubi.generateCurrentCutMesh([0,1,2,3,5], register=True)
    assert tubi.getCurrentCutMesh() is result.mesh
    with pytest.raises(ValueError, match="negative"):
        tubi.generateCurrentCutMesh(np.arange(6), negative_weights="error", register=True)
    assert tubi.getCurrentCutMesh() is result.mesh
    tubi.setTime(5)
    with pytest.raises(FileNotFoundError):
        tubi.generateCurrentCutMesh(np.arange(6))
    tubi.setTime(2)
    assert tubi.getCurrentCutMesh() is result.mesh
    assert not list(tmp_path.iterdir())
    assert TubULAR.method_status("generateCurrentCutMesh")["status"] == "adapted"


def test_original_matlab_parameterization_reference():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/data/matlab_parameterization_reference.mat"
    provenance = json.loads((root / "tests/data/parameterization_provenance.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    generator = root / "scripts/generate_parameterization_reference.m"
    assert hashlib.sha256(generator.read_bytes()).hexdigest() == provenance["generator_sha256"]
    for case in loadmat(fixture, simplify_cells=True)["cases"]:
        source = Mesh(case["vertices"], case["faces"].astype(int)-1)
        cut = cut_annulus(source, case["seam"].astype(int)-1)
        assert_array_equal(cut.mesh.v, case["cutVertices"])
        assert_array_equal(cut.mesh.f, case["cutFaces"]-1)
        assert_array_equal(cut.path_pairs, case["pathPairs"]-1)
        assert_array_equal(cut.cut_to_uncut, case["cutToUncut"]-1)
        mapped = flatten_annulus(cut.mesh, cut.path_pairs)
        assert mapped.clamped_edges == case["clampedEdges"]
        assert_allclose(mapped.mesh.uv, case["uv"], rtol=1e-11, atol=1e-12)
