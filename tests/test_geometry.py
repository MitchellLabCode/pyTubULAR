import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tubular import Mesh
from tubular.geometry import (averaging_operators, face_normals, fundamental_forms,
                              induced_metric, jacobian_2d_to_3d, jacobian_3d_to_2d,
                              pull_vectors, push_vectors, resolve_tangent_normal,
                              signed_volume, surface_area)
from tubular.mesh import topology


def test_affine_chart_analytic(plane):
    j = np.array([[2., 1.], [0., 3.], [1., 0.]])
    expected_metric = np.array([[5., 2.], [2., 10.]])
    assert_allclose(jacobian_2d_to_3d(plane), [j, j], atol=1e-14)
    assert_allclose(induced_metric(plane), [expected_metric] * 2, atol=1e-14)
    assert_allclose(surface_area(plane), np.sqrt(46))
    g, b = fundamental_forms(plane)
    assert_allclose(g, induced_metric(plane))
    assert_allclose(b, 0, atol=1e-14)
    assert_allclose(jacobian_3d_to_2d(plane) @ jacobian_2d_to_3d(plane),
                    [np.eye(2)] * 2, atol=1e-14)


def test_vector_roundtrip_and_normal_projection(plane):
    v = np.array([[1., 2.], [-3., 0.5]])
    pushed = push_vectors(plane, v)
    assert_allclose(pull_vectors(plane, pushed), v, atol=1e-14)
    n = face_normals(plane)
    normal, tangent = resolve_tangent_normal(plane, pushed + 7 * n)
    assert_allclose(normal, 7, atol=1e-14)
    assert_allclose(tangent, pushed, atol=1e-14)
    assert_allclose(pull_vectors(plane, n), 0, atol=1e-14)
    a, b = resolve_tangent_normal(plane, [n[1]], face_indices=[1])
    assert_allclose(a, 1)
    assert_allclose(b, 0, atol=1e-14)


def test_averaging_preserves_constants(plane):
    v2f, f2v = averaging_operators(plane)
    assert_allclose(v2f @ np.ones(4), 1)
    assert_allclose(f2v @ np.ones(2), 1)
    assert_allclose(v2f @ plane.vertices, plane.vertices[plane.faces].mean(axis=1))


def test_volume_orientation_translation_and_scaling(tetrahedron):
    assert_allclose(signed_volume(tetrahedron), 1)
    moved = tetrahedron.with_vertices(tetrahedron.vertices + [1e5, -2e5, 3e5])
    assert_allclose(signed_volume(moved), 1)
    scaled = tetrahedron.with_vertices(tetrahedron.vertices * 2)
    assert_allclose(signed_volume(scaled), 8)
    assert_allclose(surface_area(scaled), 4 * surface_area(tetrahedron))
    reverse = Mesh(tetrahedron.vertices, tetrahedron.faces[:, ::-1])
    assert_allclose(signed_volume(reverse), -1)
    assert_allclose(face_normals(reverse), -face_normals(tetrahedron))
    assert_allclose(face_normals(tetrahedron, convention="legacy_face_normals"),
                    -face_normals(tetrahedron))


def test_rigid_motion_metric_and_area_invariance(plane):
    q, _ = np.linalg.qr(np.random.default_rng(111).normal(size=(3, 3)))
    moved = plane.with_vertices(plane.vertices @ q + [13., -23., 53.])
    assert_allclose(induced_metric(moved), induced_metric(plane), atol=1e-12)
    assert_allclose(surface_area(moved), surface_area(plane))


def test_nonlinear_forms_change_sign_with_winding():
    from tubular.grid import cylinder_mesh
    mesh = cylinder_mesh()
    reverse = Mesh(mesh.vertices, mesh.faces[:, ::-1], mesh.uv)
    g, b = fundamental_forms(mesh)
    gr, br = fundamental_forms(reverse)
    assert_allclose(gr, g, atol=1e-13)
    assert_allclose(br, -b, atol=1e-13)
    assert np.max(np.abs(b)) > 0.1


def test_chart_reparameterization(plane):
    transform = np.array([[2., 0.3], [0.1, -1.5]])
    changed = Mesh(plane.vertices, plane.faces, plane.uv @ transform.T + [8, -4])
    inverse = np.linalg.inv(transform)
    assert_allclose(induced_metric(changed), inverse.T @ induced_metric(plane) @ inverse,
                    atol=1e-13)


def test_open_volume_rejected(plane):
    with pytest.raises(ValueError, match="closed"):
        signed_volume(plane)


def test_topology(tetrahedron, plane):
    t = topology(tetrahedron)
    assert t.closed and t.oriented and t.manifold
    assert t.euler_characteristic == 2 and t.components == 1
    p = topology(plane)
    assert p.boundary_edges == 4 and p.euler_characteristic == 1
    assert p.manifold and not p.closed
    broken = tetrahedron.faces.copy()
    broken[0] = broken[0, ::-1]
    with pytest.raises(ValueError, match="oriented"):
        signed_volume(Mesh(tetrahedron.vertices, broken))


def test_vertex_nonmanifold_and_edge_nonmanifold():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                  [-1., 0., 0.], [0., -1., 0.]])
    bowtie = Mesh(v, np.array([[0, 1, 2], [0, 3, 4]]))
    assert topology(bowtie).nonmanifold_vertices == 1
    v[3] = [0., 0., 1.]
    book = Mesh(v, np.array([[0, 1, 2], [1, 0, 4], [0, 1, 3]]))
    assert topology(book).nonmanifold_edges == 1


@pytest.mark.parametrize("scale", [1e-12, 1., 1e12])
def test_validation_is_scale_relative(plane, scale):
    scaled = Mesh(plane.vertices * scale, plane.faces, plane.uv * scale)
    assert_allclose(induced_metric(scaled), induced_metric(plane), atol=1e-12)


@pytest.mark.parametrize("kind", ["nan", "complex", "fractional", "one_based", "negative",
                                 "repeated", "duplicate", "unused", "collapsed", "bad_uv"])
def test_invalid_meshes_rejected(plane, kind):
    v, f, uv = plane.vertices.copy(), plane.faces.copy(), plane.uv.copy()
    if kind == "nan": v[0, 0] = np.nan
    if kind == "complex": v = v.astype(complex)
    if kind == "fractional": f = f.astype(float) + 0.2
    if kind == "one_based": f += 1
    if kind == "negative": f[0, 0] = -1
    if kind == "repeated": f[0, 0] = f[0, 1]
    if kind == "duplicate": f[1] = f[0]
    if kind == "unused": v, uv = np.vstack((v, [10, 10, 10])), np.vstack((uv, [10, 10]))
    if kind == "collapsed": v[1] = v[0]
    if kind == "bad_uv": uv[1] = uv[0]
    with pytest.raises(ValueError):
        Mesh(v, f, uv)


def test_mesh_copies_inputs(plane):
    v = plane.vertices.copy()
    mesh = Mesh(v, plane.faces, plane.uv)
    v[:] = 0
    assert_array_equal(mesh.vertices, plane.vertices)
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 4
