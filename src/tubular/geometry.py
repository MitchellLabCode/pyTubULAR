"""Geometry translated from TubULAR; see docs/translation.md for conventions."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

from .mesh import Mesh, real_array, require_oriented_manifold


def _cross(mesh: Mesh) -> np.ndarray:
    t = mesh.vertices[mesh.faces]
    return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])


def face_areas(mesh: Mesh) -> np.ndarray:
    return 0.5 * np.linalg.norm(_cross(mesh), axis=1)


def surface_area(mesh: Mesh) -> float:
    return float(np.sum(face_areas(mesh)))


def face_normals(mesh: Mesh, *, convention: str = "right_hand") -> np.ndarray:
    """Unit normals: MATLAB triangulation.faceNormal by default.

    convention='legacy_face_normals' reproduces the opposite sign in
    utility/mesh_handling/faceNormals.m. Neither convention infers outwardness.
    """
    if convention not in ("right_hand", "legacy_face_normals"):
        raise ValueError("unknown normal convention")
    normals = _cross(mesh)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals if convention == "right_hand" else -normals


def signed_volume(mesh: Mesh) -> float:
    """Signed enclosed volume; requires a closed, oriented manifold.

    Positive for outward winding, negative for inward winding. Self-intersections
    and shell nesting are not detected, so this is an algebraic volume.
    """
    info = require_oriented_manifold(mesh)
    if not info.closed:
        raise ValueError("volume requires a closed mesh; open tubes need explicit caps")
    v = mesh.vertices - mesh.vertices.mean(axis=0)
    t = v[mesh.faces]
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6)


def averaging_operators(mesh: Mesh):
    """Return CSR V2F (uniform) and F2V (normalized corner-angle weights)."""
    f = mesh.faces
    triangles = mesh.vertices[f]
    angles = np.empty(f.shape)
    for i in range(3):
        a = triangles[:, (i + 1) % 3] - triangles[:, i]
        b = triangles[:, (i + 2) % 3] - triangles[:, i]
        # atan2 avoids acos domain errors near 0/pi.
        angles[:, i] = np.arctan2(np.linalg.norm(np.cross(a, b), axis=1),
                                 np.einsum("ij,ij->i", a, b))
    face_ids = np.repeat(np.arange(len(f)), 3)
    vertex_ids = f.ravel()
    totals = np.bincount(vertex_ids, weights=angles.ravel(), minlength=len(mesh.vertices))
    v2f = coo_matrix((np.full(f.size, 1 / 3), (face_ids, vertex_ids)),
                     shape=(len(f), len(mesh.vertices))).tocsr()
    f2v = coo_matrix((angles.ravel() / totals[vertex_ids], (vertex_ids, face_ids)),
                     shape=(len(mesh.vertices), len(f))).tocsr()
    return v2f, f2v


def _chart(mesh: Mesh):
    if mesh.uv is None:
        raise ValueError("this operation requires an explicit uv chart")
    return mesh.uv[mesh.faces]


def jacobian_2d_to_3d(mesh: Mesh) -> np.ndarray:
    """Per-face (F, 3, 2) Jacobian acting on contravariant vectors."""
    uv, xyz = _chart(mesh), mesh.vertices[mesh.faces]
    domain = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=1)
    image = np.stack((xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    return np.swapaxes(np.linalg.solve(domain, image), 1, 2)


def jacobian_3d_to_2d(mesh: Mesh) -> np.ndarray:
    """Per-face (F, 2, 3) tangent inverse; discards the normal component."""
    return np.linalg.pinv(jacobian_2d_to_3d(mesh))


def induced_metric(mesh: Mesh) -> np.ndarray:
    j = jacobian_2d_to_3d(mesh)
    return np.einsum("fki,fkj->fij", j, j)


def fundamental_forms(mesh: Mesh):
    """Return (g, b), each (F, 2, 2), matching constructFundamentalForms.m.

    Angle-averaged vertex normals are deliberately NOT renormalized: MATLAB
    does not do so. b = -sym(dX.T @ dn). Boundary/seam neighborhoods remain
    distinct, just as in the input; this is not a periodic curvature estimator.
    """
    require_oriented_manifold(mesh)
    j = jacobian_2d_to_3d(mesh)
    _, f2v = averaging_operators(mesh)
    vertex_normals = f2v @ face_normals(mesh)
    uv, n = _chart(mesh), vertex_normals[mesh.faces]
    domain = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=1)
    dn = np.linalg.solve(domain, np.stack((n[:, 1] - n[:, 0], n[:, 2] - n[:, 0]), axis=1))
    b = -np.einsum("fki,fjk->fij", j, dn)
    return np.einsum("fki,fkj->fij", j, j), 0.5 * (b + b.swapaxes(1, 2))


def resolve_tangent_normal(mesh: Mesh, vectors, *, face_indices=None):
    """Return signed normal scalar and tangential 3-vector at each face sample."""
    vectors = real_array(vectors, "vectors", ndim=2)
    if vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    normals = face_normals(mesh)
    if face_indices is not None:
        ids = np.asarray(face_indices)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) != len(vectors)
                or np.any(ids < 0) or np.any(ids >= len(mesh.faces))):
            raise ValueError("face_indices must be valid zero-based indices, one per vector")
        normals = normals[ids]
    if normals.shape != vectors.shape:
        raise ValueError("one vector per face is required unless face_indices are supplied")
    normal = np.einsum("ij,ij->i", normals, vectors)
    return normal, vectors - normal[:, None] * normals


def push_vectors(mesh: Mesh, vectors) -> np.ndarray:
    """Map one contravariant 2-vector per face into 3D."""
    vectors = real_array(vectors, "vectors", ndim=2)
    if vectors.shape != (len(mesh.faces), 2):
        raise ValueError("vectors must have shape (F, 2)")
    return np.einsum("fij,fj->fi", jacobian_2d_to_3d(mesh), vectors)


def pull_vectors(mesh: Mesh, vectors) -> np.ndarray:
    """Project one 3-vector per face into contravariant chart coordinates."""
    vectors = real_array(vectors, "vectors", ndim=2)
    if vectors.shape != (len(mesh.faces), 3):
        raise ValueError("vectors must have shape (F, 3)")
    return np.einsum("fij,fj->fi", jacobian_3d_to_2d(mesh), vectors)
