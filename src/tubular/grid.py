"""Rectilinear chart connectivity, retaining MATLAB's vertex and face order."""

from __future__ import annotations

import operator
import numpy as np

from .mesh import Mesh


def _size(value, name, minimum=2):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def rectilinear_faces(n_u: int, n_v: int) -> np.ndarray:
    """Exact zero-based defineFacesRectilinearGrid order; u varies fastest."""
    n_u, n_v = _size(n_u, "n_u"), _size(n_v, "n_v")
    i, j = np.meshgrid(np.arange(n_u - 1), np.arange(n_v - 1), indexing="ij")
    a = (j * n_u + i).ravel()
    faces = np.empty((2 * len(a), 3), dtype=np.int64)
    faces[0::2] = np.column_stack((a, a + 1, a + n_u + 1))
    faces[1::2] = np.column_stack((a, a + n_u + 1, a + n_u))
    return faces


def glue_rectilinear_seam(mesh: Mesh, n_u: int, n_v: int, *, tolerance=1e-10):
    """Return (glued_mesh, cut_to_glued). n_v includes the duplicated seam.

    tolerance is an absolute distance in mesh units. The final ring is mapped
    to the first only when their positions agree. The glued mesh has no uv:
    the cut chart is discontinuous across its newly glued faces.
    """
    n_u, n_v = _size(n_u, "n_u"), _size(n_v, "n_v", 4)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative distance")
    if len(mesh.vertices) != n_u * n_v:
        raise ValueError("vertex count must equal n_u * n_v")
    expected = rectilinear_faces(n_u, n_v)
    # Face order and winding may change, but connectivity must be this grid.
    canonical = lambda f: sorted(map(tuple, np.sort(f, axis=1)))
    if canonical(mesh.faces) != canonical(expected):
        raise ValueError("mesh does not have rectilinear grid connectivity")
    if np.any(np.linalg.norm(mesh.vertices[-n_u:] - mesh.vertices[:n_u], axis=1) > tolerance):
        raise ValueError("the two seam boundaries do not coincide within tolerance")
    count = n_u * (n_v - 1)
    cut_to_glued = np.concatenate((np.arange(count), np.arange(n_u)))
    glued = Mesh(mesh.vertices[:count], cut_to_glued[mesh.faces], length_unit=mesh.length_unit)
    return glued, cut_to_glued


def cylinder_mesh(n_u=9, n_v=33, *, radius=2.0, length=5.0, length_unit="unknown"):
    """Synthetic open cylinder with a duplicated seam and outward winding.

    This is an analytic demonstration chart, not an algorithm for mapping data.
    """
    n_u, n_v = _size(n_u, "n_u"), _size(n_v, "n_v", 4)
    if not np.isfinite([radius, length]).all() or radius <= 0 or length <= 0:
        raise ValueError("radius and length must be positive finite numbers")
    u, v = np.meshgrid(np.linspace(0, length, n_u), np.linspace(0, 2 * np.pi, n_v))
    uv = np.column_stack((u.ravel(), v.ravel()))
    # Negative sine gives outward normals with u-cross-v face winding.
    xyz = np.column_stack((uv[:, 0], radius * np.cos(uv[:, 1]), -radius * np.sin(uv[:, 1])))
    xyz[-n_u:] = xyz[:n_u]
    return Mesh(xyz, rectilinear_faces(n_u, n_v), uv, length_unit)
