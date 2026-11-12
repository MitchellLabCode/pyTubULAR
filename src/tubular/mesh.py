"""Explicit, validated triangular meshes. Coordinates are always (x, y, z)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def real_array(value, name: str, *, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real numeric array with {ndim} dimensions")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def nondegenerate(points: np.ndarray, faces: np.ndarray, name: str) -> None:
    """Use a local, scale-relative area threshold, also for very small meshes."""
    tri = points[faces]
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
    safe = np.where(scale > 0, scale, 1)
    a, b = a / safe[:, None], b / safe[:, None]
    if points.shape[1] == 2:
        area2 = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    else:
        area2 = np.linalg.norm(np.cross(a, b), axis=1)
    if np.any((scale == 0) | (area2 <= 64 * np.finfo(float).eps)):
        raise ValueError(f"{name} has degenerate or numerically singular triangles")


@dataclass(frozen=True, eq=False)
class Mesh:
    """A snapshot of a mesh, with copied read-only arrays.

    faces use zero-based integer indices. uv, when present, is an existing
    per-vertex chart; it is never inferred. Duplicated seam vertices remain
    distinct. Units are caller-supplied and no conversion is implicit.
    Manifoldness/orientation are checked by topology(), not repaired.
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv: Optional[np.ndarray] = None
    length_unit: str = "unknown"

    def __post_init__(self):
        v = real_array(self.vertices, "vertices", ndim=2)
        f = np.asarray(self.faces)
        if v.shape[1] != 3 or len(v) < 3:
            raise ValueError("vertices must have shape (N, 3), N >= 3")
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or f.dtype.kind not in "iu":
            raise ValueError("faces must be a nonempty (M, 3) integer array")
        if np.any(f < 0) or np.any(f >= len(v)):
            raise ValueError("faces must use zero-based indices within vertices")
        f = np.array(f, dtype=np.int64, copy=True)
        sorted_f = np.sort(f, axis=1)
        if np.any(np.diff(sorted_f, axis=1) == 0):
            raise ValueError("a face cannot repeat a vertex")
        if len(np.unique(sorted_f, axis=0)) != len(f):
            raise ValueError("duplicate faces are not allowed")
        if len(np.unique(f)) != len(v):
            raise ValueError("unreferenced vertices must be removed explicitly")
        nondegenerate(v, f, "vertices")
        uv = None
        if self.uv is not None:
            uv = real_array(self.uv, "uv", ndim=2)
            if uv.shape != (len(v), 2):
                raise ValueError("uv must have shape (N, 2)")
            nondegenerate(uv, f, "uv")
            uv.setflags(write=False)
        if not isinstance(self.length_unit, str) or not self.length_unit.strip():
            raise ValueError("length_unit must be a nonempty string")
        v.setflags(write=False)
        f.setflags(write=False)
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)
        object.__setattr__(self, "uv", uv)

    def with_vertices(self, vertices) -> Mesh:
        return Mesh(vertices, self.faces, self.uv, self.length_unit)

    @property
    def v(self):
        """MATLAB-style alias for vertices (XYZ)."""
        return self.vertices

    @property
    def f(self):
        """MATLAB-style alias for faces, still ZERO-based in Python."""
        return self.faces

    @property
    def u(self):
        """MATLAB-style alias for the supplied UV chart, or None."""
        return self.uv


@dataclass(frozen=True)
class Topology:
    vertices: int
    edges: int
    faces: int
    boundary_edges: int
    nonmanifold_edges: int
    nonmanifold_vertices: int
    inconsistently_oriented_edges: int
    components: int

    @property
    def euler_characteristic(self) -> int:
        return self.vertices - self.edges + self.faces

    @property
    def manifold(self) -> bool:
        return self.nonmanifold_edges == 0 and self.nonmanifold_vertices == 0

    @property
    def oriented(self) -> bool:
        return self.inconsistently_oriented_edges == 0

    @property
    def closed(self) -> bool:
        return self.boundary_edges == 0 and self.manifold


def topology(mesh: Mesh) -> Topology:
    """Check edge incidence and vertex links, including bow-tie vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    f = mesh.faces
    directed = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    sign_sum = np.bincount(inverse, weights=signs)
    bad_orientation = np.count_nonzero((counts == 2) & (sign_sum != 0))
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    ).tocsr()
    components = connected_components(adjacency, directed=False, return_labels=False)

    # A manifold vertex has one connected link: a cycle or an open path.
    links = [[] for _ in mesh.vertices]
    for a, b, c in f:
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    bad_vertices = 0
    for link in links:
        neighbors = {}
        for a, b in link:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        seen = set()
        stack = [next(iter(neighbors))]
        while stack:
            vertex = stack.pop()
            if vertex not in seen:
                seen.add(vertex)
                stack.extend(neighbors[vertex] - seen)
        degrees = [len(n) for n in neighbors.values()]
        valid_degrees = all(d == 2 for d in degrees) or (
            degrees.count(1) == 2 and all(d in (1, 2) for d in degrees)
        )
        bad_vertices += len(seen) != len(neighbors) or not valid_degrees
    return Topology(
        len(mesh.vertices), len(edges), len(f), int(np.count_nonzero(counts == 1)),
        int(np.count_nonzero(counts > 2)), int(bad_vertices), int(bad_orientation),
        int(components),
    )


def require_oriented_manifold(mesh: Mesh) -> Topology:
    result = topology(mesh)
    if not result.manifold or not result.oriented:
        raise ValueError("mesh must be a consistently oriented manifold")
    return result
