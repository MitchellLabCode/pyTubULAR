"""Interpolation on supplied triangular charts, retaining their connectivity."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ._coordinates import points
from .mesh import Mesh, real_array


@dataclass(frozen=True)
class ChartLocation:
    """Face indices are zero-based; outside points have index -1 and NaN weights."""

    mesh: Mesh
    face_indices: np.ndarray
    barycentric: np.ndarray

    @property
    def valid(self):
        return self.face_indices >= 0

    def interpolate(self, values, *, location="vertices"):
        """Interpolate scalar/vector vertex values, or piecewise-constant face values.

        Shared-edge face values use the first face in the supplied ordering.
        No nearest-face extrapolation occurs outside the chart.
        """
        a = np.asarray(values)
        if a.ndim not in (1, 2):
            raise ValueError("values must be a scalar or vector field with shape (N,) or (N, K)")
        a = real_array(a, "values", ndim=a.ndim)
        if location not in ("vertices", "faces"):
            raise ValueError("location must be 'vertices' or 'faces'")
        expected = len(self.mesh.v) if location == "vertices" else len(self.mesh.f)
        if len(a) != expected or (a.ndim == 2 and a.shape[1] == 0):
            raise ValueError(f"values must contain one value per {location} entry")
        result = np.full((len(self.face_indices),) + a.shape[1:], np.nan)
        valid = self.valid
        ids = self.face_indices[valid]
        if location == "faces":
            result[valid] = a[ids]
        else:
            w = self.barycentric[valid]
            if a.ndim == 2:
                w = w[..., None]
            result[valid] = np.sum(w * a[self.mesh.f[ids]], axis=1)
        return result


def locate_chart(mesh: Mesh, uv) -> ChartLocation:
    """Locate arbitrary UV points without re-triangulating, filling holes, or wrapping.

    Overlap checks apply at queried points, not to all intersections of the chart.
    Shared edges/vertices resolve to the first face in the supplied ordering.
    Nonadjacent faces covering the same query are rejected, even if XYZ agrees.
    """
    if not isinstance(mesh, Mesh) or mesh.u is None:
        raise ValueError("an explicit Mesh with a UV chart is required")
    q = points(uv, "uv", 2)
    ids = np.full(len(q), -1, dtype=np.int64)
    weights = np.full((len(q), 3), np.nan)
    eps = 1e-10
    if len(q):
        tree = cKDTree(q)
        for face_id, face in enumerate(mesh.f):
            tri = mesh.u[face]
            lower, upper = tri.min(axis=0), tri.max(axis=0)
            span = upper - lower
            candidates = np.asarray(tree.query_ball_point(lower + span/2,
                                    np.max(span) * (0.5 + eps), p=np.inf), dtype=np.int64)
            if not len(candidates):
                continue
            p = q[candidates]
            keep = np.all((p >= lower - eps*span) & (p <= upper + eps*span), axis=1)
            candidates, p = candidates[keep], p[keep]
            edges = np.column_stack((tri[1] - tri[0], tri[2] - tri[0]))
            ab = np.linalg.solve(edges, (p - tri[0]).T).T
            w = np.column_stack((1 - ab.sum(axis=1), ab))
            inside = np.all(w >= -eps, axis=1)
            candidates, w = candidates[inside], w[inside]
            occupied = ids[candidates] >= 0
            for k in np.flatnonzero(occupied):
                previous = mesh.f[ids[candidates[k]]]
                old_support = set(previous[weights[candidates[k]] > eps])
                new_support = set(face[w[k] > eps])
                if len(old_support) == 3 or len(new_support) == 3 or old_support != new_support:
                    raise ValueError("overlapping chart faces at queried points")
            new = ~occupied
            ids[candidates[new]], weights[candidates[new]] = face_id, w[new]
    ids.setflags(write=False)
    weights.setflags(write=False)
    return ChartLocation(mesh, ids, weights)
