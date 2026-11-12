"""Point sampling and pullbacks on supplied charts; no surface fitting implied."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .mesh import Mesh, real_array


def sample_volume(volume, points_xyz, *, spacing_xyz=(1, 1, 1), origin_xyz=(0, 0, 0)):
    """Trilinear sample of a scalar volume stored in (z, y, x) order.

    origin is the physical position of voxel [0,0,0]'s center. spacing and
    points must use the same physical units. Out-of-volume samples are NaN.
    Returns float64 even for integer images (no interpolation rounding).
    """
    raw = np.asarray(volume)
    if raw.ndim != 3 or raw.dtype.kind not in "iuf" or min(raw.shape) < 1:
        raise ValueError("volume must be a nonempty real scalar (z, y, x) array")
    points = real_array(points_xyz, "points_xyz", ndim=2)
    spacing = real_array(spacing_xyz, "spacing_xyz", ndim=1)
    origin = real_array(origin_xyz, "origin_xyz", ndim=1)
    if points.shape[1] != 3 or spacing.shape != (3,) or origin.shape != (3,):
        raise ValueError("points, origin, and spacing must have three xyz components")
    if np.any(spacing <= 0):
        raise ValueError("spacing must be positive")
    indices = (points - origin) / spacing
    return map_coordinates(raw, indices[:, ::-1].T, output=np.float64,
                           order=1, mode="constant", cval=np.nan, prefilter=False)


@dataclass(frozen=True)
class Pullback:
    image: np.ndarray
    xyz: np.ndarray
    face_indices: np.ndarray

    @property
    def valid(self):
        return (self.face_indices >= 0) & np.isfinite(self.image)


def pullback_image(mesh: Mesh, volume, u, v, *, spacing_xyz=(1, 1, 1),
                   origin_xyz=(0, 0, 0), progress=None, cancelled=None) -> Pullback:
    """Sample a supplied chart at u columns and v rows, preserving its triangles.

    No re-triangulation, periodic wrapping, extrapolation, or normal-stack
    projection occurs. Holes stay NaN. Overlapping faces encountered at sample
    locations raise an error (this is not a global intersection test).
    progress(done, total) and cancelled() are optional GUI-independent hooks.
    """
    if mesh.uv is None:
        raise ValueError("pullback requires an explicit uv chart")
    axes = []
    for values, name in ((u, "u"), (v, "v")):
        values = real_array(values, name, ndim=1)
        if len(values) == 0 or np.any(np.diff(values) <= 0):
            raise ValueError(f"{name} must be nonempty and strictly increasing")
        axes.append(values)
    u, v = axes
    shape = (len(v), len(u))
    xyz = np.full((*shape, 3), np.nan)
    face_ids = np.full(shape, -1, dtype=np.int64)
    interior = np.zeros(shape, dtype=bool)
    eps = 1e-10
    for face_id, face in enumerate(mesh.faces):
        if cancelled is not None and cancelled():
            raise InterruptedError("pullback cancelled")
        triangle = mesh.uv[face]
        lower, upper = triangle.min(axis=0), triangle.max(axis=0)
        # A conservative bounding box keeps floating point edge points included.
        padding = eps * (upper - lower)
        cols = np.flatnonzero((u >= lower[0] - padding[0]) & (u <= upper[0] + padding[0]))
        rows = np.flatnonzero((v >= lower[1] - padding[1]) & (v <= upper[1] + padding[1]))
        if len(cols) and len(rows):
            cc, rr = np.meshgrid(cols, rows)
            q = np.column_stack((u[cc.ravel()], v[rr.ravel()]))
            edges = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
            ab = np.linalg.solve(edges, (q - triangle[0]).T).T
            weights = np.column_stack((1 - ab.sum(axis=1), ab))
            inside = np.all(weights >= -eps, axis=1)
            r, c, w = rr.ravel()[inside], cc.ravel()[inside], weights[inside]
            positions = w @ mesh.vertices[face]
            strict = np.all(w > eps, axis=1)
            occupied = face_ids[r, c] >= 0
            if np.any(occupied):
                # Shared edges agree in 3D; overlap interiors are always invalid.
                agreement = np.all(np.isclose(xyz[r, c], positions, rtol=1e-10, atol=1e-12), axis=1)
                if np.any(occupied & (strict | interior[r, c] | ~agreement)):
                    raise ValueError("overlapping or inconsistent chart faces at pullback samples")
            new = ~occupied
            xyz[r[new], c[new]] = positions[new]
            face_ids[r[new], c[new]] = face_id
            interior[r[new], c[new]] = strict[new]
        if progress is not None:
            progress(face_id + 1, len(mesh.faces))
    if cancelled is not None and cancelled():
        raise InterruptedError("pullback cancelled")
    image = np.full(shape, np.nan)
    covered = face_ids >= 0
    image[covered] = sample_volume(volume, xyz[covered], spacing_xyz=spacing_xyz, origin_xyz=origin_xyz)
    return Pullback(image, xyz, face_ids)
