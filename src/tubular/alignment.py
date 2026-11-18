"""Landmark frames and mesh alignment, independent of experiment metadata or UI."""

from __future__ import annotations

import numpy as np

from ._coordinates import boolean, rotation, scalar
from .mesh import Mesh, real_array


def landmark_frame(anterior, posterior, dorsal, *, origin=None):
    """Return (rotation, translation) for an explicit APDV landmark frame.

    Landmarks and optional origin are length-three XYZ arrays in the same
    coordinates as the source mesh. Origin defaults to anterior. +X points
    from origin to posterior; +Z points toward the component of dorsal-origin
    perpendicular to X; Y completes a right-handed frame. The transform is
    xyz @ rotation.T + translation. No scale, mirror, or landmark detection.

    This constructs the frame in computeAPDVCoords.m directly, avoiding that
    source's singular Rodrigues formula when AP points along negative X.
    """
    values = []
    for name, value in (("anterior", anterior), ("posterior", posterior),
                        ("dorsal", dorsal), ("origin", anterior if origin is None else origin)):
        point = real_array(value, name, ndim=1)
        if point.shape != (3,):
            raise ValueError(f"{name} must contain three XYZ coordinates")
        values.append(point)
    _, p, d, o = values
    ap, dv = p-o, d-o
    # Normalize each vector separately: landmark distances need not be similar.
    ap_scale, dv_scale = np.max(np.abs(ap)), np.max(np.abs(dv))
    if ap_scale == 0:
        raise ValueError("posterior must differ from the frame origin")
    if dv_scale == 0:
        raise ValueError("dorsal must lie off the anterior-posterior axis")
    x = ap/ap_scale
    x /= np.linalg.norm(x)
    dv = dv/dv_scale
    # The cross product avoids subtractive projection cancellation near the axis.
    y = np.cross(dv, x)
    y_norm = np.linalg.norm(y)
    if y_norm <= 1e-12*np.linalg.norm(dv):
        raise ValueError("dorsal must lie off the anterior-posterior axis (near-collinear landmarks)")
    y /= y_norm
    z = np.cross(x, y)
    rot = np.stack((x, y, z))
    return rot, -(rot @ o)


def align_mesh(mesh: Mesh, rot, trans, *, resolution=1, flipy=False,
               preserve_orientation=True, length_unit=None) -> Mesh:
    """Apply an APDV similarity transform without changing vertex/face IDs.

    Reflection reverses triangle winding by swapping corners 1 and 2 (zero
    based) when preserve_orientation=True, retaining signed volume and outward
    or inward normals. False keeps face arrays exactly, as in MATLAB, so a
    reflection reverses signed orientation. UV values stay attached to vertices.
    Labels do not convert units: resolution is the explicit coordinate scale.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh")
    r = rotation(rot)
    t = real_array(trans, "trans", ndim=1)
    if t.shape != (3,):
        raise ValueError("trans must contain three coordinates")
    scale = scalar(resolution, "resolution", positive=True)
    flip = boolean(flipy, "flipy")
    preserve = boolean(preserve_orientation, "preserve_orientation")
    vertices = (mesh.vertices @ r.T + t)*scale
    faces = mesh.faces
    if flip:
        vertices[:, 1] *= -1
        if preserve:
            faces = faces[:, [0, 2, 1]]
    return Mesh(vertices, faces, mesh.uv,
                mesh.length_unit if length_unit is None else length_unit)
