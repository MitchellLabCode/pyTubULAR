"""Centerline geometry and two explicit writhe conventions.

The Gauss integral on an open curve is not the directional polar writhe.
No closure, smoothing, endpoint trimming, or coordinate reflection is implicit.
"""

import numpy as np

from ._coordinates import boolean, points
from .mesh import real_array


def _curve(xyz, closed=False):
    closed = boolean(closed, "closed")
    xyz = points(xyz, "xyz", 3)
    if closed and len(xyz) > 1 and np.array_equal(xyz[0], xyz[-1]):
        xyz = xyz[:-1]
    if len(xyz) < (3 if closed else 2):
        raise ValueError("curve needs at least two points (three if closed)")
    centered = xyz - xyz[0]
    scale = np.max(np.abs(centered))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("curve must have a finite, nonzero extent")
    normalized = centered / scale
    ends = np.roll(normalized, -1, axis=0) if closed else normalized[1:]
    starts = normalized if closed else normalized[:-1]
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 64*np.finfo(float).eps):
        raise ValueError("curve has repeated or numerically indistinguishable neighboring points")
    return xyz, normalized, starts, ends, lengths, scale


def curve_arclength(xyz):
    """Cumulative polygonal path length, starting at zero, in coordinate units."""
    _, _, _, _, ds, scale = _curve(xyz)
    return np.r_[0., np.cumsum(ds)*scale]


def _separation_squared(a, b, c, d):
    """Squared distance from segment a--b to each segment c--d."""
    u, v, w = b-a, d-c, a-c
    aa = np.dot(u, u)
    cc = np.einsum("ij,ij->i", v, v)
    candidates = []
    for p in (a, b):
        t = np.clip(np.einsum("ij,ij->i", p-c, v)/cc, 0, 1)
        delta = p-c-t[:, None]*v
        candidates.append(np.einsum("ij,ij->i", delta, delta))
    for p in (c, d):
        s = np.clip((p-a) @ u/aa, 0, 1)
        delta = p-a-s[:, None]*u
        candidates.append(np.einsum("ij,ij->i", delta, delta))
    distance = np.minimum.reduce(candidates)
    # Cross products avoid cancellation in aa*cc-bb**2 for nearly parallel
    # segments, which can otherwise hide a shallow-angle intersection.
    normal = np.cross(u, v)
    determinant = np.einsum("ij,ij->i", normal, normal)
    stable = determinant > np.finfo(float).tiny
    s = np.zeros_like(determinant)
    t = np.zeros_like(determinant)
    s[stable] = np.einsum("ij,ij->i", np.cross(-w[stable], v[stable]), normal[stable])/determinant[stable]
    t[stable] = np.einsum("ij,ij->i", np.cross(-w[stable], u), normal[stable])/determinant[stable]
    inside = stable & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    delta = w+s[:, None]*u-t[:, None]*v
    interior = np.einsum("ij,ij->i", delta, delta)
    distance[inside] = np.minimum(distance[inside], interior[inside])
    return distance


def _disjoint(starts, ends, lengths, closed):
    """Reject crossings/contacts within 1e-12 of the normalized curve extent."""
    unit = (ends-starts)/lengths[:, None]
    left = unit if closed else unit[:-1]
    right = np.roll(unit, -1, axis=0) if closed else unit[1:]
    if np.any((np.linalg.norm(np.cross(left, right), axis=1) < 1e-12)
              & (np.einsum("ij,ij->i", left, right) < 0)):
        raise ValueError("neighboring curve segments retrace one another")
    for i in range(2, len(starts)):
        ids = np.arange(i-1)
        if closed and i == len(starts)-1:
            ids = ids[ids != 0]
        if len(ids) and np.any(_separation_squared(starts[i], ends[i], starts[ids], ends[ids]) <= 1e-24):
            raise ValueError("nonadjacent curve segments intersect or nearly touch")


def _unit_cross(a, b):
    value = np.cross(a, b)
    size = np.linalg.norm(value, axis=1)
    return np.divide(value, size[:, None], out=np.zeros_like(value), where=size[:, None] > 0)


def writhe_levitt(xyz, *, closed=False):
    """Return (writhe, segment_density) from polygonal segment solid angles.

    Port of writheLevitt.m, with clamped asin inputs and explicit intersection
    checks. Density integrates with segment lengths, and has inverse-length
    units. Closed curves accept either an implicit or repeated closing point.
    """
    _, _, start, end, ds, scale = _curve(xyz, closed)
    _disjoint(start, end, ds, closed)
    omega = np.zeros(len(start))
    for i in range(2, len(start)):
        ids = np.arange(i-1)
        if closed and i == len(start)-1:
            ids = ids[ids != 0]
        if not len(ids):
            continue
        r12, r13, r14 = end[i]-start[i], start[ids]-start[i], end[ids]-start[i]
        r23, r24, r34 = start[ids]-end[i], end[ids]-end[i], end[ids]-start[ids]
        n1, n2 = _unit_cross(r13, r14), _unit_cross(r14, r24)
        n3, n4 = _unit_cross(r24, r23), _unit_cross(r23, r13)
        angles = np.zeros(len(ids))
        for a, b in ((n1, n2), (n2, n3), (n3, n4), (n1, n4)):
            angles += np.arcsin(np.clip(np.einsum("ij,ij->i", a, b), -1, 1))
        cross = np.cross(r34, r12)
        triple = np.einsum("ij,ij->i", cross, r13)
        # Planar pairs have exactly zero contribution; avoid assigning a sign
        # to roundoff from rigidly rotating a planar curve.
        coplanar = np.abs(triple) <= 64*np.finfo(float).eps*np.linalg.norm(cross, axis=1)*np.linalg.norm(r13, axis=1)
        signed = angles*np.sign(triple)
        signed[coplanar] = 0
        omega[i] += signed.sum()
        omega[ids] += signed
    contribution = omega/(4*np.pi)
    return float(contribution.sum()), contribution/(ds*scale)


def writhe_gauss(xyz, ss=None, *, legacy=False):
    """Return (writhe, vertex_density) by open-curve Gauss quadrature.

    Matches the original gradient/tangent quadrature. Corrected normalization
    is 1/(4*pi); legacy=True additionally multiplies by 3/(2*pi), reproducing
    the original duplicated numerator and final extra division. This sampled
    integral is an approximation, distinct from the polygon solid-angle result.
    """
    legacy = boolean(legacy, "legacy")
    xyz, normalized, start, end, ds, scale = _curve(xyz)
    _disjoint(start, end, ds, False)
    if ss is None:
        s = np.r_[0., np.cumsum(ds)]
    else:
        ss = real_array(ss, "ss", ndim=1)
        if len(ss) != len(xyz) or np.any(np.diff(ss) <= 0):
            raise ValueError("ss must match curve points and be strictly increasing")
        s = (ss-ss[0])/scale
    quadrature = np.gradient(s)
    gradient = np.gradient(normalized, axis=0)
    norm = np.linalg.norm(gradient, axis=1)
    if np.any(norm <= 64*np.finfo(float).eps):
        raise ValueError("curve has an undefined centered tangent")
    tangent = gradient/norm[:, None]
    density = np.zeros(len(xyz))
    for i in range(len(xyz)):
        ids = np.arange(len(xyz)) != i
        delta = normalized[i]-normalized[ids]
        distance = np.linalg.norm(delta, axis=1)
        numerator = np.einsum("ij,ij->i", np.cross(tangent[i], tangent[ids]), delta)
        density[i] = np.sum(numerator/distance**3*quadrature[ids])/(4*np.pi)
    if legacy:
        density *= 3/(2*np.pi)
    return float(density @ quadrature), density/scale
