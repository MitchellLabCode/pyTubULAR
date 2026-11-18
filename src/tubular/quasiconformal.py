"""Per-face Beltrami coefficients of supplied maps (TubULAR bc_metric.m)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, nondegenerate, real_array, require_oriented_manifold


@dataclass(frozen=True)
class BeltramiResult:
    """Read-only face fields in source face order; both are dimensionless.

    mu is complex. Dilatation is the ratio of the Jacobian's largest to
    smallest singular value, not a measure of isotropic growth or area change.
    """
    mu: np.ndarray
    dilatation: np.ndarray
    target_dimension: int


def beltrami_coefficient(reference: Mesh, mapped_coordinates=None) -> BeltramiResult:
    """Measure the piecewise-affine map from reference.uv to supplied vertices.

    mapped_coordinates has shape (N, 2) or (N, 3) in reference vertex order;
    omitted coordinates use reference.vertices. No correspondence is inferred.
    Planar maps must preserve orientation on every face. Surface maps use the
    induced metric, which cannot detect a spatial reflection or self-intersection.
    Rank-deficient maps and singular-value ratios above 1e12 are rejected.
    """
    if not isinstance(reference, Mesh):
        raise TypeError("reference must be a Mesh")
    require_oriented_manifold(reference)
    if reference.uv is None:
        raise ValueError("reference requires an explicit UV chart")
    target = real_array(reference.vertices if mapped_coordinates is None else mapped_coordinates,
                        "mapped_coordinates", ndim=2)
    if len(target) != len(reference.v) or target.shape[1] not in (2, 3):
        raise ValueError("mapped_coordinates must have shape (N, 2) or (N, 3) in reference vertex order")
    nondegenerate(target, reference.f, "mapped_coordinates")
    uv, xyz = reference.uv[reference.f], target[reference.f]
    domain = uv[:, 1:]-uv[:, :1]
    image = xyz[:, 1:]-xyz[:, :1]
    j = np.linalg.solve(domain, image).swapaxes(1, 2)
    # Uniform per-face scale does not affect mu and prevents squared overflow.
    j /= np.max(np.abs(j), axis=(1, 2))[:, None, None]
    singular = np.linalg.svd(j, compute_uv=False)
    if np.any(singular[:, 1] <= 1e-12*singular[:, 0]):
        raise ValueError("map is numerically singular; inspect distorted or collapsed triangles")
    du, dv = j[:, :, 0], j[:, :, 1]
    if target.shape[1] == 2:
        det = du[:, 0]*dv[:, 1]-du[:, 1]*dv[:, 0]
        if np.any(det <= 0):
            raise ValueError("planar map reverses orientation or folds a triangle")
        dz = (du[:, 0]+dv[:, 1])+1j*(du[:, 1]-dv[:, 0])
        dbar = (du[:, 0]-dv[:, 1])+1j*(du[:, 1]+dv[:, 0])
        mu = dbar/dz
    else:
        e = np.einsum("ij,ij->i", du, du)
        g = np.einsum("ij,ij->i", dv, dv)
        f = np.einsum("ij,ij->i", du, dv)
        area = np.linalg.norm(np.cross(du, dv), axis=1)
        mu = (e-g+2j*f)/(e+g+2*area)
    dilation = singular[:, 0]/singular[:, 1]
    if not np.isfinite(mu).all() or np.any(np.abs(mu) >= 1):
        raise ValueError("Beltrami coefficient is outside the numerically resolved range")
    mu.setflags(write=False)
    dilation.setflags(write=False)
    return BeltramiResult(mu, dilation, target.shape[1])
