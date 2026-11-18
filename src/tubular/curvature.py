"""Rusinkiewicz mesh curvature, adapted from Itzik Ben Shabat's MATLAB code.

Copyright (c) 2016, Itzik Ben Shabat. BSD terms: licenses/curvature-BSD.txt.
TubULAR's mean-curvature convention is H = -(k1+k2)/2 for this normal derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, require_oriented_manifold


@dataclass(frozen=True)
class CurvatureResult:
    """Vertex estimates in input vertex order; all arrays are read-only.

    Principal values are ordered by decreasing absolute magnitude, as in
    GetCurvatures. Directions have shape (N, 2, 3), with arbitrary signs and
    arbitrary axes at repeated eigenvalues. The shape operator is ambient
    (N, 3, 3), restricted to the estimated tangent plane. Boundary estimates
    use only supplied incident triangles; no seam or endcap is inferred.
    """

    principal_curvatures: np.ndarray
    principal_directions: np.ndarray
    mean: np.ndarray
    gaussian: np.ndarray
    normals: np.ndarray
    shape_operator: np.ndarray
    vertex_areas: np.ndarray
    corner_areas: np.ndarray
    boundary_vertices: np.ndarray
    length_unit: str


def _unit(vectors, name, *, reference=None):
    norms = np.linalg.norm(vectors, axis=-1)
    threshold = 64*np.finfo(float).eps
    if reference is not None:
        threshold = threshold*reference
    if np.any(norms <= threshold) or not np.isfinite(norms).all():
        raise ValueError(f"{name} is undefined or numerically singular")
    return vectors/norms[..., None]


def estimate_curvature(mesh: Mesh) -> CurvatureResult:
    """Estimate per-vertex principal, mean, and Gaussian mesh curvature.

    Uses Max-weighted vertex normals, mixed corner areas, per-face least
    squares fits to normal differences, and transported vertex tensors.
    Requires an oriented manifold; open boundaries are allowed and flagged.
    Rejects canceled normals and ambiguous antiparallel frame transport.
    Derivatives of curvature and mesh smoothing are not implemented.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh")
    require_oriented_manifold(mesh)
    f = mesh.faces
    # Work at order-one scale; restore curvature and area units at the end.
    xyz = mesh.vertices-mesh.vertices[0]
    scale = np.max(np.abs(xyz))
    xyz = xyz/scale
    tri = xyz[f]
    edges = tri[:, [2, 0, 1]]-tri[:, [1, 2, 0]]
    l2 = np.einsum("fij,fij->fi", edges, edges)
    cross = np.cross(edges[:, 0], edges[:, 1])
    area2 = np.linalg.norm(cross, axis=1)
    nf = cross/area2[:, None]
    area = area2/2
    edge_unit = edges/np.sqrt(l2)[..., None]

    ew = l2*(np.roll(l2, 1, axis=1)+np.roll(l2, -1, axis=1)-l2)
    corner = np.empty_like(l2)
    acute = np.all(ew > 0, axis=1)
    corner[acute] = (0.5*area[acute]/ew[acute].sum(axis=1))[:, None]*(
        np.roll(ew[acute], 1, axis=1)+np.roll(ew[acute], -1, axis=1))
    for i in range(3):
        mask = ~acute & (np.argmin(ew, axis=1) == i)
        j, k = (i+1) % 3, (i+2) % 3
        a = edges[mask]
        corner[mask, j] = -0.25*l2[mask, k]*area[mask]/np.einsum("ij,ij->i", a[:, i], a[:, k])
        corner[mask, k] = -0.25*l2[mask, j]*area[mask]/np.einsum("ij,ij->i", a[:, i], a[:, j])
        corner[mask, i] = area[mask]-corner[mask, j]-corner[mask, k]
    if np.any(corner <= 0) or not np.isfinite(corner).all():
        raise ValueError("mixed corner areas are not positive and finite")
    n = len(xyz)
    vertex_area = np.bincount(f.ravel(), weights=corner.ravel(), minlength=n)
    weights = area[:, None]/(np.roll(l2, 1, axis=1)*np.roll(l2, -1, axis=1))
    normals = np.zeros((n, 3))
    np.add.at(normals, f.ravel(), (weights[..., None]*nf[:, None, :]).reshape(-1, 3))
    normal_weights = np.bincount(f.ravel(), weights=weights.ravel(), minlength=n)
    normals = _unit(normals, "vertex normal", reference=normal_weights)

    # Match the source's last-incident-face seed, with deterministic row order.
    last = np.full(n, -1, dtype=int)
    np.maximum.at(last, f.ravel(), np.arange(f.size))
    seed = edge_unit[:, [2, 0, 1], :].reshape(-1, 3)[last]
    up = _unit(np.cross(seed, normals), "vertex tangent frame")
    vp = np.cross(normals, up)
    basis = np.stack((up, vp), axis=1)
    tangent = edge_unit[:, 0]
    bitangent = _unit(np.cross(nf, tangent), "face tangent frame")
    face_basis = np.stack((tangent, bitangent), axis=1)
    ab = np.einsum("fij,fkj->fik", edges, face_basis)
    system = np.zeros((len(f), 6, 3))
    system[:, 0::2, :2] = ab
    system[:, 1::2, 1:] = ab
    differences = normals[f[:, [2, 0, 1]]]-normals[f[:, [1, 2, 0]]]
    rhs = np.einsum("fij,fkj->fik", differences, face_basis).reshape(-1, 6)
    # QR avoids squaring the condition number through normal equations.
    q, r = np.linalg.qr(system, mode="reduced")
    if np.any(np.linalg.cond(r) > 1e12):
        raise ValueError("curvature fit is numerically singular; inspect triangle quality")
    fit = np.linalg.solve(r, np.einsum("fji,fj->fi", q, rhs)[..., None])[..., 0]
    face_tensor = fit[:, [0, 1, 1, 2]].reshape(-1, 2, 2)

    tensor = np.zeros((n, 2, 2))
    for corner_id in range(3):
        ids = f[:, corner_id]
        vn = normals[ids]
        dot = np.einsum("ij,ij->i", nf, vn)
        if np.any(1+dot <= 1e-12):
            raise ValueError("antiparallel vertex/face normals make tensor transport ambiguous")
        perp = nf-dot[:, None]*vn
        dperp = (vn+nf)/(1+dot[:, None])
        transported = basis[ids]-np.einsum("fij,fj->fi", basis[ids], perp)[..., None]*dperp[:, None, :]
        change = np.einsum("fij,fkj->fik", transported, face_basis)
        projected = change @ face_tensor @ change.swapaxes(1, 2)
        np.add.at(tensor, ids, projected*(corner[:, corner_id]/vertex_area[ids])[:, None, None])
    tensor = (tensor+tensor.swapaxes(1, 2))/2
    k, eig = np.linalg.eigh(tensor)
    order = np.argsort(-np.abs(k), axis=1, kind="stable")
    k = np.take_along_axis(k, order, axis=1)/scale
    eig = np.take_along_axis(eig, order[:, None, :], axis=2)
    d1 = np.einsum("ni,nij->nj", eig[:, :, 0], basis)
    directions = np.stack((d1, np.cross(normals, d1)), axis=1)
    operator = np.einsum("ni,nij,nik->njk", k, directions, directions)
    boundary = np.zeros(n, dtype=bool)
    edges_idx = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges_idx, axis=0, return_counts=True)
    boundary[unique[counts == 1].ravel()] = True
    values = (k, directions, -k.mean(axis=1), k.prod(axis=1), normals, operator,
              vertex_area*scale**2, corner*scale**2, boundary)
    for value in values:
        if not np.isfinite(value).all():
            raise ValueError("curvature output overflowed; rescale input coordinates")
        value.setflags(write=False)
    return CurvatureResult(*values, mesh.length_unit)
