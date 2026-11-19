"""Explicit annular seam cutting and the constrained flattenAnnulus solve.

The boundary constraints follow Dillon Cislo's TubULAR flattenAnnulus.m
(MIT). The mesh cutting uses local vertex fans, retaining every original
face and its winding. No automatic path selection or mesh repair is done.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from ._coordinates import scalar
from .mesh import Mesh, require_oriented_manifold


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _indices(value, name, n, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"{name} must contain zero-based integer vertex indices")
    return np.array(a, dtype=np.int64, copy=True)


def _incidence(mesh):
    edges = {}
    incident = [[] for _ in mesh.v]
    for face_id, face in enumerate(mesh.f):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((a, b))), []).append((face_id, a, b))
            incident[a].append(face_id)
    return edges, incident


def _boundary_loops(edges):
    successors = {a: b for entries in edges.values() if len(entries) == 1
                  for _, a, b in entries}
    loops = []
    while successors:
        start = next(iter(successors))
        loop, vertex = [], start
        while True:
            loop.append(vertex)
            vertex = successors.pop(vertex)
            if vertex == start:
                break
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


@dataclass(frozen=True, eq=False)
class SeamCut:
    """Topology change with explicit zero-based vertex correspondence.

    Face order and face corner order are unchanged. ``cut_to_uncut[mesh.f]``
    reproduces the source faces exactly. Vertex fields transfer by indexing
    with cut_to_uncut; uncut_to_cut gives the one or two copies of each vertex.
    path_pairs[:, 0] retains the left side of the directed seam, while column
    1 contains appended right-side duplicates. The old UV chart is discarded.
    """

    mesh: Mesh
    path_pairs: np.ndarray
    cut_to_uncut: np.ndarray
    uncut_to_cut: tuple


def cut_annulus(mesh: Mesh, seam_path) -> SeamCut:
    """Cut an oriented annulus along a supplied simple edge path.

    Endpoints must lie on different boundary loops. All intermediate vertices
    must be interior. Input geometry and connectivity are never repaired.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh")
    topo = require_oriented_manifold(mesh)
    edges, incident = _incidence(mesh)
    loops = _boundary_loops(edges)
    if topo.components != 1 or topo.euler_characteristic != 0 or len(loops) != 2:
        raise ValueError("mesh must be a connected topological annulus with two boundaries")
    path = _indices(seam_path, "seam_path", len(mesh.v), 1)
    if len(path) < 2 or len(np.unique(path)) != len(path):
        raise ValueError("seam_path must contain at least two distinct vertices with no repeats")
    boundary = set(np.concatenate(loops))
    if path[0] not in boundary or path[-1] not in boundary:
        raise ValueError("seam endpoints must lie on the boundary")
    if any(path[0] in loop and path[-1] in loop for loop in loops):
        raise ValueError("seam endpoints must lie on different boundary loops")
    if any(v in boundary for v in path[1:-1]):
        raise ValueError("intermediate seam vertices must be interior")
    path_edges = [tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])]
    if any(e not in edges or len(edges[e]) != 2 for e in path_edges):
        raise ValueError("seam must follow interior mesh edges")
    blocked = set(path_edges)
    faces = mesh.f.copy()
    # Split each seam vertex's incident fan locally. Global flood filling of
    # path-adjacent faces can confuse the sides when a path runs close to itself.
    for i, vertex in enumerate(path):
        fan = set(incident[vertex])
        adjacency = {f: set() for f in fan}
        for f in fan:
            for neighbor in mesh.f[f]:
                if neighbor == vertex:
                    continue
                edge = tuple(sorted((vertex, neighbor)))
                if edge not in blocked:
                    adjacency[f].update(g for g, _, _ in edges[edge] if g != f)
        a, b = (path[i], path[i+1]) if i < len(path)-1 else (path[i-1], path[i])
        seed = next(f for f, x, y in edges[tuple(sorted((a, b)))] if (x, y) == (b, a))
        right, stack = set(), [seed]
        while stack:
            f = stack.pop()
            if f not in right:
                right.add(f)
                stack.extend(adjacency[f] - right)
        if not right or right == fan:
            raise ValueError("seam does not split the incident vertex fan")
        for f in right:
            faces[f, mesh.f[f] == vertex] = len(mesh.v) + i
    mapping = np.concatenate((np.arange(len(mesh.v)), path))
    pairs = np.column_stack((path, np.arange(len(mesh.v), len(mapping))))
    cut = Mesh(mesh.v[mapping], faces, length_unit=mesh.length_unit)
    cut_topo = require_oriented_manifold(cut)
    if (cut_topo.components != 1 or cut_topo.euler_characteristic != 1
            or len(_boundary_loops(_incidence(cut)[0])) != 1):
        raise ValueError("seam did not produce a topological disk")
    copies = [[i] for i in range(len(mesh.v))]
    for a, b in pairs:
        copies[a].append(b)
    return SeamCut(cut, _readonly(pairs), _readonly(mapping),
                   tuple(_readonly(ids) for ids in copies))


def _annular_boundary(mesh, path_pairs):
    topo = require_oriented_manifold(mesh)
    loops = _boundary_loops(_incidence(mesh)[0])
    if topo.components != 1 or topo.euler_characteristic != 1 or len(loops) != 1:
        raise ValueError("cut mesh must be a connected topological disk")
    pairs = _indices(path_pairs, "path_pairs", len(mesh.v), 2)
    if pairs.shape[1] != 2 or len(pairs) < 2 or len(np.unique(pairs)) != pairs.size:
        raise ValueError("path_pairs must have shape (K, 2), K >= 2, with distinct vertices")
    # Seam duplicates must be exact: cut_annulus creates exact copies. Imported
    # nearly coincident seams require explicit reconciliation by the caller.
    if not np.array_equal(mesh.v[pairs[:, 0]], mesh.v[pairs[:, 1]]):
        raise ValueError("paired seam vertices must have identical positions")
    loop = loops[0]
    if not np.isin(pairs, loop).all():
        raise ValueError("all path_pairs vertices must lie on the disk boundary")
    start = int(np.flatnonzero(loop == pairs[0, 0])[0])
    loop = np.roll(loop, -start)
    if np.flatnonzero(loop == pairs[-1, 0])[0] < np.flatnonzero(loop == pairs[0, 1])[0]:
        loop = np.concatenate((loop[:1], loop[:0:-1]))
    corners = [0] + [int(np.flatnonzero(loop == v)[0])
                     for v in (pairs[0, 1], pairs[-1, 1], pairs[-1, 0])]
    if np.any(np.diff(corners) <= 0):
        raise ValueError("seam endpoints have inconsistent boundary order")
    sides = [loop[a:b+1] for a, b in zip(corners[:-1], corners[1:])]
    sides.append(np.concatenate((loop[corners[-1]:], loop[:1])))
    if not np.array_equal(sides[1], pairs[:, 1]) or not np.array_equal(sides[3][::-1], pairs[:, 0]):
        raise ValueError("path_pairs must enumerate both complete seam boundary chains in order")
    return pairs, loops[0], sides


def _cotangent_laplacian(mesh, negative_weights):
    if negative_weights not in ("clamp", "error"):
        raise ValueError("negative_weights must be 'clamp' or 'error'")
    tri = mesh.v[mesh.f]
    rows, cols, values = [], [], []
    # Standard FEM half-cotangent weights, evaluated with local scaling to
    # avoid squared-length overflow. This does not import a geometry library.
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a, b = tri[:, j]-tri[:, i], tri[:, k]-tri[:, i]
        scale = np.maximum(np.max(np.abs(a), axis=1), np.max(np.abs(b), axis=1))
        a, b = a/scale[:, None], b/scale[:, None]
        weight = np.einsum("ij,ij->i", a, b) / (2*np.linalg.norm(np.cross(a, b), axis=1))
        rows.extend((mesh.f[:, j], mesh.f[:, k]))
        cols.extend((mesh.f[:, k], mesh.f[:, j]))
        values.extend((weight, weight))
    n = len(mesh.v)
    weights = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n)).tocsr()
    negative = weights.data < 0
    count = int(np.count_nonzero(negative)//2)
    if count and negative_weights == "error":
        raise ValueError(f"cotangent Laplacian has {count} negative edge weights")
    # Exact policy in flattenAnnulus: replace negative assembled edge weights
    # with 0.01, including boundary edges, then recompute the diagonal.
    weights.data[negative] = 0.01
    laplacian = weights - sparse.diags(np.asarray(weights.sum(axis=1)).ravel())
    return laplacian, count


def _check_embedding(mesh, boundary):
    uv = mesh.uv
    tri = uv[mesh.f]
    a, b = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    det = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if not (np.all(det > 0) or np.all(det < 0)):
        raise ValueError("parameterization has flipped triangles")
    # Local orientation alone is insufficient. A disk embedding also needs a
    # simple boundary; reject nonadjacent crossings AND contacts. Bounding-box
    # rejection keeps the vectorized segment check small on typical meshes.
    points = uv[boundary]
    points = (points-points.min(axis=0))/max(np.ptp(points, axis=0))
    end = np.roll(points, -1, axis=0)
    low, high = np.minimum(points, end), np.maximum(points, end)
    eps = 256*np.finfo(float).eps
    cross = lambda x, y: x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    tangent = end-points
    following = np.roll(tangent, -1, axis=0)
    if np.any((np.abs(cross(tangent, following)) <= eps)
              & (np.einsum("ij,ij->i", tangent, following) < 0)):
        raise ValueError("parameterization boundary backtracks along itself")
    for i in range(len(points)):
        js = np.arange(i+2, len(points)-(1 if i == 0 else 0))
        js = js[np.all(high[js] >= low[i]-eps, axis=1)
                & np.all(low[js] <= high[i]+eps, axis=1)]
        if not len(js):
            continue
        a, b, c, d = points[i], end[i], points[js], end[js]
        o1, o2 = cross(b-a, c-a), cross(b-a, d-a)
        o3, o4 = cross(d-c, a-c), cross(d-c, b-c)
        straddle1 = (np.minimum(o1, o2) <= eps) & (np.maximum(o1, o2) >= -eps)
        straddle2 = (np.minimum(o3, o4) <= eps) & (np.maximum(o3, o4) >= -eps)
        if np.any(straddle1 & straddle2):
            raise ValueError("parameterization boundary intersects or touches itself")
    return 1 if det[0] > 0 else -1


@dataclass(frozen=True, eq=False)
class AnnularEmbedding:
    """A solved chart and diagnostics; UV and seam indices are read-only.

    The boundary rings lie on u=0 and u=aspect_ratio. Paired seam vertices
    differ by (0, 1), first minus second. The seams can curve: this is a
    periodic fundamental domain, not necessarily a rectangular cut polygon.
    orientation is the common UV signed-area sign relative to face winding.
    Residuals refer to the unit-width solve before explicit lateral scaling.
    """

    mesh: Mesh
    path_pairs: np.ndarray
    clamped_edges: int
    linear_residual: float
    constraint_residual: float
    orientation: int


def flatten_annulus(mesh: Mesh, path_pairs, *, aspect_ratio=1.0,
                    negative_weights="clamp") -> AnnularEmbedding:
    """Port the Dirichlet branch of TubULAR's flattenAnnulus on a cut annulus.

    Boundary rings use normalized 3D edge arclength. Interior seam pairs have
    equal u and v differing by one; their positions are otherwise free. Solve
    the sparse constrained cotangent system, then check residuals, triangle
    degeneracy/orientation and boundary intersections before returning a chart.

    ``clamp`` reproduces MATLAB's 0.01 replacement of negative edge weights;
    the result reports their count. ``error`` rejects such meshes. MVC, Ricci
    flow, automatic aspect-ratio optimization and reflected charts are outside
    this function's scope. Existing UV coordinates do not enter the solve.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a Mesh")
    width = scalar(aspect_ratio, "aspect_ratio", positive=True)
    pairs, boundary, sides = _annular_boundary(mesh, path_pairs)
    laplacian, clamped = _cotangent_laplacian(mesh, negative_weights)
    rows, cols, data, rhs = [], [], [], []

    def constrain(indices, coefficients, value):
        rows.extend([len(rhs)]*len(indices))
        cols.extend(indices)
        data.extend(coefficients)
        rhs.append(value)

    for side, x, descending in ((sides[0], 0, True), (sides[2], 1, False)):
        edges = np.diff(mesh.v[side], axis=0)
        edges /= np.max(np.abs(edges))
        lengths = np.linalg.norm(edges, axis=1)
        fraction = np.concatenate(([0.], np.cumsum(lengths)/sum(lengths)))
        for vertex, y in zip(side, 1-fraction if descending else fraction):
            constrain([2*vertex], [1], x)
            constrain([2*vertex+1], [1], y)
    for a, b in pairs[1:-1]:
        constrain([2*a, 2*b], [1, -1], 0)
        constrain([2*a+1, 2*b+1], [1, -1], 1)
    n = 2*len(mesh.v)
    constraint = sparse.coo_matrix((data, (rows, cols)), shape=(len(rhs), n)).tocsr()
    operator = sparse.kron(laplacian, sparse.eye(2), format="csr")
    system = sparse.bmat([[operator, constraint.T], [constraint, None]], format="csc")
    target = np.concatenate((np.zeros(n), rhs))
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            solution = spsolve(system, target)
        except (MatrixRankWarning, RuntimeError) as exc:
            raise ValueError("parameterization linear system is singular") from exc
    residual = float(np.max(np.abs(system @ solution-target)))
    constraint_residual = float(np.max(np.abs(constraint @ solution[:n]-rhs)))
    if not np.isfinite(solution).all() or residual > 1e-6 or constraint_residual > 1e-10:
        raise ValueError("parameterization linear system failed the residual check")
    uv = solution[:n].reshape(-1, 2).copy()
    unit_mesh = Mesh(mesh.v, mesh.f, uv, mesh.length_unit)
    orientation = _check_embedding(unit_mesh, boundary)
    uv[:, 0] *= width
    mapped = Mesh(mesh.v, mesh.f, uv, mesh.length_unit)
    return AnnularEmbedding(mapped, _readonly(pairs), clamped, residual,
                            constraint_residual, orientation)


@dataclass(frozen=True, eq=False)
class ParameterizedCutMesh:
    """The topology change and embedding returned by generateCurrentCutMesh."""

    cut: SeamCut
    embedding: AnnularEmbedding

    @property
    def mesh(self):
        return self.embedding.mesh

    @property
    def path_pairs(self):
        return self.cut.path_pairs

    @property
    def cut_to_uncut(self):
        return self.cut.cut_to_uncut

    @property
    def uncut_to_cut(self):
        return self.cut.uncut_to_cut
