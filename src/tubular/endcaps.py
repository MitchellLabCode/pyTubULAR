"""Explicit endcap removal and edge-constrained seam selection.

Adapts the custom selection in N. P. Mitchell's sliceMeshEndcaps and the index
compaction of remove_vertex_from_mesh (Dillon Cislo, N. P. Mitchell; MIT).
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .mesh import Mesh, require_oriented_manifold
from .parameterization import _indices, _incidence, _boundary_loops, _readonly


@dataclass(frozen=True, eq=False)
class OpenedSurface:
    """An annulus and exact source IDs; source_to_open is -1 for removed vertices."""

    mesh: Mesh
    vertex_sources: np.ndarray
    face_sources: np.ndarray
    source_to_open: np.ndarray
    boundary_loops: tuple


def remove_endcaps(mesh, anterior_vertices, posterior_vertices) -> OpenedSurface:
    """Remove specified cap vertices and incident faces from a closed sphere mesh.

    This adapts sliceMeshEndcaps' custom vertex selection branch. No geometric
    plane cutting, component deletion, new vertices, smoothing or cap inference
    occurs. The retained surface must be one oriented annulus. Unreferenced
    vertices are compacted by index, avoiding MATLAB's coordinate-based matching
    of coincident but topologically distinct vertices. Output UV is absent.
    """
    topo = require_oriented_manifold(mesh)
    if not topo.closed or topo.components != 1 or topo.euler_characteristic != 2:
        raise ValueError("endcap removal requires one closed genus-zero surface")
    a = _indices(anterior_vertices,"anterior_vertices",len(mesh.v),1)
    p = _indices(posterior_vertices,"posterior_vertices",len(mesh.v),1)
    if (not len(a) or not len(p) or len(np.unique(a)) != len(a)
            or len(np.unique(p)) != len(p) or np.intersect1d(a,p).size):
        raise ValueError("endcaps must be nonempty disjoint sets without repeated vertices")
    removed = np.zeros(len(mesh.v),dtype=bool)
    removed[np.r_[a,p]] = True
    face_ids = np.flatnonzero(~removed[mesh.f].any(axis=1))
    if not len(face_ids):
        raise ValueError("endcap selection removes every face")
    vertex_ids = np.unique(mesh.f[face_ids])
    mapping = np.full(len(mesh.v),-1,dtype=np.int64)
    mapping[vertex_ids] = np.arange(len(vertex_ids))
    result = Mesh(mesh.v[vertex_ids],mapping[mesh.f[face_ids]],length_unit=mesh.length_unit)
    topo = require_oriented_manifold(result)
    loops = _boundary_loops(_incidence(result)[0])
    if topo.components != 1 or topo.euler_characteristic != 0 or len(loops) != 2:
        raise ValueError("endcap selection must leave one annulus with two boundary loops")
    # Each output boundary must belong exclusively to one supplied cap. This
    # rejects mixed/interleaved selections that happen to leave two holes.
    cap_neighbors = []
    for cap in (a,p):
        touching = np.isin(mesh.f,cap).any(axis=1)
        cap_neighbors.append(set(np.unique(mesh.f[touching])))
    labels = []
    for loop in loops:
        vertices = set(vertex_ids[loop])
        matches = [i for i,neighbors in enumerate(cap_neighbors) if vertices <= neighbors]
        if len(matches) != 1:
            raise ValueError("each boundary must correspond to exactly one selected endcap")
        labels.append(matches[0])
    if sorted(labels) != [0,1]:
        raise ValueError("endcap selections must produce distinct boundaries")
    ordered = tuple(_readonly(loops[labels.index(i)]) for i in (0,1))
    return OpenedSurface(result,_readonly(vertex_ids),_readonly(face_ids),_readonly(mapping),ordered)


def shortest_seam(mesh, endpoints):
    """Shortest Euclidean edge path between explicit vertices on opposite ends.

    All other boundary vertices and boundary edges are excluded. This is a
    graph shortest path, not a continuous surface geodesic; ties may select
    different equally short paths under reindexing. No dorsal landmark is
    inferred. Returns zero-based IDs in endpoint order.
    """
    topo = require_oriented_manifold(mesh)
    edges = _incidence(mesh)[0]
    loops = _boundary_loops(edges)
    if topo.components != 1 or topo.euler_characteristic != 0 or len(loops) != 2:
        raise ValueError("seam search requires one topological annulus")
    ends = _indices(endpoints,"endpoints",len(mesh.v),1)
    if ends.shape != (2,) or not any(ends[0] in x and ends[1] in y
                                    for x,y in (loops,loops[::-1])):
        raise ValueError("endpoints must be on opposite boundary loops")
    blocked = set(np.concatenate(loops))-set(ends)
    pairs = [(a,b) for (a,b),entries in edges.items()
             if len(entries) == 2 and a not in blocked and b not in blocked]
    if not pairs:
        raise ValueError("no admissible interior edge path connects endpoints")
    ij = np.asarray(pairs)
    weights = np.linalg.norm(mesh.v[ij[:,0]]-mesh.v[ij[:,1]],axis=1)
    graph = coo_matrix((np.r_[weights,weights],(np.r_[ij[:,0],ij[:,1]],
                       np.r_[ij[:,1],ij[:,0]])),shape=(len(mesh.v),len(mesh.v))).tocsr()
    distances, parents = dijkstra(graph,indices=int(ends[0]),return_predecessors=True)
    if not np.isfinite(distances[ends[1]]):
        raise ValueError("no admissible interior edge path connects endpoints")
    path = [int(ends[1])]
    while path[-1] != ends[0]:
        path.append(int(parents[path[-1]]))
    return _readonly(path[::-1])
