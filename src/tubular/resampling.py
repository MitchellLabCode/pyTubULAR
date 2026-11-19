"""Periodic annular tiling and regular UV resampling with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import Delaunay, QhullError

from ._coordinates import boolean
from .grid import _size, rectilinear_faces
from .mesh import Mesh, real_array
from .parameterization import (_annular_boundary, _boundary_loops, _check_embedding,
                               _incidence, _readonly)
from .surface import locate_chart


def _periodic_chart(mesh, path_pairs):
    if not isinstance(mesh, Mesh) or mesh.uv is None:
        raise ValueError("an explicit Mesh with a UV chart is required")
    pairs, boundary, sides = _annular_boundary(mesh, path_pairs)
    orientation = _check_embedding(mesh, boundary)
    delta = mesh.uv[pairs[:,0]]-mesh.uv[pairs[:,1]]
    shift = float(delta[0,1])
    width = np.ptp(mesh.uv[:,0])
    if shift == 0 or not np.isfinite(shift):
        raise ValueError("seam must have a finite nonzero V period")
    if (np.any(np.abs(delta[:,0]) > 1e-10*width)
            or np.any(np.abs(delta[:,1]-shift) > 1e-10*abs(shift))):
        raise ValueError("seam pairs must differ by one uniform vertical UV translation")
    return pairs, sides, orientation, shift


def _field(values, source, pairs):
    a = np.asarray(values)
    if a.ndim not in (1,2):
        raise ValueError("values must be a scalar or vector vertex field")
    a = real_array(a, "values", ndim=a.ndim)
    if len(a) != len(source.v) or (a.ndim == 2 and a.shape[1] == 0):
        raise ValueError("values must have one scalar or vector per source vertex")
    if not np.array_equal(a[pairs[:,0]], a[pairs[:,1]]):
        raise ValueError("periodic vertex fields must agree exactly at paired seam vertices")
    return a


def _interpolate(values, vertices, weights):
    w = weights if values.ndim == 1 else weights[...,None]
    return np.sum(w*values[vertices], axis=1)


@dataclass(frozen=True, eq=False)
class TiledAnnulus:
    """A sewn strip of copies with zero-based correspondence to the source.

    vertex_sources and face_sources index the original vertices/faces. V shifts
    reconstruct the tiled coordinates; XYZ is copied. Face corner order stays
    unchanged. Seam corners can refer to either identical source seam copy.
    """

    source: Mesh
    mesh: Mesh
    path_pairs: np.ndarray
    vertex_sources: np.ndarray
    vertex_v_shifts: np.ndarray
    face_sources: np.ndarray
    face_v_shifts: np.ndarray


def tile_annular_mesh(mesh: Mesh, path_pairs, tile_count=(1,1)) -> TiledAnnulus:
    """Sew copies using the tileAnnularCutMesh vertex/face order.

    tile_count=(forward, backward) counts copies in the direction of the signed
    seam shift, first column minus second. Both counts are nonnegative integers.
    Only declared seam pairs are welded, with exactly coincident XYZ. UV seam
    translations must agree to relative tolerance 1e-10; roundoff differences
    at a weld use the already present copy. No arbitrary nearby points merge.
    """
    pairs, _, _, shift = _periodic_chart(mesh, path_pairs)
    if not isinstance(tile_count, (tuple,list,np.ndarray)) or np.shape(tile_count) != (2,):
        raise ValueError("tile_count must contain two nonnegative integers")
    forward, backward = (_size(x, "tile_count", 0) for x in tile_count)
    keep = np.ones(len(mesh.v), dtype=bool)
    keep[pairs[:,1]] = False
    kept = np.flatnonzero(keep)
    vertices, charts, faces = [mesh.v], [mesh.uv], [mesh.f]
    v_sources, v_shifts = [np.arange(len(mesh.v))], [np.zeros(len(mesh.v))]
    f_sources, f_shifts = [np.arange(len(mesh.f))], [np.zeros(len(mesh.f))]
    top = pairs[:,0].copy()
    size = len(mesh.v)
    for i in range(1, forward+backward+1):
        index = np.empty(len(mesh.v), dtype=np.int64)
        index[pairs[:,1]] = top
        index[kept] = np.arange(size, size+len(kept))
        vertices.append(mesh.v[kept])
        charts.append(mesh.uv[kept]+[0,i*shift])
        faces.append(index[mesh.f])
        v_sources.append(kept)
        v_shifts.append(np.full(len(kept),i*shift))
        f_sources.append(np.arange(len(mesh.f)))
        f_shifts.append(np.full(len(mesh.f),i*shift))
        top = index[pairs[:,0]]
        size += len(kept)
    uv = np.vstack(charts)-[0,backward*shift]
    tiled = Mesh(np.vstack(vertices), np.vstack(faces), uv, mesh.length_unit)
    loops = _boundary_loops(_incidence(tiled)[0])
    if len(loops) != 1:
        raise ValueError("tiling must produce one disk boundary")
    _check_embedding(tiled, loops[0])
    return TiledAnnulus(mesh, tiled, _readonly(pairs), _readonly(np.concatenate(v_sources)),
                        _readonly(np.concatenate(v_shifts)-backward*shift),
                        _readonly(np.concatenate(f_sources)),
                        _readonly(np.concatenate(f_shifts)-backward*shift))


@dataclass(frozen=True, eq=False)
class UVGrid:
    """A sampled unit UV grid plus correspondence to the source cut mesh.

    Vertex order is U-fastest, index=j*n_u+i. n_v includes the duplicated seam.
    path_pairs describes the OUTPUT seam, first row (V=0) then last row (V=1).
    source_vertices and barycentric have shape (n_u*n_v,3). corner_v_shifts
    records each sampled corner's periodic UV translation. source_faces indexes
    the source faces in mesh mode, and is None for Delaunay interpolation.

    Resampled triangles connect sampled points and approximate the original
    surface; only mesh-mode sample points themselves lie on original triangles.
    normals, if supplied, are interpolated/normalized source normals, not normals
    recomputed from the new grid. No coordinate frame transformation is applied.
    """

    mesh: Mesh
    source: Mesh
    n_u: int
    n_v: int
    path_pairs: np.ndarray
    source_path_pairs: np.ndarray
    source_vertices: np.ndarray
    barycentric: np.ndarray
    corner_v_shifts: np.ndarray
    source_query_uv: np.ndarray
    source_faces: Optional[np.ndarray]
    normals: Optional[np.ndarray]
    interpolation: str
    preserve_orientation: bool

    def interpolate(self, values):
        """Transfer a periodic scalar/vector SOURCE VERTEX field using this map.

        Paired seam values must agree exactly; no averaging of a discontinuous
        field is inferred. Chart-coordinate vectors/tensors require separate
        basis transforms; this method only interpolates supplied components.
        """
        a = _field(values, self.source, self.source_path_pairs)
        return _interpolate(a, self.source_vertices, self.barycentric)


def resample_annulus(mesh: Mesh, path_pairs, n_u: int, n_v: int, *, normals=None,
                     interpolation="mesh", preserve_orientation=True) -> UVGrid:
    """Sample a periodic cut chart onto a normalized rectilinear UV mesh.

    U boundary rings must lie at 0 and a positive width. V's positive period is
    inferred from explicit seam pairs; query V runs from zero to that period.
    Mesh-mode tile coverage adapts to curved seams. Endpoint samples share the
    exact same weights and positions, with V shifts one period apart.

    mesh (default) uses original triangle barycentric interpolation.
    matlab_scattered uses a Delaunay triangulation of MATLAB's three-tile point
    cloud, matching its linear scatteredInterpolant strategy where triangulation
    is unambiguous. Neither mode extrapolates. Co-circular Delaunay tie choices
    need not agree across libraries. Returned weights make the choice explicit.

    preserve_orientation=True keeps source winding relative to UV. False uses
    MATLAB's positive-UV rectilinear face order, which may reverse physical
    normals. Explicit input normals are periodic unit vectors in source XYZ.
    """
    n_u, n_v = _size(n_u,"n_u"), _size(n_v,"n_v",4)
    preserve = boolean(preserve_orientation,"preserve_orientation")
    if interpolation not in ("mesh","matlab_scattered"):
        raise ValueError("interpolation must be 'mesh' or 'matlab_scattered'")
    pairs, sides, orientation, shift = _periodic_chart(mesh,path_pairs)
    period = abs(shift)
    ring_u = np.array([np.mean(mesh.uv[side,0]) for side in (sides[0],sides[2])])
    width = float(max(ring_u))
    if (width <= 0 or abs(min(ring_u)) > 1e-10*width
            or any(np.max(np.abs(mesh.uv[side,0]-u)) > 1e-10*width
                   for side,u in zip((sides[0],sides[2]),ring_u))):
        raise ValueError("boundary rings must lie on U=0 and a positive constant U width")
    normal_field = None
    if normals is not None:
        normal_field = _field(normals,mesh,pairs)
        if (normal_field.shape != mesh.v.shape
                or np.any(np.abs(normal_field) > 1+1e-10)
                or not np.allclose(np.linalg.norm(normal_field,axis=1),1,rtol=0,atol=1e-10)):
            raise ValueError("normals must be unit XYZ vectors at every source vertex")
    # Remove an integer V offset before adaptive tiling, so a distant but
    # equivalent chart origin does not allocate thousands of unused copies.
    phase_offset = 0.0
    tiling_mesh = mesh
    if interpolation == "mesh":
        center = .5*np.min(mesh.uv[:,1])+.5*np.max(mesh.uv[:,1])
        phase_offset = period*np.floor(center/period)
        if phase_offset != 0:
            tiling_mesh = Mesh(mesh.v,mesh.f,mesh.uv-[0,phase_offset],mesh.length_unit)
    # Include the base and sufficient copies on both sides, leaving a guard
    # period around the query band. Cost depends on seam span, not V origin.
    below = max(1,int(np.ceil(np.max(tiling_mesh.uv[:,1])/period)))
    above = max(1,int(np.ceil(1-np.min(tiling_mesh.uv[:,1])/period)))
    counts = (above,below) if shift > 0 else (below,above)
    if interpolation == "matlab_scattered":
        counts = (1,1)
    tiled = tile_annular_mesh(tiling_mesh,pairs,counts)
    u, v = np.meshgrid(np.linspace(0,width,n_u),np.linspace(0,period,n_v)[:-1])
    query = np.column_stack((u.ravel(),v.ravel()))
    source_faces = None
    if interpolation == "mesh":
        location = locate_chart(tiled.mesh,query)
        if not location.valid.all():
            raise ValueError("UV grid contains samples outside the tiled chart; no extrapolation")
        face_ids, weights = location.face_indices, location.barycentric
        source_faces = tiled.face_sources[face_ids]
        corners = mesh.f[source_faces]
        shifts = np.repeat(tiled.face_v_shifts[face_ids,None]-phase_offset,3,axis=1)
    else:
        try:
            delaunay = Delaunay(tiled.mesh.uv)
        except QhullError as exc:
            raise ValueError("tiled chart could not be Delaunay triangulated") from exc
        ids = delaunay.find_simplex(query,tol=1e-10)
        if np.any(ids < 0):
            raise ValueError("UV grid contains samples outside the tiled point cloud; no extrapolation")
        transform = delaunay.transform[ids]
        ab = np.einsum("nij,nj->ni",transform[:,:2],query-transform[:,2])
        weights = np.column_stack((ab,1-ab.sum(axis=1)))
        tri = delaunay.simplices[ids]
        corners = tiled.vertex_sources[tri]
        shifts = tiled.vertex_v_shifts[tri]
    if not np.isfinite(weights).all() or np.any(weights < -1e-10):
        raise ValueError("invalid barycentric interpolation weights")
    # Duplicate samples, not a second near-equal interpolation at the seam.
    corners = np.vstack((corners,corners[:n_u]))
    weights = np.vstack((weights,weights[:n_u]))
    shifts = np.vstack((shifts,shifts[:n_u]+period))
    query = np.vstack((query,query[:n_u]+[0,period]))
    if source_faces is not None:
        source_faces = _readonly(np.concatenate((source_faces,source_faces[:n_u])))
    xyz = _interpolate(mesh.v,corners,weights)
    faces = rectilinear_faces(n_u,n_v)
    if preserve and orientation < 0:
        faces = faces[:,[0,2,1]]
    grid = Mesh(xyz,faces,query/[width,period],mesh.length_unit)
    output_normals = None
    if normal_field is not None:
        interpolated = _interpolate(normal_field,corners,weights)
        lengths = np.linalg.norm(interpolated,axis=1)
        if np.any(lengths <= 1e-12):
            raise ValueError("interpolated normals cancel; cannot normalize them")
        output_normals = _readonly(interpolated/lengths[:,None])
    output_pairs = np.column_stack((np.arange(n_u),np.arange((n_v-1)*n_u,n_v*n_u)))
    return UVGrid(grid,mesh,n_u,n_v,_readonly(output_pairs),_readonly(pairs),
                  _readonly(corners),_readonly(weights),_readonly(shifts),_readonly(query),
                  source_faces,output_normals,interpolation,preserve)
