"""Surface-arclength annular coordinates, with explicit sampling correspondence.

Adapts the geometric stages of generateCurrentSPCutMesh.m (N. P. Mitchell).
The source curve sampler credits Yo Fukushima; this implementation uses linear
interpolation of cumulative segment lengths instead of its iterative traversal.
Phi is in turns, as in MATLAB, and is not generally a polar angle or an
equal-arclength parameter around each hoop. Temporal phase fitting is separate.
"""

from dataclasses import dataclass

import numpy as np

from .curves import curve_arclength
from .grid import _size
from .mesh import Mesh, real_array
from .parameterization import _readonly
from .resampling import UVGrid, resample_annulus


def _resample_curve(curve, count):
    ss = curve_arclength(curve)
    samples = np.linspace(0,ss[-1],count)
    result = np.column_stack([np.interp(samples,ss,curve[:,i]) for i in range(3)])
    result[0], result[-1] = curve[0], curve[-1]
    return result


def _ring_lengths(curves, seam_weight):
    steps = np.linalg.norm(np.diff(curves,axis=0),axis=2)
    if seam_weight == "unique":
        steps = steps[:,:-1]
    lengths = np.r_[0.,np.cumsum(steps.mean(axis=1))]
    if not np.isfinite(lengths).all() or np.any(np.diff(lengths) <= 0):
        raise ValueError("successive hoops must have finite positive mean separation")
    return lengths


@dataclass(frozen=True, eq=False)
class SPhiGrid:
    """A two-stage surface approximation with s in XYZ units and phi in turns.

    mesh.uv is (s,phi), not normalized UV. n_v includes an exact duplicate seam.
    first_sampling maps source cut vertices to the intermediate grid;
    final_sampling maps intermediate vertices to the output. Interpolate a
    source field through both maps with interpolate(). Neither stage transforms
    vector/tensor bases. Retain the result when registering mesh in TubULAR.
    """

    mesh: Mesh
    first_sampling: UVGrid
    final_sampling: UVGrid
    crude_ring_s: np.ndarray
    ring_s: np.ndarray
    u_samples: np.ndarray
    phi_offsets: np.ndarray
    uniform_hoops: np.ndarray
    hoop_centers: np.ndarray
    centerline_s: np.ndarray
    hoop_radii: np.ndarray
    seam_weight: str

    @property
    def path_pairs(self):
        return self.final_sampling.path_pairs

    @property
    def n_u(self):
        return self.final_sampling.n_u

    @property
    def n_v(self):
        return self.final_sampling.n_v

    @property
    def sphi(self):
        return self.mesh.uv

    @property
    def phi_radians(self):
        return _readonly(2*np.pi*self.sphi[:,1])

    def interpolate(self, values):
        return self.final_sampling.interpolate(self.first_sampling.interpolate(values))


def parameterize_sphi(mesh, path_pairs, n_u, n_v, *, phi_offsets=None,
                      seam_weight="unique") -> SPhiGrid:
    """Build surface-based (s,phi) from a validated periodic cut chart.

    1. Sample equal U/V, estimate longitudinal surface lengths and invert s(U).
    2. Sample those U rings; resample each ring by polygonal 3D arclength for
       length/center/radius diagnostics. Their mean longitudinal separations
       define s(U). These equal-arclength hoops do NOT redefine source V.
    3. Relabel the sampled geometry (s(U), V-phi_offsets(U)) and resample it
       onto uniform s/phi. Optional offsets are n_u finite values in turns,
       evaluated on the returned nonuniform u_samples; zero fixes reference
       phase to the source chart. They are never estimated from time or texture.

    Default averages count the duplicate seam once. seam_weight='matlab'
    includes both copies in longitudinal means, matching MATLAB's formulas.
    Original-triangle interpolation, exact boundaries, and preserved winding
    avoid the source's epsilon nudges. Input XYZ must already have the desired
    physical scale. Grid triangles approximate the surface; convergence with
    resolution and experimental-data validation remain the caller's concern.
    """
    n_u, n_v = _size(n_u,"n_u"), _size(n_v,"n_v",4)
    if seam_weight not in ("unique","matlab"):
        raise ValueError("seam_weight must be 'unique' or 'matlab'")
    offsets = np.zeros(n_u) if phi_offsets is None else real_array(phi_offsets,"phi_offsets",ndim=1)
    if offsets.shape != (n_u,):
        raise ValueError("phi_offsets must have length n_u")
    crude = resample_annulus(mesh,path_pairs,n_u,n_v)
    curves = crude.mesh.v.reshape(n_v,n_u,3).transpose(1,0,2)
    crude_s = _ring_lengths(curves,seam_weight)
    us = np.interp(np.linspace(0,crude_s[-1],n_u),crude_s,np.linspace(0,1,n_u))
    first = resample_annulus(mesh,path_pairs,n_u,n_v,u_samples=us)
    curves = first.mesh.v.reshape(n_v,n_u,3).transpose(1,0,2)
    hoops = np.stack([_resample_curve(curve,n_v) for curve in curves])
    ring_s = _ring_lengths(hoops,seam_weight)
    # Exclude the duplicated endpoint, as srFromDVCurves does for closed hoops.
    centers = hoops[:,:-1].mean(axis=1)
    radii = np.linalg.norm(hoops-centers[:,None,:],axis=2)
    center_s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
    s, phi = np.meshgrid(ring_s,np.linspace(0,1,n_v))
    phi -= offsets[None,:]
    chart = np.column_stack((s.ravel(),phi.ravel()))
    intermediate = Mesh(first.mesh.v,first.mesh.f,chart,mesh.length_unit)
    final = resample_annulus(intermediate,first.path_pairs,n_u,n_v)
    result_mesh = Mesh(final.mesh.v,final.mesh.f,final.source_query_uv,mesh.length_unit)
    return SPhiGrid(result_mesh,first,final,*map(_readonly,
                   (crude_s,ring_s,us,offsets,hoops,centers,center_s,radii)),seam_weight)
