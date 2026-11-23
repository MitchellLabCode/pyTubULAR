"""Surface-arclength annular coordinates, with explicit sampling correspondence.

Adapts the geometric stages of generateCurrentSPCutMesh.m (N. P. Mitchell).
The source curve sampler credits Yo Fukushima; this implementation uses linear
interpolation of cumulative segment lengths instead of its iterative traversal.
Phi is in turns, as in MATLAB, and is not generally a polar angle or an
equal-arclength parameter around each hoop. Geometric reference fitting is
optional; texture matching and material tracking remain separate.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .curves import curve_arclength
from .grid import _size
from .mesh import Mesh, real_array
from .parameterization import _readonly
from .resampling import UVGrid, resample_annulus
from .phase import HoopPhaseFit, fit_hoop_phase


@dataclass(frozen=True, eq=False)
class PhaseAlignment:
    """Geometric phase fit and explicit correspondence to a reference s-phi mesh.

    reference_s is normalized reference arclength at each intermediate current
    ring. reference_sampling retains the reference vertex weights. This is
    geometric chart alignment; it does not establish material correspondence.
    """

    fit: HoopPhaseFit
    reference_sampling: UVGrid
    reference_s: np.ndarray


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
    phase_alignment: Optional[PhaseAlignment] = None

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


def _fit_reference(first, ring_s, reference, reference_s, bounds):
    if not isinstance(reference,SPhiGrid):
        raise TypeError("reference must be an SPhiGrid result")
    if reference.mesh.length_unit != first.mesh.length_unit:
        raise ValueError("current and reference mesh length units must match")
    rs = ring_s/ring_s[-1] if reference_s is None else real_array(reference_s,"reference_s",ndim=1)
    sampled = resample_annulus(reference.mesh,reference.path_pairs,first.n_u,first.n_v,u_samples=rs)
    shape = (first.n_v,first.n_u,3)
    current = first.mesh.v.reshape(shape).transpose(1,0,2)
    target = sampled.mesh.v.reshape(shape).transpose(1,0,2)
    fit = fit_hoop_phase(current,target,bounds=bounds)
    return PhaseAlignment(fit,sampled,_readonly(rs))


def fit_sphi_phase(current, reference, *, reference_s=None, bounds=(-.5,.5)) -> PhaseAlignment:
    """Inspect a geometric phase fit without applying it, including failed fits.

    Both inputs are SPhiGrid results. Fit absolute offsets of current's original
    first_sampling chart against reference's final mesh, even if current already
    has offsets. Reference_s sets normalized axial correspondence (default:
    current normalized ring_s); both endpoints must be 0/1. Ambiguous and
    bound-limited fits are returned as diagnostics rather than rejected here.
    """
    if not isinstance(current,SPhiGrid):
        raise TypeError("current must be an SPhiGrid result")
    return _fit_reference(current.first_sampling,current.ring_s,reference,reference_s,bounds)


def parameterize_sphi(mesh, path_pairs, n_u, n_v, *, phi_offsets=None,
                      seam_weight="unique", reference=None, reference_s=None,
                      phase_bounds=None) -> SPhiGrid:
    """Build surface-based (s,phi) from a validated periodic cut chart.

    1. Sample equal U/V, estimate longitudinal surface lengths and invert s(U).
    2. Sample those U rings; resample each ring by polygonal 3D arclength for
       length/center/radius diagnostics. Their mean longitudinal separations
       define s(U). These equal-arclength hoops do NOT redefine source V.
    3. Relabel the sampled geometry (s(U), V-phi_offsets(U)) and resample it
       onto uniform s/phi. Optional offsets are n_u finite values in turns,
       evaluated on the returned nonuniform u_samples; zero fixes reference
       phase to the source chart.

    Alternatively supply a previous SPhiGrid as reference to fit geometric
    offsets. Reference hoops are sampled at the current normalized ring_s by
    default, or at explicit reference_s (strictly increasing n_u values from
    0 to 1). Units must match; XYZ must already share a frame. The fit uses
    periodic piecewise-linear hoops without per-trial arclength resampling,
    counts each phase sample once, searches phase_bounds globally, and unwraps
    offsets along the axis. phase_bounds defaults to (-.5,.5) when fitting.
    Ambiguous/bound-limited fits raise before returning or registration. Raw
    fits are separately inspectable with fit_sphi_phase or fit_hoop_phase.
    phi_offsets and reference are mutually exclusive. No texture matching,
    temporal smoothing, landmark detection or rigid alignment is inferred.

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
    if reference is not None:
        if not isinstance(reference,SPhiGrid):
            raise TypeError("reference must be an SPhiGrid result")
        if phi_offsets is not None:
            raise ValueError("reference and phi_offsets are mutually exclusive")
        if reference.mesh.length_unit != mesh.length_unit:
            raise ValueError("current and reference mesh length units must match")
    elif reference_s is not None or phase_bounds is not None:
        raise ValueError("reference_s and phase_bounds require a reference SPhiGrid")
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
    alignment = None
    if reference is not None:
        alignment = _fit_reference(first,ring_s,reference,reference_s,
                                   (-.5,.5) if phase_bounds is None else phase_bounds)
        fit = alignment.fit
        if fit.ambiguous.any():
            raise ValueError("geometric phase fit is ambiguous; inspect fit_sphi_phase diagnostics")
        if fit.at_bounds.any():
            raise ValueError("geometric phase fit reached phase_bounds; inspect or widen the interval")
        offsets = fit.offsets
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
                   (crude_s,ring_s,us,offsets,hoops,centers,center_s,radii)),seam_weight,alignment)
