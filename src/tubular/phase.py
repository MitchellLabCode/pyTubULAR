"""Geometric angular alignment of corresponding sampled surface hoops.

The least-squares objective follows N. P. Mitchell's phiOffsetsFromPrevMesh
(TubULAR, MIT), in its non-resampled-hoop branch. Uniform periodic sampling
makes this objective piecewise quadratic; every interval is searched exactly.
"""

from dataclasses import dataclass

import numpy as np

from .mesh import real_array
from .parameterization import _readonly


def _hoops(value, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} must contain complete unmasked hoop coordinates")
    a = real_array(value,name,ndim=3)
    if a.shape[0] < 1 or a.shape[1] < 4 or a.shape[2] != 3:
        raise ValueError(f"{name} must have shape (n_hoops, n_phi>=4, 3)")
    if not np.array_equal(a[:,0],a[:,-1]):
        raise ValueError(f"{name} must have an exact duplicated closing endpoint")
    if np.any(np.linalg.norm(np.diff(a,axis=1),axis=2) == 0):
        raise ValueError(f"{name} has coincident consecutive hoop samples")
    return a


def _sample(curve, phase):
    m = len(curve)
    x = (phase % 1)*m
    k = int(np.floor(x))
    t = x-k
    start = np.roll(curve,-k,axis=0)
    return (1-t)*start+t*np.roll(start,-1,axis=0)


def _local_minimum(c,r,w,phase,lower,upper):
    """Discard interval endpoints that are not minima of the whole objective."""
    m = len(c)
    q = (phase % 1)*m
    knot = int(round(q))
    if abs(q-knot) < 1e-9:
        here = np.roll(c,-knot,axis=0)
        left = here-np.roll(here,1,axis=0)
        right = np.roll(here,-1,axis=0)-here
    else:
        start = np.roll(c,-int(np.floor(q)),axis=0)
        left = right = np.roll(start,-1,axis=0)-start
        here = _sample(c,phase)
    a = np.sum(w[:,None]*(here-r)*left)
    b = np.sum(w[:,None]*(here-r)*right)
    tol = 128*np.finfo(float).eps
    periodic = upper-lower >= 1-1e-12
    return ((not periodic and phase <= lower+1e-12) or a <= tol) and (
            (not periodic and phase >= upper-1e-12) or b >= -tol)


@dataclass(frozen=True, eq=False)
class HoopPhaseFit:
    """Offsets solve current(phi + offset) ~= reference(phi), in turns.

    raw_offsets lie in bounds; offsets unwrap adjacent hoops by integer turns
    for a continuous chart seam. No smoothing or temporal prediction is done.
    Residuals omit the duplicate endpoint; RMS uses the requested seam weight.
    ambiguous marks multiple geometrically distinct minima within a normalized
    objective tolerance. at_bounds marks a minimum limited by a sub-period
    search interval. These diagnostics are not statistical confidence bounds.
    """

    raw_offsets: np.ndarray
    offsets: np.ndarray
    aligned_hoops: np.ndarray
    residuals: np.ndarray
    rms_before: np.ndarray
    rms_after: np.ndarray
    ambiguous: np.ndarray
    at_bounds: np.ndarray
    bounds: tuple
    seam_weight: str


def fit_hoop_phase(current_hoops, reference_hoops, *, bounds=(-.5,.5),
                   seam_weight="unique") -> HoopPhaseFit:
    """Globally minimize squared 3D separation on explicitly corresponding hoops.

    Both arrays have shape (n_hoops,n_phi,3) and uniform phase samples from
    0 to 1 inclusive. XYZ must share a frame and physical units. n_phi includes
    the exactly duplicated seam. Curves are linearly interpolated in phase;
    they are NOT resampled in arclength during fitting. No hoop correspondence,
    rigid registration, physical tracking, or missing data is inferred.

    The search interval has positive width <= one turn and lies in [-1,1].
    Every piecewise-quadratic interval is minimized, including both bounds;
    this avoids the unimodality assumption of MATLAB's single fminbnd search.
    seam_weight='matlab' counts the closing endpoint twice, reproducing the
    non-resampled MATLAB objective. The default counts each physical point once.
    """
    current = _hoops(current_hoops,"current_hoops")
    reference = _hoops(reference_hoops,"reference_hoops")
    if current.shape != reference.shape:
        raise ValueError("current and reference hoops must have identical shapes")
    limits = real_array(bounds,"bounds",ndim=1)
    if (limits.shape != (2,) or limits[0] < -1 or limits[1] > 1
            or not 0 < limits[1]-limits[0] <= 1):
        raise ValueError("bounds must be increasing, inside [-1,1], spanning at most one turn")
    if seam_weight not in ("unique","matlab"):
        raise ValueError("seam_weight must be 'unique' or 'matlab'")
    lower,upper = map(float,limits)
    n,m = current.shape[0],current.shape[1]-1
    weights = np.ones(m)
    if seam_weight == "matlab":
        weights[0] = 2
    raw = np.zeros(n)
    ambiguous = np.zeros(n,dtype=bool)
    at_bounds = np.zeros(n,dtype=bool)
    for i in range(n):
        c,r = current[i,:-1],reference[i,:-1]
        scale = max(np.ptp(c,axis=0).max(),np.ptp(r,axis=0).max())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("hoops must have finite positive spatial extent")
        origin = c[0]
        c,r = (c-origin)/scale,(r-origin)/scale
        if seam_weight == "unique":
            # Uniform samples have phase-independent means. Removing the two
            # means removes only a constant from the loss, improving conditioning
            # when the frames differ by translation. Full residuals retain it.
            c,r = c-c.mean(axis=0),r-r.mean(axis=0)
        candidates,losses = [],[]
        for k in range(int(np.floor(lower*m)),int(np.ceil(upper*m))):
            lo,hi = max(0.,lower*m-k),min(1.,upper*m-k)
            if lo > hi:
                continue
            start = np.roll(c,-k,axis=0)
            direction = np.roll(start,-1,axis=0)-start
            denom = np.sum(weights[:,None]*direction**2)
            t = np.clip(-np.sum(weights[:,None]*(start-r)*direction)/denom,lo,hi)
            candidates.append((k+t)/m)
            losses.append(np.sum(weights[:,None]*(start+t*direction-r)**2)/weights.sum())
        candidates,losses = np.asarray(candidates),np.asarray(losses)
        best = int(np.argmin(losses))
        raw[i] = candidates[best]
        # Only actual local minima count as competing fits. A nearby boundary
        # of the winning interval is not a second minimum.
        tol = 1e-10*max(np.mean(c*c)+np.mean(r*r),np.finfo(float).eps)
        near = np.flatnonzero(losses <= losses[best]+tol)
        for j in near:
            separation = abs((candidates[j]-raw[i]+.5)%1-.5)
            if separation > 1e-7 and _local_minimum(c,r,weights,candidates[j],lower,upper):
                ambiguous[i] = True
        at_bounds[i] = upper-lower < 1-1e-12 and min(abs(raw[i]-lower),abs(raw[i]-upper)) < 1e-10
    offsets = np.unwrap(raw*2*np.pi)/(2*np.pi)
    aligned = np.stack([_sample(c[:-1],phase) for c,phase in zip(current,offsets)])
    residuals = np.linalg.norm(aligned-reference[:,:-1],axis=2)
    before = np.linalg.norm(current[:,:-1]-reference[:,:-1],axis=2)
    rms_before = np.sqrt(np.average(before**2,weights=weights,axis=1))
    rms_after = np.sqrt(np.average(residuals**2,weights=weights,axis=1))
    aligned = np.concatenate((aligned,aligned[:,:1]),axis=1)
    return HoopPhaseFit(*map(_readonly,(raw,offsets,aligned,residuals,rms_before,rms_after,
                                       ambiguous,at_bounds)),(lower,upper),seam_weight)
