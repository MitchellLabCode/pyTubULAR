"""Periodic texture alignment on supplied, axially corresponding pullbacks.

This native adaptation of TubULAR's texture-fit stage uses normalized strip
cross-correlation, not the original 2D whitened phase correlation or polynomial
smoothing. No volume projection, axial motion estimation, or file I/O occurs.
"""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.fft import fft, ifft

from .grid import _size
from .mesh import real_array
from .parameterization import _readonly
from .piv import _images


class TexturePhaseFlag(IntEnum):
    VALID = 0
    MASKED = 1
    LOW_TEXTURE = 2
    AMBIGUOUS_PEAK = 3
    AT_BOUND = 4


@dataclass(frozen=True, eq=False)
class TexturePhaseFit:
    """Offsets solve current(phi + offset, s) ~= reference(phi, s), in turns.

    Rows sample ONE period without a duplicated endpoint. Columns share an
    explicit axial correspondence. Invalid strips have NaN offsets. Unwrapping
    is only within contiguous valid runs; it assumes changes below half a turn.
    peak_correlation is a normalized integer-lag correlation, not uncertainty.
    """
    axial_samples: np.ndarray
    raw_offsets: np.ndarray
    offsets: np.ndarray
    flags: np.ndarray
    peak_correlation: np.ndarray
    image_shape: tuple
    bounds: tuple
    method: str

    @property
    def valid(self):
        return _readonly(self.flags == TexturePhaseFlag.VALID)

    @property
    def shift_pixels(self):
        return _readonly(self.offsets*self.image_shape[0])

    def phase_offsets(self, axial_samples):
        """Interpolate valid unwrapped offsets at normalized image-column positions.

        The caller must map its surface hoops to these image positions. Rejects
        missing/ambiguous/bound-limited strips; never fills or extrapolates.
        To refine an already shifted chart, add these incremental offsets to
        the existing offsets before rebuilding it from the same original cut.
        """
        if not self.valid.all():
            raise ValueError("all texture strips must be valid before constructing surface offsets")
        if np.ma.isMaskedArray(axial_samples):
            raise ValueError("axial_samples must be unmasked")
        u = real_array(axial_samples,"axial_samples",ndim=1)
        if not len(u) or np.any((u < 0) | (u > 1)):
            raise ValueError("axial_samples must be nonempty and inside [0,1]")
        return _readonly(np.interp(u,self.axial_samples,self.offsets))


def fit_texture_phase(current, reference, *, strip_width=32, step=None,
                      bounds=(-.5,.5), method="parabolic", mask=None) -> TexturePhaseFit:
    """Fit periodic row shifts in overlapping axial strips, without axial shifts.

    Both images are finite grayscale arrays shaped (n_phi>=4, n_s>=2), with
    unique periodic rows and the same increasing column coordinates. Strip
    centers include both image edges; strips there are clipped, never padded.
    Positive per-strip gain and per-column intensity offsets are removed.
    method='integer' uses the peak lag; 'parabolic' refines its three neighbors.
    Restricted bounds are in turns; peaks touching their sampled limits or
    refined outside them are flagged. True mask pixels reject whole strips.
    """
    a,b = _images(current,reference)
    h,w = a.shape
    width = _size(strip_width,"strip_width",1)
    stride = max(1,width//2) if step is None else _size(step,"step",1)
    if h < 4 or w < 2 or width > w:
        raise ValueError("images require >=4 periodic rows, >=2 columns, and a fitting strip_width")
    if method not in ("integer","parabolic"):
        raise ValueError("method must be 'integer' or 'parabolic'")
    if np.ma.isMaskedArray(bounds):
        raise ValueError("bounds must be unmasked")
    limits = real_array(bounds,"bounds",ndim=1)
    if limits.shape != (2,) or not (-.5 <= limits[0] < limits[1] <= .5):
        raise ValueError("bounds must increase inside [-0.5,0.5] turns")
    if mask is None:
        mask = np.zeros(a.shape,dtype=bool)
    elif np.ma.isMaskedArray(mask):
        raise ValueError("mask must be a complete boolean array")
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != 'b' or mask.shape != a.shape:
            raise ValueError("mask must be boolean and match the images")
    lags = np.arange(h,dtype=float)
    lags[lags >= (h+1)//2] -= h
    eligible = np.flatnonzero((lags/h >= limits[0]) & (lags/h <= limits[1]))
    if not len(eligible):
        raise ValueError("bounds contain no sampled row shifts")
    full = tuple(limits) == (-.5,.5)
    centers = np.unique(np.r_[np.arange(0,w,stride),w-1])
    raw = np.full(len(centers),np.nan)
    peaks = raw.copy()
    flags = np.zeros(len(centers),dtype=np.uint8)
    eps = 64*np.finfo(float).eps
    for i,x in enumerate(centers):
        lo,hi = max(0,x-width//2),min(w,x+(width+1)//2)
        if mask[:,lo:hi].any():
            flags[i] = TexturePhaseFlag.MASKED
            continue
        c,r = a[:,lo:hi],b[:,lo:hi]
        c = c/max(float(np.max(np.abs(c))),np.finfo(float).tiny)
        r = r/max(float(np.max(np.abs(r))),np.finfo(float).tiny)
        c,r = c-c.mean(axis=0),r-r.mean(axis=0)
        cn,rn = np.linalg.norm(c),np.linalg.norm(r)
        if min(cn,rn) <= eps*np.sqrt(c.size):
            flags[i] = TexturePhaseFlag.LOW_TEXTURE
            continue
        c,r = c/cn,r/rn
        corr = ifft(np.sum(fft(c,axis=0)*np.conj(fft(r,axis=0)),axis=1)).real
        peak = np.max(corr[eligible]); peaks[i] = peak
        best = eligible[corr[eligible] >= peak-eps]
        if len(best) != 1:
            flags[i] = TexturePhaseFlag.AMBIGUOUS_PEAK
            continue
        k = int(best[0]); delta = 0.
        left,right = corr[(k-1)%h],corr[(k+1)%h]
        if method == "parabolic":
            denominator = 2*(left-2*peak+right)
            if denominator >= -eps:
                flags[i] = TexturePhaseFlag.AMBIGUOUS_PEAK
                continue
            delta = (left-right)/denominator
        offset = (lags[k]+delta)/h
        if full:
            offset = (offset+.5)%1-.5
        elif (lags[k] in (np.min(lags[eligible]),np.max(lags[eligible]))
              or not limits[0] <= offset <= limits[1]):
            flags[i] = TexturePhaseFlag.AT_BOUND
            continue
        raw[i] = offset
    unwrapped = np.full_like(raw,np.nan)
    valid = np.flatnonzero(flags == TexturePhaseFlag.VALID)
    for run in np.split(valid,np.flatnonzero(np.diff(valid)>1)+1):
        if len(run):
            unwrapped[run] = np.unwrap(2*np.pi*raw[run])/(2*np.pi)
    return TexturePhaseFit(*map(_readonly,(centers/(w-1),raw,unwrapped,flags,peaks)),
                           a.shape,tuple(map(float,limits)),method)
