"""Single-pass PIV using translated PIVlab correlation/peak kernels.

Derived from PIVlab piv_FFTmulti.m, copyright (c) 2021 William Thielicke,
MIT license (licenses/PIVlab-MIT.txt). Its three-point Gaussian formula credits
copyright (c) 1998 Uri Shavit, Roi Gurka, Alex Liberzon, Technion, Israel.
This module does not implement the full PIVlab multipass/preprocessing workflow.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.fft import fft2, ifft2, fftshift

from ._coordinates import scalar
from .grid import _size
from .mesh import real_array
from .parameterization import _readonly


class PIVFlag(IntEnum):
    VALID = 0
    MASKED = 1
    LOW_TEXTURE = 2
    AMBIGUOUS_PEAK = 3
    BORDER_PEAK = 4
    INVALID_GAUSSIAN = 5


def _images(a,b):
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        raise ValueError("use the explicit mask argument, not masked image arrays")
    a,b = real_array(a,"image_a",ndim=2),real_array(b,"image_b",ndim=2)
    if a.shape != b.shape or min(a.shape,default=0) < 1:
        raise ValueError("images must be nonempty 2D arrays with identical shapes")
    return a,b


def correlate_windows(a,b,*,correlation="circular"):
    """PIVlab do_correlations: FFT cross-correlation of two even square windows.

    Zero displacement lies at [W//2,W//2]. Linear mode subtracts window means,
    pads to 2W-1, then crops the central W samples with MATLAB's index offset.
    Circular mode does not subtract means. Returns raw float64 correlation.
    """
    a,b = _images(a,b)
    w = a.shape[0]
    if a.shape != (w,w) or w < 4 or w % 2:
        raise ValueError("correlation windows must be square, even, and at least 4 pixels")
    if correlation not in ("circular","linear"):
        raise ValueError("correlation must be 'circular' or 'linear'")
    shape = (w,w)
    if correlation == "linear":
        a,b = a-a.mean(),b-b.mean()
        shape = (2*w-1,2*w-1)
    result = fftshift(ifft2(np.conj(fft2(a,shape))*fft2(b,shape)).real)
    if correlation == "linear":
        start = w//2-1
        result = result[start:start+w,start:start+w]
    if not np.isfinite(result).all():
        raise ValueError("correlation overflow; rescale image intensities")
    return result


def _gaussian_peak(correlation):
    span = np.ptp(correlation)
    if span <= 64*np.finfo(float).eps*np.max(np.abs(correlation)):
        return np.nan,np.nan,PIVFlag.AMBIGUOUS_PEAK
    c = (correlation-correlation.min())/span*255
    peaks = np.argwhere(c >= 255*(1-64*np.finfo(float).eps))
    if len(peaks) != 1:
        return np.nan,np.nan,PIVFlag.AMBIGUOUS_PEAK
    y,x = peaks[0]
    w = len(c)
    if x in (0,w-1) or y in (0,w-1):
        return np.nan,np.nan,PIVFlag.BORDER_PEAK
    values = c[[y,y,y-1,y+1,y],[x-1,x+1,x,x,x]]
    if np.any(values <= 0):
        return np.nan,np.nan,PIVFlag.INVALID_GAUSSIAN
    lx,rx,ty,by,peak = np.log(values)
    denominators = np.array([2*lx-4*peak+2*rx,2*ty-4*peak+2*by])
    if np.any(denominators >= -64*np.finfo(float).eps):
        return np.nan,np.nan,PIVFlag.INVALID_GAUSSIAN
    offsets = np.array([lx-rx,ty-by])/denominators
    if not np.isfinite(offsets).all() or np.any(np.abs(offsets) > .5+1e-12):
        return np.nan,np.nan,PIVFlag.INVALID_GAUSSIAN
    return x-w//2+offsets[0],y-w//2+offsets[1],PIVFlag.VALID


@dataclass(frozen=True, eq=False)
class PIVResult:
    """Image-plane displacements; +x right, +y down, positions are zero-based.

    Invalid vectors remain NaN and have a nonzero flag. dx/dy are pixels per
    pair, u/v properties divide by dt (caller-specified time units). No physical
    scaling or mapping onto the 3D surface is implicit.
    """

    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    flags: np.ndarray
    dt: float
    window_size: int
    step: int
    correlation: str

    @property
    def valid(self):
        return _readonly(self.flags == PIVFlag.VALID)

    @property
    def u(self):
        return _readonly(self.dx/self.dt)

    @property
    def v(self):
        return _readonly(self.dy/self.dt)


def piv_pair(image_a,image_b,*,window_size=32,step=None,dt=1.,
             correlation="circular",mask=None) -> PIVResult:
    """One pass of square-window PIV with three-point Gaussian peak refinement.

    Only complete windows are evaluated, anchored at pixel [0,0], without
    padding, preprocessing, deformation, validation filters or inpainting.
    A True mask pixel invalidates every overlapping window. A single mask
    applies to both frames (use the union if they differ). All images must be
    finite; replace missing intensities explicitly and mark them in the mask.
    Exact/tolerance-level tied peaks and invalid Gaussian fits remain NaN.
    Accepts only these supported settings; no silent fallback for PIVlab options.
    """
    a,b = _images(image_a,image_b)
    w = _size(window_size,"window_size",4)
    stride = w//2 if step is None else _size(step,"step",1)
    dt = scalar(dt,"dt",positive=True)
    if w % 2 or w > min(a.shape):
        raise ValueError("window_size must be even and fit inside both images")
    if correlation not in ("circular","linear"):
        raise ValueError("correlation must be 'circular' or 'linear'")
    if mask is None:
        mask = np.zeros(a.shape,dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b" or mask.shape != a.shape:
            raise ValueError("mask must be a boolean array matching image shape")
    rows = np.arange(0,a.shape[0]-w+1,stride)
    cols = np.arange(0,a.shape[1]-w+1,stride)
    x,y = np.meshgrid(cols+(w-1)/2,rows+(w-1)/2)
    dx,dy = np.full(x.shape,np.nan),np.full(x.shape,np.nan)
    flags = np.zeros(x.shape,dtype=np.uint8)
    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            region = np.s_[r:r+w,c:c+w]
            if mask[region].any():
                flags[i,j] = PIVFlag.MASKED
                continue
            aa,bb = a[region],b[region]
            # Positive scaling cancels in rescale_array and avoids overflow.
            aa = aa/max(float(np.max(np.abs(aa))),np.finfo(float).tiny)
            bb = bb/max(float(np.max(np.abs(bb))),np.finfo(float).tiny)
            if min(np.std(aa),np.std(bb)) <= 64*np.finfo(float).eps:
                flags[i,j] = PIVFlag.LOW_TEXTURE
                continue
            dx[i,j],dy[i,j],flags[i,j] = _gaussian_peak(correlate_windows(aa,bb,correlation=correlation))
    return PIVResult(*map(_readonly,(x,y,dx,dy,flags)),dt,w,stride,correlation)
