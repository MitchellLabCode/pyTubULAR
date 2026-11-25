"""Single-pass PIV kernels and a native integer-window-shift multipass driver.

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
    PREDICTOR_UNAVAILABLE = 6
    OUTSIDE_IMAGE = 7


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
    elif np.ma.isMaskedArray(mask):
        raise ValueError("mask must be a complete boolean array")
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


@dataclass(frozen=True, eq=False)
class MultiPassPIVResult:
    """All coarse-to-fine passes; final is a PIVResult with total displacement.

    Predictors are total pixel displacements on the preceding pass's grid.
    Each next window uses an integer shift of frame B, followed by a residual
    correlation with subpixel refinement. Images are never deformed/resampled.
    """
    passes: tuple

    @property
    def final(self):
        return self.passes[-1]


def _predictor(previous, x, y):
    """Bilinear interpolation requiring every positive-weight corner valid.

    No extrapolation, nearest-valid substitution, or zero*NaN contamination.
    Singleton axes support their sole coordinate only.
    """
    axes = (previous.y[:,0],previous.x[0])
    corners = []
    for axis,q in zip(axes,(y,x)):
        if q < axis[0] or q > axis[-1]:
            return None
        k = int(np.searchsorted(axis,q))
        if k < len(axis) and q == axis[k]:
            corners.append(((k,1.),))
        else:
            t = (q-axis[k-1])/(axis[k]-axis[k-1])
            corners.append(((k-1,1-t),(k,t)))
    dx = dy = 0.
    for i,wy in corners[0]:
        for j,wx in corners[1]:
            if previous.flags[i,j] != PIVFlag.VALID:
                return None
            dx += wx*wy*previous.dx[i,j]
            dy += wx*wy*previous.dy[i,j]
    return dx,dy


def piv_multipass(image_a,image_b,*,window_sizes=(64,32),steps=None,
                  dt=1.,correlation="circular",mask=None) -> MultiPassPIVResult:
    """Coarse-to-fine PIV with integer window shifts and bilinear predictors.

    A native extension of the validated single-pass kernels, NOT PIVlab's
    deforming-window algorithm. Even window_sizes must decrease (repeating is
    allowed); steps defaults to half each size. At least two passes are needed.
    Fine-grid centers outside the preceding grid are invalid, as are predictors
    touching invalid vectors. Shifted windows outside either image are invalid.
    The union mask is checked in both the original A and shifted B windows.
    No outlier replacement or interpolation across missing vectors is implicit.
    """
    a,b = _images(image_a,image_b)
    if np.ma.isMaskedArray(window_sizes) or np.ma.isMaskedArray(steps):
        raise ValueError("window_sizes and steps must be unmasked sequences")
    try:
        sizes = tuple(_size(w,"window_sizes",4) for w in window_sizes)
    except TypeError as exc:
        raise ValueError("window_sizes must be a sequence of even integers") from exc
    if len(sizes) < 2 or any(w%2 or w>min(a.shape) for w in sizes) or any(
            b>a for a,b in zip(sizes,sizes[1:])):
        raise ValueError("window_sizes must contain >=2 nonincreasing even sizes fitting the images")
    if steps is None:
        strides = tuple(w//2 for w in sizes)
    else:
        try:
            strides = tuple(_size(s,"steps",1) for s in steps)
        except TypeError as exc:
            raise ValueError("steps must be an integer sequence matching window_sizes") from exc
        if len(strides) != len(sizes):
            raise ValueError("steps must match window_sizes")
    dt = scalar(dt,"dt",positive=True)
    if correlation not in ("circular","linear"):
        raise ValueError("correlation must be 'circular' or 'linear'")
    if mask is None:
        mask = np.zeros(a.shape,dtype=bool)
    elif np.ma.isMaskedArray(mask):
        raise ValueError("mask must be a complete boolean array")
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != 'b' or mask.shape != a.shape:
            raise ValueError("mask must be a boolean array matching image shape")
    passes = [piv_pair(a,b,window_size=sizes[0],step=strides[0],dt=dt,
                       correlation=correlation,mask=mask)]
    for w,stride in zip(sizes[1:],strides[1:]):
        rows = np.arange(0,a.shape[0]-w+1,stride)
        cols = np.arange(0,a.shape[1]-w+1,stride)
        x,y = np.meshgrid(cols+(w-1)/2,rows+(w-1)/2)
        dx,dy = np.full(x.shape,np.nan),np.full(x.shape,np.nan)
        flags = np.zeros(x.shape,dtype=np.uint8)
        for i,r in enumerate(rows):
            for j,c in enumerate(cols):
                aa = np.s_[r:r+w,c:c+w]
                if mask[aa].any():
                    flags[i,j] = PIVFlag.MASKED
                    continue
                predicted = _predictor(passes[-1],x[i,j],y[i,j])
                if predicted is None:
                    flags[i,j] = PIVFlag.PREDICTOR_UNAVAILABLE
                    continue
                # Nearest integer, ties away from zero; residual retains subpixels.
                sx,sy = (int(np.copysign(np.floor(abs(q)+.5),q)) for q in predicted)
                rr,cc = r+sy,c+sx
                if rr < 0 or cc < 0 or rr+w > a.shape[0] or cc+w > a.shape[1]:
                    flags[i,j] = PIVFlag.OUTSIDE_IMAGE
                    continue
                bb = np.s_[rr:rr+w,cc:cc+w]
                if mask[bb].any():
                    flags[i,j] = PIVFlag.MASKED
                    continue
                one = piv_pair(a[aa],b[bb],window_size=w,step=w,dt=dt,correlation=correlation)
                flags[i,j] = one.flags.item()
                if one.valid.item():
                    dx[i,j],dy[i,j] = sx+one.dx.item(),sy+one.dy.item()
        passes.append(PIVResult(*map(_readonly,(x,y,dx,dy,flags)),dt,w,stride,correlation))
    return MultiPassPIVResult(tuple(passes))
