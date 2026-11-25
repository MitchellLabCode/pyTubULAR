# PIV implementation and package audit

Version 0.11 retains the **single-pass image-pair PIV core** introduced in 0.9, with translated
PIVlab FFT correlation and three-point Gaussian peak kernels. It is exposed as
`tubular.piv.piv_pair` and `TubULAR.measurePIV2d`. It does not yet reproduce the
four-pass deforming-window workflow in MATLAB TubULAR. Version 0.11 adds a
native coarse-to-fine integer-window-shift driver, documented below.

```python
result = tubi.measurePIV2d(image_a, image_b, window_size=32, step=16,
                         dt=0.5, correlation="circular", mask=invalid_pixels)
dx, dy = result.dx, result.dy       # pixels per image pair
vx, vy = result.u, result.v         # pixels per supplied time unit
good = result.valid
```

Images are finite 2D arrays indexed `[row, column]`. Positions `x,y` use
**zero-based pixel centers**, positive x right and positive y down. This differs
from the class's older one-based `XY2uv` helpers: add 1 to these positions before
using those helpers, and explicitly supply the chart extents. A window of width
32 starting at pixel 0 has center 15.5. Only complete windows are sampled,
anchored at the image's top-left; trailing incomplete strips are omitted.
This is intentionally different from PIVlab's boundary padding and grid setup.
The experiment's selected time, APDV settings and unit labels do not change dt.

Circular correlation is the PIVlab/TubULAR setting used by default. Linear
correlation subtracts each window's mean, zero-pads to `2W-1` and crops the
central W lag samples exactly as PIVlab does. Even square windows of at least
four pixels are required. `step` defaults to half the window width. Window
intensities are positively rescaled before correlation to avoid overflow; this
does not change the rescaled peak or displacement. No CLAHE, Wiener filter,
high-pass filter, clipping, ROI selection, periodic image extension, physical
calibration or 3D surface velocity mapping is implicit.

Any True pixel in the supplied boolean mask invalidates an overlapping window;
use the union of both frame masks. This is stricter than PIVlab's center mask.
Invalid vectors stay NaN, with `PIVFlag` reasons for masks, low texture, tied
peaks, border peaks, or invalid Gaussian fits. Ties are not resolved arbitrarily.
These are measurement flags, not a complete PIV quality estimate: a unique peak
can still be wrong. Global/local outlier validation, uncertainty, peak-ratio
thresholds and inpainting are pending. Unsupported settings raise errors.
Run `python examples/piv_pair.py` for a known-shift example.

## Why this implementation, and available alternatives

Audit date: 2026-11-19. The exact TubULAR calls are in
[`measurePIV2d.m` at 9bd49db](https://github.com/npmitchell/tubular/blob/9bd49dbffd38b199c32552b8cc089f611f584679/@TubULAR/measurePIV2d.m).
Its configured pipeline uses CLAHE/Wiener preprocessing, up to four decreasing
window sizes, Gaussian peak fitting, spline deformation, repeated correlation,
circular correlation, repeat-last-pass, standard-deviation/local-median
validation and missing-vector inpainting. These features must be compared
separately; optical flow is not a numerical substitute for that pipeline.

| Candidate | Relevant capabilities | Constraints and decision |
| --- | --- | --- |
| [OpenPIV 0.26.1](https://pypi.org/project/OpenPIV/0.26.1/) | [Multipass window deformation](https://openpiv.readthedocs.io/en/latest/src/windef.html), circular/linear correlation, Gaussian subpixels, vector validation/replacement | GPLv3; Python >=3.10. Strong candidate for a later backend. Not installed or benchmarked here; its defaults/settings do not establish PIVlab equivalence. |
| [FluidImage 0.5.5](https://pypi.org/project/fluidimage/) | [Multipass PIV](https://fluidimage.readthedocs.io/en/latest/_generated/fluidimage.topologies.piv.html), decreasing windows, thin-plate-spline options and parallel workflows | BSD-3-Clause; Python >=3.11. Another candidate, particularly for batch processing. The linked development docs identify as 0.5.6; release behavior needs a pinned-version test. Not installed or benchmarked here. |
| [PIVlab at 10ca97f](https://github.com/Shrediquette/PIVlab/tree/10ca97f90bb14cd58f644855ed84ad4d5016cbc2) | Original MATLAB algorithm family used by TubULAR | [MIT license](https://github.com/Shrediquette/PIVlab/blob/10ca97f90bb14cd58f644855ed84ad4d5016cbc2/License.txt). Selected small kernels translated directly, with attribution and original-kernel fixtures. No new dependency or Python-version change. |

The package recommendation is an assessment of documented capabilities, not
measured equivalence. The current pyTubULAR floor is Python 3.9. Neither current
Python candidate can simply become a mandatory dependency without changing that
floor, and their license/dependency requirements must stay explicit. No OpenPIV
or FluidImage code is copied or bundled. PIVlab's notice is included under
`licenses/PIVlab-MIT.txt`; the Gaussian formula's original author credit is
retained in `piv.py`.

At the pinned PIVlab revision, `piv_FFTmulti` accepts named arguments, whereas
TubULAR calls an older positional signature. A future MATLAB parity harness
must choose/pin a compatible API explicitly rather than rely on the source's
catch-and-retry fallback. The current kernel harness bypasses that wrapper.

## Validation and remaining work

`scripts/generate_piv_reference.m` extracts unchanged local functions
`do_correlations`, `meanzeropad`, `rescale_array` and `SUBPIXGAUSS` from the
pinned upstream source into a temporary folder, then runs them in MATLAB.
It does not edit the checkout. Source and generator hashes accompany the
reference fixture. Six cases cover circular/linear correlation on zero,
integer and fractional shifts; Python displacements differ from the MATLAB
kernel outputs by at most approximately `1.8e-15` pixels on the tested platform.

Independent tests check direct-sum linear correlation, displacement signs,
frame reversal, zero-based centers, dt scaling, masks and ambiguous/invalid
measurements. The periodic synthetic fractional-shift case has a circular-mode
error of approximately `0.00393` pixels relative to its known translation.
That is one synthetic case, not a general accuracy guarantee. The same data
with linear correlation have up to `0.131` pixels of bias, reproduced from
MATLAB, because changing correlation changes the estimator.

Next steps are multipass prediction/deformation, repeated correlation, explicit
preprocessing and outlier filters, followed by paired real pullback-image
comparisons and mapping validated vectors into 3D. Surface phase correspondence
between frames must also be established before interpreting PIV as tissue flow.

## Coarse-to-fine window shifting (0.11)

A sequence of window sizes now selects a native multipass driver:

```python
series = tubi.measurePIV2d(image_a, image_b, window_size=(64, 32),
                          step=(32, 16), dt=0.5, mask=invalid_pixels)
result = series.final               # PIVResult: total displacement, not residual
first_pass = series.passes[0]       # every pass is retained for inspection
```

The standalone API is `piv_multipass(..., window_sizes=(64,32), steps=None)`.
A scalar class `window_size` still returns the original single-pass `PIVResult`.
Sequences require at least two nonincreasing even sizes; repeating a size is
allowed. Steps default to half each size, or accept a matching sequence.

The first pass uses the existing PIVlab kernels. Each later window bilinearly
interpolates total displacement from the previous grid, shifts the window in
frame B by the nearest integer (half ties away from zero), then adds its measured
subpixel residual to that integer shift. Positions remain the centers in frame
A. Both image windows must fit, and both must avoid the union mask. New flags
`PREDICTOR_UNAVAILABLE` and `OUTSIDE_IMAGE` explain rejected later vectors.

Every positive-weight predictor corner must be valid. Zero-weight corners do
not contaminate exact grid-node queries. There is no extrapolation beyond the
previous grid, so smaller windows near image edges can remain invalid even
when they fit in the image. Invalid vectors are not filled. An incorrect but
unflagged coarse vector can guide later passes incorrectly; a valid flag is
not evidence of accuracy. Inspect all passes and the mask.

This is **integer window shifting**, not PIVlab's spline image deformation,
repeated correlation or repeat-until-converged workflow. The driver is a native
extension, not a direct translation of `piv_FFTmulti`. It uses the already
validated kernels without changing their default numerical behavior. Periodic
surface seams are not automatically wrapped in PIV image sampling.

Independent tests recover displacements `(12,-8)` and `(-11,9)` pixels using
64→32→16 windows, where the same single 16-pixel pass aliases the motion.
They check time scaling, masks in both windows, exact affine interpolation,
missing predictors, out-of-image samples and unchanged first-pass outputs.
A prescribed fractional affine shear with 64→32 windows has maximum component
errors 0.089 and 0.072 pixels, within the original 0.15-pixel test threshold.
For that same texture, continuing to 16 pixels increases the maxima to 0.193
and 0.165 pixels: smaller windows do not guarantee better estimates. The example
reports this limitation; the regression test retains the 0.15-pixel accuracy
requirement with a 32-pixel final window. These bounds describe those synthetic cases only.

Image deformation, signal-to-noise and outlier validation, real-data comparison,
backend benchmarking, and calibrated lifting onto a moving surface remain next
steps. The package audit above remains dated 2026-11-19.
