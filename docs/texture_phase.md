# Texture-based surface phase alignment (0.11)

`TubULAR.fitPhiOffsetsViaTexture(current, reference, ...)` now fits the angular
origin of supplied pullback images. The native function is
`tubular.texture_phase.fit_texture_phase`. It returns a `TexturePhaseFit` with
strip positions, raw and unwrapped offsets, normalized peak correlations and
validity flags. No mesh or selected time is changed by fitting.

Images are finite grayscale arrays `(n_phi, n_s)`: rows uniformly sample **one
period, without a duplicated closing row**, and columns already correspond
axially. Double covers, reversed image axes, channels and axial registration
must be handled explicitly. Nonconsecutive reference times are allowed because
the caller supplies both images. A True pixel in the optional union mask
invalidates every strip containing it. Masked arrays and nonfinite images are
rejected.

## From texture to a surface map

The sign matches the geometric phase API:
`current(phi + offset, s) ~= reference(phi, s)`. Offsets are in turns. If the
current image is rolled down by eight rows out of 64, the offset is `+8/64`.

```python
# current_image samples the unshifted chart's s uniformly in its columns.
initial = tubi.generateCurrentSPCutMesh(path_pairs)
fit = tubi.fitPhiOffsetsViaTexture(current_image, reference_image,
                                  strip_width=32, step=16)
# These intermediate hoops need not be uniformly spaced in physical s.
image_positions = initial.ring_s / initial.ring_s[-1]
phi_offsets = fit.phase_offsets(image_positions)
aligned = tubi.generateCurrentSPCutMesh(path_pairs, phi_offsets=phi_offsets,
                                        register=True)
field_on_aligned_mesh = aligned.interpolate(field_on_original_cut_vertices)
```

`phase_offsets` accepts explicit normalized image-column positions in `[0,1]`.
It linearly interpolates the unwrapped strip offsets and requires every strip
to be valid. The example assumes columns span the unshifted chart's full s
extent uniformly. Supply the actual image correspondence for other sampling.
If fitting images of an already shifted chart, add the fitted increments to
its existing offsets on the same intermediate hoops before rebuilding from
the original cut mesh. Do not apply them twice. This process changes the chart;
it does not recalculate the longitudinal s measured before phase alignment.

The runnable `examples/texture_phase_and_piv.py` checks surface geometry, exact
seam duplication, source-field transfer and a separate large-displacement PIV
pair. It uses synthetic data and requires no plotting package.

## Algorithm and diagnostics

Overlapping strips cover both axial image edges. Their centers include columns
0 and `n_s-1`; strips at those edges are clipped. Each strip subtracts each
column's mean, normalizes its total intensity norm, and sums periodic row
cross-correlations over columns using FFTs. Positive strip-wide gain and
column-wise additive backgrounds do not change the fit. Axial shifts are not
searched. `method="integer"` selects an integer row lag; the default
`"parabolic"` refines the peak with its two periodic neighbors.

`bounds=(-.5,.5)` searches one whole turn. A narrower interval constrains the
sampled lags; peaks at either sampled limit or refined outside the interval
are flagged. Invalid strips retain NaN offsets. Flags distinguish masked
strips, low texture, tied peaks and bound-limited results. Unwrapping runs only
through adjacent valid strips and assumes phase changes below half a turn.
It cannot resolve arbitrary integer turns or shifts over half a period between
samples. No smoothing, missing-value replacement, or extrapolation is performed.

`peak_correlation` reports the normalized correlation at the integer peak;
it is not a confidence interval or an automatic quality threshold. Unique
peaks can still be wrong when texture changes or axial correspondence fails.
Texture alignment may absorb physical tissue motion. Preserve chart changes
when interpreting PIV; this operation alone does not establish material
tracking or physical velocity.

## Original source and validation

This adapts the intent of MATLAB `fitPhiOffsetsViaTexture` and
`fitPhiOffsetsFromPrevPullback`, with explicit images replacing implicit volume
projection and files. It deliberately uses 1D normalized cross-correlation,
rather than the original 2D whitened phase correlation, Gaussian filtering,
nearest-value filling, moving median, Savitzky–Golay filtering and degree-10
polynomial fit. The original wrapper references undefined variables, compares
literal `'method'` rather than the argument for its subpixel branch, and has
an iteration condition that ignores negative increments. Those behaviors are
not reproduced.

The reference generator runs the unchanged `PhaseCorrelationPotentialBound`
helper in the pinned MATLAB utilities checkout on 12 integer-shift cases,
including odd and rectangular images. Python recovers every known shift;
11 cases also agree directly with the original result. On a 32-by-79 image
shifted by -9 rows, the original reports +23 because its wrap threshold uses
the other image dimension. The fixture preserves that defect; Python returns
-9. This is an integer sign/periodicity comparison, not subpixel or full
workflow equivalence. SHA-256 hashes and source revisions accompany the fixture.
No helper implementation is copied into the Python package.

Independent tests cover direct correlation sums, gains/backgrounds, known
Fourier subpixel shifts (0.025-pixel tolerance), ambiguity, masks, bounds,
phase unwrapping and rebuilding a rotated cylinder chart with field transfer.
Biological image validation and automatic projection are still pending.
