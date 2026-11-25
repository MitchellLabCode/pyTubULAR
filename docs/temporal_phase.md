# Geometric phase alignment between surface maps

Version 0.10 extends `generateCurrentSPCutMesh` to fit angular offsets against
an explicitly supplied reference `SPhiGrid`. It supports forward or backward
comparisons and nonconsecutive timestamps; the caller chooses the reference.
It does not guess the previous frame, read files or change time selection.

```python
tubi.setTime(0)
reference = tubi.generateCurrentSPCutMesh(reference_cut.path_pairs)
tubi.setTime(7)
current = tubi.generateCurrentSPCutMesh(current_cut.path_pairs,
                                      reference=reference, register=True)
fit = current.phase_alignment.fit
print(fit.rms_before, fit.rms_after)
```

Register each timepoint's cut mesh before this example. Keep the complete
result objects: `getCurrentSPCutMesh()` returns only the registered mesh.
`examples/temporal_phase_alignment.py` is a runnable three-frame synthetic
example, including nonconsecutive times and exact seam checks.

## Meaning of the fit

For each current intermediate U ring, the fit minimizes the sum of squared
3D distances between `current(phi + offset)` and `reference(phi)`. New chart
coordinates remain `phi = V - offset`; positive offsets mean sampling at larger
current V to match a reference phase. Offsets use turns, not radians.

Current hoops come from the first sampling stage, before phase shifts or final
s-phi resampling. Reference hoops are sampled on the reference's final mesh at
the current normalized `ring_s`. This default assumes equal fractions of axial
surface length correspond. Override it with `reference_s`, one strictly
increasing normalized reference s value per current ring, beginning at 0 and
ending at 1. It is an explicit correspondence choice, not an axial motion
estimate. Different reference grid sizes are supported through interpolation.

Input XYZ must already share a coordinate frame and physical length units;
different unit labels are rejected. The routine does not estimate APDV frames,
rigid motion or centerline-nearest correspondences. It compares geometry, so it
can remove a real physical rotation as well as an arbitrary chart-origin shift.
**This is chart stabilization, not material tracking or a tissue velocity.**
Texture matching and independently established material correspondences are
still needed to interpret later PIV measurements.

Each current hoop is a polygon sampled at uniform phase increments, including
one duplicate endpoint. Linear interpolation within its segments makes the
distance objective quadratic between successive phase knots. Python minimizes
every interval in the requested bounds, including endpoints, then selects the
global minimum. The reference samples are held fixed. The core does not
resample a hoop by arclength on each trial or assume a unimodal objective.

`phase_bounds` defaults to `(-0.5, 0.5)` turns when a reference is supplied.
An explicit interval must have positive width at most one turn and lie inside
`[-1,1]`. Geometrically equivalent endpoints of a full-period interval are not
distinct solutions. Adjacent fitted phases are unwrapped by integer turns
before constructing the final chart. This avoids an artificial full-turn jump
at the phase branch cut, but assumes adjacent rings' phase changes are less
than half a turn. No smoothing, prediction in time or motion threshold is used.

The s calculation still precedes angular fitting, as in MATLAB. Its estimated
surface lengths depend on the initial hoop phases. Fitting phi does not
recompute s or make it independent of a twisted initial chart. The final grid
also introduces the approximation described in the [s-phi guide](sphi_parameterization.md).
Inspect sampling resolution and real-data overlays before using the result.

## Diagnostics and rejected fits

`current.phase_alignment` retains:

- `fit.raw_offsets`: bounded minimizers, one per intermediate current ring.
- `fit.offsets`: the same minimizers unwrapped along the axis; these may differ
  from bounded values by integer turns.
- `fit.aligned_hoops`, `residuals`, `rms_before`, `rms_after`: the actual 3D
  distance measurements in XYZ units. Residual arrays omit the duplicate endpoint.
- `fit.ambiguous`: competing local minima within a scale-normalized objective
  tolerance. This is a numerical diagnostic, not a statistical confidence bound.
- `fit.at_bounds`: a minimum limited by a search interval shorter than one turn.
- `reference_sampling` and `reference_s`: the explicit reference mesh sampling
  weights and axial correspondence used for the comparison.

Residuals describe the fitting hoops, before final s-phi regridding. No guarantee
of small residuals is implied by uniqueness: shape change or a poor axial match
can produce a unique but poor fit. Inspect residuals in physical units.
Field transfer through `current.interpolate` continues to use both current
mesh sampling maps; reference sampling is diagnostic, not a material field map.

Ambiguous or bound-limited fits raise before class registration, preserving the
previous mesh. To inspect such a fit without applying it:

```python
from tubular.sphi import fit_sphi_phase

provisional = tubi.generateCurrentSPCutMesh(current_cut.path_pairs)
alignment = fit_sphi_phase(provisional, reference, bounds=(-0.1, 0.1))
print(alignment.fit.ambiguous, alignment.fit.at_bounds)
```

`fit_sphi_phase` always estimates absolute offsets relative to the current
result's original first-sampling chart, even if that result already has applied
offsets. It never silently composes corrections. The lower-level
`tubular.phase.fit_hoop_phase` accepts explicitly corresponding arrays of shape
`(n_hoops,n_phi,3)` with identical shapes, uniform phase sampling and exact
duplicated endpoints. It returns diagnostics without rejecting ambiguous fits.

Even an ordinary circle can become ambiguous after noncommensurate polygonal
resampling. For example, a 24-segment reference grid made from a 16-segment
polygon omits some of the original corners; fitting this changed shape can
produce two equally good small shifts. The tests exercise this case. Preserving
source corners with a commensurate angular resolution resolves that synthetic
example; it is not a general prescription for biological surfaces.

## MATLAB reference and deliberate differences

Source: TubULAR `9bd49db`, `phiOffsetsFromPrevMesh`,
`fitPhiOffsetsFromPrevMesh`, and the geometric branch of
`generateCurrentSPCutMesh`. `scripts/generate_phase_reference.m` runs the
**unchanged** `phiOffsetsFromPrevMesh` with `resample_hoops=false`, no previews,
one output and straight centerlines. Its original centerline matching is then
independently reproduced through triangle interpolation. Three cases contain
33 hoop fits: an exact vertex shift, a fractional shift and axially varying
phase. Pinned source, fixture and generator hashes are recorded in
`tests/data/phase_provenance.json`.

On the tested platform, Python's bounded offsets differ from MATLAB by at most
`1.2e-16` turns and summed squared residuals by `2.1e-17`. These are selected
unimodal examples, not a promise that a global optimizer reproduces every
local result of MATLAB's single `fminbnd` search.

The reference harness requests one output because the original non-resampled
branch tries to assign nV residual values into nV-1 slots. Python returns
correctly sized residuals. Other deliberate differences are explicit:

- The lower-level `seam_weight="matlab"` reproduces duplicate-endpoint weighting
  for parity. The default and class integration count each physical point once.
- Axial correspondence is explicit or normalized-s based. The original
  centerline-nearest matching, with its exact-hit edge cases, is not copied.
- Global piecewise-quadratic search replaces a single bounded local search.
- Circular unwrapping replaces arithmetic smoothing of wrapped offsets.
- Iterative per-trial arclength resampling, geometric/texture combination,
  smoothing filters, image output and anatomical tracking remain pending.

Independent tests cover known polygon shifts, sign, branch crossing, complete
global-search comparisons, rigid transforms, scaling, frame translation,
seam weighting, ambiguous geometry, bound-limited fits, correspondence,
different grid sizes, nonconsecutive times and failed-registration preservation.
Validation on biological data remains outstanding. The separate 0.11 extension
adds [texture phase alignment](texture_phase.md) and a native integer-window-shift
[PIV driver](piv.md); neither establishes full material tracking.
