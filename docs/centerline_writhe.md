# Centerline length and writhe

The 0.4 milestone implements the numerical portions of `measureLength` and
`measureWrithe` for existing physical centerlines. It does not extract a
centerline from a surface. Polar writhe, curve smoothing, temporal smoothing,
plots, and saved MATLAB analysis artifacts remain unsupported.

## Supply a curve and measure it

```python
from tubular import TubULAR

tubi = TubULAR(time_points=[5, 9], opts={"timeInterval": 2, "timeUnits": "min"})
tubi.register_centerline(5, xyz_at_5)
tubi.register_centerline(9, xyz_at_9)
Wr, density, dWr, length, curves = tubi.measureWrithe(
    {"omit_endpts": 0, "filter_curve": 0}
)
```

Each XYZ array is N-by-3 in physical `spaceUnits`. `register_centerline` copies
the input, computes cumulative polygonal arc length if `mss` is omitted, and
uses `mcline` as `avgpts` if that array is omitted. Registered values take
precedence over the same timepoint's MAT file. Re-registering the selected time
clears the current centerline cache. Alternatively, configure
`fullFileBase.clineDVhoop` for existing MAT files containing `mss`, `mcline`,
and `avgpts`, as in the previous milestone.

`mss` must be a physical arc-length coordinate, finite and strictly increasing.
It may have a nonzero origin. Differences between a spline's stored arc length
and its polygonal approximation are possible; `measureLength` reports both.
Unit labels are caller supplied and never imply an automatic coordinate conversion.

Both measurement methods preserve the currently selected time and caches.
They use acquisition coordinates in filenames, even when `t0` differs from zero.
No scientific result or partial series is saved to disk.

## Length results

`measureLength({"timePoints": [...]})` returns:

| Key | Meaning |
| --- | --- |
| `lengths` | Full stored path-length span, `mss[-1] - mss[0]`. |
| `polygonal_length` | Independent sum of Euclidean distances between successive XYZ points. |
| `dl` | Unsmoothed length rate per physical time unit. |
| `timepoints`, `times` | Original acquisition coordinates and physical times relative to `t0`. |
| `length_unit`, `time_unit` | The object's `spaceUnits` and `timeUnits`. |

Only `timePoints` is an accepted option, defaulting to all acquired timepoints.
No writhe calculation, curve trimming, or smoothing is needed to measure length.
Derivatives use the actual, possibly nonuniform physical time coordinates, with
one-sided endpoint differences. With one timepoint, derivatives are NaN rather
than an invented zero rate.

## Writhe results and conventions

`measureWrithe` returns five items with MATLAB-style result names:

| Result | Contents |
| --- | --- |
| `Wr` | `Levitt` and `Gauss` arrays, one value per timepoint. |
| `Wr_density` | Tuples of local density arrays for each timepoint. Levitt has one value per retained segment; Gauss one per retained point. |
| `dWr` | Unsmoothed physical-time rates for the two supported definitions. |
| `Length_t` | The full-curve length results above plus `gauss_convention`. |
| `curves` | Tuple of dictionaries containing full `ssx`, full `xyzp`, and zero-based `kept_indices` used for writhe. |

The Levitt calculation sums polygon-segment solid angles. Its density integrates
against geometric segment lengths. Gauss quadrature uses normalized finite-
difference tangents and the original endpoint-weighted `gradient(mss)` rule.
Its density integrates against those quadrature weights. It is an approximation
and converges with refinement; it need not equal the polygonal result at a
finite sampling resolution.

Both class results treat centerlines as **open curves**. No artificial closing
segment is appended. The Gauss double integral and directional polar writhe
are distinct for open curves; polar fields are absent, not filled with an
alternative calculation. For definitions and the standard `1/(4*pi)` Gauss
normalization, see Berger & Prior, *The writhe of open and closed curves*
([original paper](https://empslocal.ex.ac.uk/people/staff/mab215/home_files/Writhe.pdf)).

Supported options:

| Option | Behavior |
| --- | --- |
| `timePoints` | Nonempty increasing subset of acquired times; default all. |
| `omit_endpts` | Nonnegative integer or `(front, back)` counts; default 4 from each end. At least three points must remain. |
| `filter_curve` | Only 0 is supported and is the Python default. Positive windows raise an explicit error. |
| `flipy_centerline` | Optional reflection of stored Y coordinates; default false, as in MATLAB's measurement. It is independent of the display/loading flag `tubi.flipy`. |
| `gauss_legacy` | Default false uses standard normalization. True reproduces the original Gauss helper's arithmetic factor. |

## Deliberate corrections to the original

The source Gauss helper broadcasts a scalar dot product over three columns
before summing, tripling each term. It then divides both total and density by
an additional `2*pi`. Consequently its result is `3/(2*pi)` times the same
quadrature with standard normalization. Python defaults to the standard
formula. `gauss_legacy=True` preserves the original values explicitly, and
`Length_t["gauss_convention"]` records which was used. Neither mode changes
the independent Levitt calculation.

The original `omit_endpts=k` indexes MATLAB points `k:N-k`, omitting `k-1`
points at the front and `k` at the back; zero is invalid indexing. Python uses
actual omission counts: `k` at each end, including a valid zero setting. To
reproduce a legacy selection for `k>=1`, pass `(k-1, k)`. Length always uses
the full stored curve, regardless of writhe trimming.

Original length uses `max(mss)`; Python subtracts its origin. Original temporal
rates apply robust smoothing, a causal seven-point filter, and a unit-frame
gradient. Python returns explicitly unsmoothed physical-time derivatives.
It does not claim parity with those filtered rates. The full MATLAB method
also subtracts `t0` before constructing input filenames; Python uses the actual
acquisition coordinates so a nonzero reference time does not change input files.

The original curve filter includes a degree-21 polynomial fit and resampling.
This is not silently replaced by a different spline or smoothing algorithm.
Until it is validated, `filter_curve=0` is the only supported setting.

## Pure numerical API and geometric checks

```python
from tubular.curves import curve_arclength, writhe_levitt, writhe_gauss

s = curve_arclength(xyz)
W_polygon, segment_density = writhe_levitt(xyz, closed=False)
W_sampled, vertex_density = writhe_gauss(xyz, s)
```

The standalone Levitt function also supports `closed=True`; a repeated final
copy of the first point is removed before creating the closing segment. The
standalone Gauss function is open-curve quadrature only. Both support straight
and planar curves, which have zero writhe.

Writhe rejects repeated neighboring points, immediate retracing, nonadjacent
segment crossings, and nearly touching segments (within `1e-12` of the maximum
coordinate extent measured from the first point). Segment distances use cross
products to avoid losing shallow-angle intersections through cancellation.
Clamping inverse-sine inputs avoids complex results from floating-point roundoff;
numerically coplanar pairs contribute zero. These checks are numerical, not an
exact-arithmetic topological certificate. The algorithms take quadratic time
in the number of curve points; large biological series have not been benchmarked.

## Validation

```sh
python -m pytest tests/test_curves.py
python examples/centerline_writhe.py
```

The committed fixture runs unchanged original `writheLevitt`,
`writheGaussIntegral`, and `aux_compute_writhe` on nonuniformly sampled helices,
a reflected/scaled time series, and a nonplanar closed polygon. It compares
totals, densities, retained curves, and full lengths; the helper's polar and
filtered-rate fields are intentionally not ported. Hashes and MATLAB version
are recorded in `tests/data/writhe_provenance.json`.

`writheLevitt` depends on `linesegments.m`, which is absent from this TubULAR
checkout. The reference generator reads the existing original helper from
`MitchellLabCode/matlab`, whose revision and file hash are also recorded. Neither
MATLAB checkout is modified. To regenerate:

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_writhe_reference('/path/to/tubular', ...
    '/path/to/matlab/curve_functions', ...
    '/path/to/pyTubULAR/tests/data/matlab_writhe_reference.mat');
```

Independent Gauss-Legendre integration over line-segment pairs verifies the
polygonal result without using solid-angle arithmetic. Further tests cover
rigid motion, scaling and density units, reflection, reversal, planar curves,
Gauss quadrature refinement, closed-endpoint handling, near-parallel crossings,
input ownership, nonuniform acquisition intervals, single-time derivatives,
nonzero `t0`/path-length origins, and unsupported options. These are synthetic
tests; biological reproduction remains a later validation gate.
