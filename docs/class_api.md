# TubULAR class API and migration guide

```python
from tubular import TubULAR, Mesh, MeshSource, MethodNotPortedError
```

Version 0.9.0a1 extends the stateful experiment object using MATLAB method
names. Of 200 original callable method names, 63 are **adapted** and 137 are
**not ported**. Adapted means working within the scope below, not every legacy
argument, format, return value, or side effect. The constructor is additional
to this count. The original MATLAB repository is only a reference source.

```python
TubULAR.available_methods()                # Only working MATLAB-named methods
TubULAR.method_status("getCurrentData")     # Source, static/hidden flags, status
TubULAR.method_status()                     # All 200 coverage records
```

Pending methods are visible in `dir(tubi)` and `help(tubi.method)` and raise
`MethodNotPortedError` immediately. They do not return substitute scientific
results. [Complete coverage](method_coverage.md) lists every name. Native
helpers such as `register_mesh` are not counted as MATLAB ports.

## Construct and select an experiment

```python
tubi = TubULAR(
    time_points=[0, 2, 5],
    opts={"timeInterval": 3, "timeUnits": "min", "spaceUnits": "um", "t0": 2},
)
tubi.setTime(5)
elapsed, label = tubi.trueTime(5, return_label=True)  # (9.0, "9 min")
tubi.clearTime()
```

Construction is keyword-only: `TubULAR(time_points=..., opts=...,
mesh_sources=..., data_sources=...)`. Access acquired times through
`tubi.timePoints`. Inputs are copied and timepoints are read-only.
Version 0.5 removes the `xp` argument, simulated `xp.fileMeta`/`xp.expMeta`
namespace, and metadata-derived `stackResolution` and `dataDir`. Supply the
APDV scale explicitly with `opts={"resolution": ...}` and register image/mesh
paths directly. ImSAnE backwards compatibility is not a project requirement.

Timepoints are finite, strictly increasing acquisition coordinates. `setTime`
only accepts an acquired timepoint. `trueTime(tp)` computes
`(tp - t0) * timeInterval`, so missing frame numbers do not collapse time gaps.
Use physical timestamps with `timeInterval=1` for irregular acquisition times.
`trueTime` also accepts arrays; `return_label=True` requires a scalar.
`div60=True` divides the value by 60, converting minutes to hours or seconds to
minutes when a label is requested.

The constructor accepts these options:

| Options | Meaning / default |
| --- | --- |
| `timeInterval` (or `timeinterval`) | Positive time per acquisition-coordinate unit; default 1 |
| `timeUnits`, `spaceUnits` | Labels; defaults `min`, `um`. Labels do not rescale mesh coordinates. |
| `t0` | Reference time; otherwise load `fileName.t0` if present, otherwise first timepoint |
| `meshDir` | Directory for the default `t0_referenceTime.txt` lookup; default current directory |
| `nU`, `nV` | Grid metadata; defaults 100. They do not resample supplied meshes. |
| `flipy` | Reflect APDV Y after rotation, translation, and scaling; default false |
| `resolution` | Positive scalar APDV length per coordinate unit; see frame section |
| `rot`, `trans` | Explicit APDV rotation and translation; supply both or neither |
| `normalShift` | Bounds padding parameter in raw-coordinate units; default 0. Does not displace loaded meshes. |
| `fullFileBase`, `fileName` | Explicit filename mappings, described below |
| `makeDirs` | Only false is supported; constructor creates no files or directories |

Other legacy options raise `NotImplementedError`. `t0set()` returns the current
reference time; `t0set(value)` changes it in memory without writing or reloading
a file. Time selection starts unset. Changing time, including assigning
`currentTime`, clears all current mesh/data/velocity/segmentation/strain/centerline
caches. Selecting the same time retains them; `clearTime()` clears them explicitly.

## Load existing meshes by stage

```python
tubi = TubULAR(time_points=[4])
tubi.register_mesh(4, MeshSource("mesh_0004.mat", "spcutMeshSm", "um"), kind="spcutMeshSm")
tubi.setTime(4)
mesh = tubi.getCurrentSPCutMeshSm()   # Load once, then reuse the cache
mesh = tubi.loadCurrentSPCutMeshSm()  # Force a reload from the registered source
vertices, faces, uv = mesh.v, mesh.f, mesh.u
```

`Mesh` exposes read-only aliases `v`, `f`, `u` as well as `vertices`, `faces`,
`uv`. **Faces are zero-based**, including `mesh.f`. MAT import performs the
one-based conversion once. Only geometry, UV, and length units are retained;
additional fields such as cut paths, normals, gridded UV metadata, or tracked
correspondences are not imported. Units describe the stored coordinates.

Register a validated `Mesh`, a native `.npz` path, a `.mat` path containing
top-level `v/f/u`, or a `MeshSource` specifying a named MATLAB struct and units.
NPZ stores units itself; an explicit contradictory `MeshSource` unit is rejected.
File-backed data are loaded lazily. PLY/OBJ and MATLAB v7.3 are not supported yet.

| Stage key | Getter / loader suffix |
| --- | --- |
| `rawMesh` | `RawMesh` |
| `alignedMesh` | `AlignedMesh` |
| `cylinderMesh` | `CylinderMesh` |
| `cylinderMeshClean` | `CylinderMeshClean` |
| `cutMesh` | `CutMesh` |
| `uvcutMesh` | `UVCutMesh` |
| `spcutMesh` | `SPCutMesh` |
| `spcutMeshSm` | `SPCutMeshSm` |
| `spcutMeshSmRS` | `SPCutMeshSmRS` |
| `spcutMeshSmRSC` | `SPCutMeshSmRSC` |
| `ricciMesh` | `RicciMesh` |

Each suffix has `getCurrent…()` and `loadCurrent…()` methods. They are zero-argument
loaders in this release: UV-generation options and Ricci iteration selection are
not supported. Missing sources raise an error; no automatic mesh generation is
attempted. A registered source for a timepoint takes precedence over filename
patterns. Re-registering the selected stage invalidates that stage's cache.

Bulk registration is available through `mesh_sources={kind: {time: source}}`.
Alternatively use `opts={"fullFileBase": {"spcutMeshSm": "/data/mesh_%04d.mat"}}`.
Patterns accept a single MATLAB `%04d`/`%06d` integer placeholder or Python
`{time:04d}`/`{timepoint}`. Resolve multiple legacy filename parameters yourself
and register the resulting paths. Paths are literal, relative to the current
directory if not absolute; `meshDir` does not prefix them automatically.

The `rawMesh` pattern key is `mesh`; other keys match their stage. For MAT
patterns, cut/UV/SP/Ricci stages select the struct matching their stage name;
raw/aligned/cylinder stages expect top-level arrays. Use explicit `MeshSource`
registration whenever the file differs. Named MAT selection never applies to NPZ.

## Images and pullbacks

```python
import numpy as np

tubi.register_data(4, np.load("volume.npy", allow_pickle=False))
channels = tubi.getCurrentData(False)  # Tuple of read-only (Z, Y, X) arrays
projection = tubi.mip(4, 1)           # Project Z; MATLAB-style dim=1
```

Register one real 3D array, a `.npy` path, or a sequence of channels. Arrays are
copied on registration; files load on demand. All channels at a timepoint must
have the same shape. A 4D array is not interpreted implicitly. Channels use
zero-based Python indices; MIP `dim` and optional `pages` retain one-based
MATLAB numbering. With ZYX storage, `dim=1,2,3` projects Z,Y,X respectively.
`mip` selects its requested timepoint. `loadCurrentData()` forces a reload.

Data are not automatically resampled, transposed, masked, or intensity-scaled.
`getCurrentData()` defaults to the identity adjustment limits 0/100; changing
these limits requires `adjustIV=False` until the LUT port is validated. Bioformats,
TIFF, HDF5, implicit isotropic resampling, and `adjustIV` are pending.

`sample_current_pullback(u, v, mesh_kind="spcutMeshSm", channel=0,
spacing_xyz=(1,1,1), origin_xyz=(0,0,0))` calls the tested supplied-chart sampler.
It returns a result with `image`, `xyz`, `valid`, and `face_indices`. Mesh XYZ
and volume XYZ must already share a frame and units. This helper does not align
the volume to APDV, create a chart, or implement MATLAB's normal-stack projection.
Progress and cancellation callbacks are supported by the sampling core.

## APDV transforms and measurements

```python
tubi.set_apdv_frame(np.eye(3), [0, 0, 0], resolution=0.5, flipy=False)
aligned = tubi.xyz2APDV([[10, 20, 30]])
raw = tubi.APDV2xyz(aligned)
```

Positions use `(xyz @ rot.T + trans) * resolution`, then optional Y reflection.
Vectors use `dx2APDV` / `APDV2dxyz` without translation. Inputs are N-by-3, and
rotation must be proper orthonormal. Supply a frame explicitly or define it
from landmarks with `computeAPDVCoords`, described below;
`getRotTrans()` can instead read explicit `fileName.rot` / `fileName.trans` text
paths. No anatomical landmarks are inferred. `resolution` is an explicit scalar,
defaulting to 1. Convert anisotropic voxel coordinates to physical XYZ before
aligning a mesh and its landmarks; the class does not resample images.
The image sampler separately accepts all three physical voxel spacings.

`setAPDVOptions` and `setEndcapOptions` copy option dictionaries into memory;
their downstream estimation, load, and save workflows remain pending.

`computeAPDVCoords(anterior, posterior, dorsal, origin=None, resolution=None,
flipy=None)` defines and stores a frame from supplied raw XYZ points. The last
three arguments are keyword-only. It returns five copied arrays: the three
input landmarks, rotation, and translation. Origin defaults to anterior;
X points toward posterior and Z toward dorsal perpendicular to X.

`alignMeshesAPDV(time_points=None, register=False, preserve_orientation=True)`
takes keyword-only arguments and transforms registered raw meshes into new
snapshots. It returns a dictionary of meshes, timepoints, transform parameters,
units, and a per-face corner mapping. Registration as `alignedMesh` is explicit
and happens only after the whole batch succeeds. Selection is preserved.
Reflection swaps triangle corners by default to preserve outward/inward
orientation; `preserve_orientation=False` retains MATLAB's original face order.
Neither method performs landmark detection, endpoint displacement, smoothing,
or plotting. See [alignment details](landmark_alignment.md) for correspondence,
reference precision, and the intentional change from the MATLAB return schema.

`measureSurfaceAreaVolume(options=None)` returns a dictionary of `timepoints`,
physical `times`, `time_unit`, `area`, `volume`, `signed_volume`, `length_unit`,
and `mesh_kind`. Supported options are `mesh_kind` (default `alignedMesh`),
`timePoints` (default all), and `allow_open` (default false). Area/volume units
are the squared/cubed mesh length unit. Mesh units must agree across times.
Open surfaces require `allow_open=True` and return NaN volume. This entry point
computes raw measurements; MATLAB's smoothing, rejection filters, figures, and
saved analysis files are not replicated. Selection and current caches are preserved.

`measure_current_geometry(mesh_kind="spcutMeshSm")` returns the first and second
fundamental forms of a supplied chart. This native helper is separate from the
vertex estimator in `measureCurvatures`.

## Annular parameterization (0.7)

`generateCurrentCutMesh(seam_path=None, *, seam_endpoints=None, aspect_ratio=1.0,
negative_weights="clamp", register=False)` cuts the selected `cylinderMeshClean`
and solves for a new UV chart. The supplied zero-based vertex path must connect
its two boundary loops through interior edges. The result contains a mapped
`mesh`, `path_pairs`, `cut_to_uncut`, `uncut_to_cut`, and `embedding` diagnostics.
With `register=True`, subsequent `getCurrentCutMesh()` calls return that mesh.
Failed solves leave the registered cut mesh intact. Retain the result separately
for its correspondence metadata.

The algorithm ports the custom-path cut and Dirichlet `flattenAnnulus` solve;
automatic landmark selection, Ricci/MVC, optimal aspect ratio, reflected charts,
normal shifts and file output are not included. The constructor's `flipy`
does not alter this chart. Negative cotangent weights use the MATLAB 0.01 clamp
unless the caller chooses `negative_weights="error"`. See
[parameterization conventions and validation](annular_parameterization.md)
before using a generated chart for periodic sampling.

## Periodic UV-grid resampling (0.8)

`generateCurrentUVCutMesh(path_pairs, *, n_u=None, n_v=None, normals=None,
interpolation="mesh", preserve_orientation=True, register=False)` samples the
selected `cutMesh` on a regular periodic grid. Pass the source's explicit seam
pairs, such as `cut.path_pairs` from `generateCurrentCutMesh`. Sizes default to
the object's `nU`, `nV`; overrides affect this result only. The output `UVGrid`
contains its `mesh`, exact output seam pairs, optional normalized interpolated
normals, source vertex indices, weights, periodic shifts and original query UV.
`grid.interpolate(source_vertex_field)` transfers periodic scalar/vector fields.

Default interpolation preserves original triangles and winding. Explicit
`interpolation="matlab_scattered"` uses MATLAB-style three-tile Delaunay
interpolation; `preserve_orientation=False` uses MATLAB's grid face order.
Co-circular Delaunay ties can differ between libraries. No mode extrapolates,
infers normals, transforms to APDV, or writes files. `register=True` installs
the output as `uvcutMesh` only after all checks succeed. Keep the result for its
correspondence; the registered `Mesh` alone does not carry that metadata. See
[UV-grid scope and comparisons](uv_grid_resampling.md).

## Surface curvature and Beltrami measurements (0.6)

`measureCurvatures(mesh_kind="spcutMeshSmRSC", time_points=None)` takes
keyword-only arguments and returns a time-series dictionary with a tuple of
`CurvatureResult` objects. Each contains vertex principal values/directions,
mean and Gaussian curvature, normals, an ambient shape operator, corner/vertex
areas, and a boundary mask. Meshes need no UV chart and can have different
vertex counts across time. The estimator retains MATLAB's negative mean
curvature for outward convex spheres. Open neighborhoods are flagged, never
implicitly glued or capped. See [curvature details](surface_curvature.md).

`measureBeltramiCoefficient(reference, vertex_positions, time_points=None)`
uses a fixed reference `Mesh` with UV and an explicit mapping from timepoints
to N-by-3 coordinates in reference vertex order. The chart counterpart,
`measureBeltramiCoefficientPullbackToPullback(reference, target_charts,
time_points=None)`, requires N-by-2 chart coordinates in that same order.
For both, `time_points` is keyword-only. Results include T-by-F raw complex
`mu`, `dilatation`, and a separate `reference_mu`, plus time metadata.
No tracking or correspondence is inferred from registered mesh stages.
Planar folds/reflections are rejected; a 3D metric cannot detect reflection
or global self-intersection. See [Beltrami details](beltrami.md).

These methods preserve selection/caches and write no files. Smoothing,
pathline construction, Ricci flow, automatic reference-offset subtraction,
and plotting are outside this numerical scope.

## Surface queries and field interpolation (0.3)

```python
# First register a spcutMeshSmRS whose XYZ vertices are already in APDV.
tubi.setTime(4)
xyz, face_ids, chart, weights = tubi.uv2APDV(
    [[0.25, 0.3], [0.75, 0.6]], return_details=True,
)
```

`uv2APDV(uv, coordSys="spsmrs", umax=1, vmax=1)` uses the supplied triangle
connectivity to interpolate each query onto the existing 3D surface. It
normalizes the chart's U values by their maximum and multiplies by `umax`, as
in MATLAB. It neither translates U's origin nor rescales V. Only `spsmrs` is
supported: the legacy `sprs` branch requires a separate `sphi` field that the
current `Mesh` schema does not store. MATLAB ignores `vmax` in this method;
Python rejects non-unit values explicitly. This does not constrain the stored
V range or the query V values to [0, 1].

Return value is N-by-3 XYZ by default. `return_details=True` returns the four
items shown above. The third item is the normalized chart `Mesh`, replacing
the MATLAB triangulation object. Face IDs are zero-based and weights follow
that face's vertex order. Out-of-chart points have face ID -1 and NaN XYZ and
weights (MATLAB uses NaN face IDs and zero weights outside). Holes are preserved;
there is no re-triangulation, extrapolation, or periodic wrapping. No second
APDV transform or reflection is applied to the already aligned vertices.

Shared edges and vertices use the first incident face in the supplied ordering;
MATLAB may select a different adjacent face there. This does not change
continuous vertex interpolation. Overlapping interiors or nonadjacent faces
covering the same query raise an error, even if their XYZ coordinates agree.
These are query-local checks, not a global proof of an overlap-free chart.

The reusable Python operation supports arbitrary scalar and vector fields:

```python
from tubular.surface import locate_chart

located = locate_chart(mesh, uv_points)  # Uses the mesh's actual UV units/range.
positions = located.interpolate(mesh.v)
vertex_values = located.interpolate(values_per_vertex)
face_values = located.interpolate(values_per_face, location="faces")
```

Vertex fields use barycentric interpolation; face fields are piecewise
constant and use the same first-face rule on shared boundaries. Values must
be finite, with one row per vertex/face. Queries and source meshes are not
mutated. `located.valid`, `face_indices`, and `barycentric` retain coverage
information for callers. The query spatial index limits per-face candidates;
large biological meshes have not yet been performance-benchmarked.

`interpolateOntoPullbackXY(XY, scalar_field, options)` is a **different** operation:
it reproduces the existing rectilinear, tiled-field interpolation used by the
MATLAB class. It does not use a triangular chart. Its supported options are:

| Option | Supported behavior |
| --- | --- |
| `Lx`, `Ly` | Required image width/height, each greater than 1; XY is one-based. No image file is guessed. |
| `imCoords` | `sp_sme` only; otherwise defaults to `tubi.piv.imCoords` (initially `sp_sme`). |
| `sfLocation` | `vertices` only; the original face-field branch is incomplete and remains unsupported. |
| `iMethod` | `linear` (default, bilinear on this grid) or `nearest`. Exact midpoint ties choose the upper node, matching MATLAB. |

The field must contain `nU*nV` values, supplied as a flat U-fastest vector,
MATLAB column vector, or `(nU, nV)` array. The original tiling is retained:
`[field[:, :-1], field, field[:, 1:]]`. X nodes span 1 to Lx; Y nodes span
`1 - Ly/4` to `1.25*Ly` over `3*nV - 2` rows. Outside that entire tiled grid,
nearest-node extrapolation applies in both axes. This is not infinite periodic
wrapping. The seam values are not smoothed or repaired.

The source contains an `imCoords`/`coordSys` typo and never applies supplied
`sfLocation` or `iMethod`. Python honors the documented option names and rejects
unsupported branches. Default linear outputs match the actual original class;
the newly honored nearest option is checked against MATLAB's own
`griddedInterpolant`, including midpoint ties.

## Experiment bounds and existing centerlines (0.3)

`measureXYZLims(options=None)` returns `(raw, pixels, physical, buffered)`, each
a 3-by-2 array with XYZ rows and minimum/maximum columns. It loads registered
raw mesh sources over `timePoints` (all acquired timepoints by default), applies
the configured APDV transform including Y reflection, and computes bounds from
the transformed vertices. The raw coordinates must be in the frame expected
by that transform; raw mesh unit labels must agree across the series.

`physical` is in the configured `spaceUnits`; the legacy `um` property names do
not enforce micrometers. `pixels = physical / APDV.resolution`. Padding on each
side is `2 * abs(normalShift) * APDV.resolution`, matching the original. The
optional `normalShift` overrides the object's value for this calculation.
Only `timePoints` and `normalShift` options are supported.

Results are copied into `plotting.xyzlim_raw`, `xyzlim_pix`, `xyzlim_um`, and
`xyzlim_um_buff` after the entire calculation succeeds. Current time and mesh
caches are preserved. `getXYZLims()` recomputes from current sources and frame,
so changes do not leave stale bounds. Unlike MATLAB, these methods do not load
or write bound text files; old `fileName.xyzlim_*` entries are not consulted.

`getCurrentClineDVhoop()` reads a registered centerline, or loads top-level `mss`,
`mcline`, and `avgpts` from the selected time's existing MAT file. Set
`fullFileBase.clineDVhoop` to its filename
pattern. `mss` must be an increasing row/column vector matching the number of
N-by-3 `mcline` points. `avgpts` is a separate M-by-3 array; M may differ from N.
Both curves must have at least two points and finite coordinates. A preexisting
centerline is required: extraction, smoothing, and spline fitting remain pending.

Returns `(mss, mcline, avgpts)`, with `mss` flattened to a Python vector. The
stored coordinates are assumed to be physical coordinates before optional Y
reflection; no additional rotation or scale is applied. Each call reloads the
file (unless a registered source takes precedence) and reflects once if
`flipy=True`, matching MATLAB. Independent read-only
copies enter `currentCline`, and time changes clear that cache. Invalid files
raise before updating any centerline cache entry.

## Centerline length and writhe (0.4)

`register_centerline(timepoint, mcline, mss=None, avgpts=None)` supplies an
existing physical curve without a MAT file. This native helper copies inputs;
`mss` defaults to cumulative polygonal length and `avgpts` to `mcline`. Coordinates
are stored before optional Y reflection, as for the MAT reader.

`measureLength(options=None)` returns a dictionary containing `lengths`,
`polygonal_length`, `dl`, `timepoints`, `times`, `length_unit`, and `time_unit`.
The only option is `timePoints`, defaulting to all acquired times. It measures
the full stored path-length span, independently reports the geometric polyline
length, and estimates the unsmoothed rate using physical acquisition times.

`measureWrithe(options=None)` returns `(Wr, Wr_density, dWr, Length_t, curves)`.
Supported entries in the first three dictionaries are `Levitt` and `Gauss`;
polar writhe is not supplied. Supported options are `timePoints`, `omit_endpts`
(default 4 from each end), `filter_curve` (only 0, the Python default),
`flipy_centerline` (default false), and `gauss_legacy` (default false).

Both methods preserve selection and current caches and write no files. Results
are numerical series, without MATLAB's smoothing or plotting side effects.
Some original arithmetic and indexing defects are corrected deliberately;
see the [full conventions and comparison guide](centerline_writhe.md) before
comparing datasets. A single timepoint has NaN time derivatives.

## Image-coordinate helpers and intentional differences

`XY2uv`, `uv2XY`, and `dXY2duv` accept either an image or `(width, height)`;
the pair is **not** a NumPy `(height, width)` shape. XY locations are one-based,
and UV extents are positive `umax`, `vmax` (default 1). Double cover defaults true.

| Method | Documented behavior |
| --- | --- |
| `XY2uv` | Correct double-cover offset is `-0.5*vmax`; `legacy=True` reproduces MATLAB's `-0.5` for reference comparisons. |
| `dXY2duv` | Displacements transform linearly, including zero to zero. `legacy=True` reproduces the original erroneous `-0.5` Y offset. |
| `uv2XY` | Inverts the corrected `XY2uv`, including non-unit `vmax`. |
| `clipXY` | Clips X, wraps Y by period `Ly-1` with repeated wraps supported. `legacy=True` reproduces the asymmetric one-step original. |
| `extendImage2DoubleCover` | Concatenates bottom half, whole image, top half with MATLAB half-up rounding. Preserves dtype instead of forcing uint8; histogram equalization is unsupported. |
| `singleToDoubleCover` | Adds `round(Ly/2)` to Y, where Ly is the single-cover height. |
| `doubleToSingleCover` | Preserves legacy recentering within the middle half of the double-cover coordinates. Ly is the double-cover height; this is not the inverse of `singleToDoubleCover`. |
| `dvAverageNematic` | Averages nonnegative magnitudes and doubled angles across axis 1; angles returned modulo pi. |
| `quarterIndicesDV` | Preserves the original inclusive, overlapping groups in dorsal/ventral/left/right order, converted to zero-based indices. Requires nV >= 4. The dorsal set extends through the first 3/8, as in the source; these are not four disjoint quarters. |
| `invertRotation` | Validates a proper rotation and returns its transpose. |

Corrected defaults are explicit Python behavior changes. Use `legacy=True` only
when reproducing the documented legacy arithmetic, not as a general guarantee
of MATLAB workflow compatibility. See [validation evidence](validation.md).

## Segmented mesh to s-phi and initial PIV (0.9)

`sliceMeshEndcaps(anterior_vertices, posterior_vertices, register=False)` adapts
explicit vertex removal on the selected closed rawMesh. It returns an annulus
and source maps, optionally registered as cylinderMeshClean.
`generateCurrentCutMesh` now also accepts `seam_endpoints=[start,end]` instead of
`seam_path`, finding an interior shortest edge path between opposite boundaries.

`generateCurrentSPCutMesh(path_pairs, n_u=None, n_v=None, phi_offsets=None,
seam_weight="unique", register=False)` adapts the geometric s-phi stages on
the selected cutMesh. It returns SPhiGrid, optionally registering spcutMesh.
XYZ is already in the desired length units, s is dimensional, and phi uses
turns. It does not estimate phase offsets between timepoints. See the
[complete workflow and correspondence guide](sphi_parameterization.md).

`measurePIV2d(image_a, image_b, window_size=32, step=None, dt=1,
correlation="circular", mask=None)` measures an explicitly supplied image pair.
It does not require time selection. It returns image-plane pixel displacements
and validity flags, with velocities divided by explicit dt. **Only one pass**
is implemented; the method name does not imply the MATLAB four-pass pipeline.
See [PIV scope, validation and package comparison](piv.md).
