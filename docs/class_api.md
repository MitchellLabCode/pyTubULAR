# TubULAR class API and migration guide

```python
from tubular import TubULAR, Mesh, MeshSource, MethodNotPortedError
```

Version 0.2.0a1 introduces a stateful experiment object using MATLAB method
names. Of 200 original callable method names, 46 are **adapted** and 154 are
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

Alternatively, `TubULAR(xp, opts)` accepts a Python mapping with
`xp["fileMeta"]["timePoints"]`, optional `stackResolution`/`dataDir`, and optional
`expMeta`. This is not an ImSAnE object or a reader for a saved MATLAB object.
Metadata is copied. Access uses `tubi.xp.fileMeta.timePoints` after construction.
Do not supply both `time_points` and metadata timepoints.

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
| `fullFileBase`, `fileName` | Explicit filename mappings, described below |
| `makeDirs` | Only false is supported; constructor creates no files or directories |

Other legacy options raise `NotImplementedError`. `t0set()` returns the current
reference time; `t0set(value)` changes it in memory without writing or reloading
a file. Time selection starts unset. Changing time, including assigning
`currentTime`, clears all current mesh/data/velocity/segmentation/strain/centerline
caches. Selecting the same time retains them; `clearTime()` clears them explicitly.
`xp.tIdx` positions are zero-based Python positions.

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
rotation must be proper orthonormal. The frame must be supplied explicitly;
`getRotTrans()` can instead read explicit `fileName.rot` / `fileName.trans` text
paths. No anatomical landmarks are inferred. With isotropic metadata,
`stackResolution` supplies the scalar resolution; anisotropic metadata requires
an explicit scalar because this class does not perform legacy image resampling.
The image sampler separately accepts all three physical voxel spacings.

`setAPDVOptions` and `setEndcapOptions` copy option dictionaries into memory;
their downstream estimation, load, and save workflows remain pending.

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
unported full `measureCurvatures` workflow.

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
