# Architecture and path to a GUI

## Scientific boundary

TubULAR's current MATLAB object combines scientific calculations, experiment
state, file naming, interactive choices, and plotting. Python should separate
these responsibilities. A scientific result must not depend on which viewer
is open, which timepoint is selected, or whether a preview was rendered.

The first release target is a reproducible workflow from existing segmented
surfaces to validated pullbacks and measurements. Raw-image segmentation can
be integrated through adapters after that workflow is reliable. The present
implementation only reaches measurements and sampling on *existing* charts.

```mermaid
flowchart TD
    GUI[Desktop GUI: future napari plugin] --> Services[Application services]
    CLI[CLI and notebooks] --> Services
    Services --> Core[Mesh and numerical core]
    Services --> IO[Explicit file adapters]
    Services --> Jobs[Future background jobs and project records]
    IO --> Legacy[MATLAB mesh snapshots]
    IO --> Native[Versioned Python artifacts]
    Jobs --> Core
```

## Implemented modules

| Module | Responsibility | Dependency boundary |
| --- | --- | --- |
| `mesh` | Validated mesh snapshots and topology checks | NumPy; SciPy only for topology |
| `geometry` | Pure per-mesh numerical operations | NumPy/SciPy, no files or viewer |
| `alignment` | Landmark-defined frames and oriented mesh transforms | NumPy; explicit scale, reflection policy, and corner correspondence |
| `curvature` | Normal estimation, mixed areas, transported curvature tensors, and principal values | NumPy/SciPy topology; BSD-derived numerical estimator, no chart required |
| `quasiconformal` | Per-face Beltrami coefficients and dilatation of supplied maps | NumPy; explicit vertex correspondence and planar orientation checks |
| `curves` | Path length, polygonal writhe, Gauss quadrature, and segment-contact checks | NumPy; explicit normalization and open/closed conventions |
| `grid` | Rectilinear connectivity and explicit seam mapping | No implicit parameterization |
| `io` | MATLAB and NPZ adapters | Conversion happens at this boundary |
| `imaging` | Trilinear sampling and supplied-chart pullbacks | No segmentation or temporal tracking |
| `surface` | Barycentric query locations and scalar/vector interpolation | Retains original triangles and holes; no surface fitting |
| `analysis` | JSON-compatible mesh inspection | Same callable for CLI and GUI |
| `cli` | Input/output and user-facing failures | Delegates calculations to core |
| `experiment` | `TubULAR` state, time selection, explicit sources, cache invalidation | Calls core operations; no viewer or implicit output writes |
| `_coordinates` | MATLAB-named image/APDV support helpers | Explicit coordinate conventions and legacy arithmetic options |
| `_method_catalog` | Original method inventory and actionable pending-method errors | Packaged metadata; no runtime MATLAB dependency |

`Mesh` copies its arrays and marks them read-only to prevent ordinary accidental
mutation. This is an API convention, not a security boundary. New geometry is
represented by a new mesh. Computational functions return values and do not
save files or change global state. Pullback sampling accepts progress and
cancellation callbacks without importing a UI library.

## Implemented experiment object

`TubULAR` retains useful MATLAB method names over a Python-native experiment.
It accepts explicit acquisition timepoints and options, loads
explicit mesh/image sources, and owns caches for the selected timepoint.
Changing time clears all current caches, including aligned and Ricci meshes.
Each instance owns its state. `getCurrent…` reuses a loaded mesh;
`loadCurrent…` reloads it. No constructor, getter, or measurement writes files.

The object adapts 60 original method names and explicitly marks 140 as pending.
ImSAnE compatibility is not a design requirement: version 0.5 removes the
simulated `xp.fileMeta`/`xp.expMeta` object and the `xp` constructor argument.
The in-memory object is separate from the future durable project manifest.
See [class API](class_api.md) for the supported scope and changed conventions.
GUI controllers can use it for selection and loading; background workers should
receive immutable mesh/data inputs rather than share a mutable selected time.
Alignment supports previewing a full series before explicitly registering it;
registration occurs only after all mesh transforms succeed. Landmark inputs
can later come from a GUI without coupling the calculation to that viewer.
Annular parameterization likewise takes an explicit seam and returns a complete
cut/embedding result before optional registration. Its numerical core is
independent of time selection and file output. Registration stores the mapped
mesh; callers retain the result's topology correspondence and solver diagnostics.
Periodic UV resampling returns a similarly independent result with source
triangle/vertex sampling weights and seam metadata. The same weights transfer
geometry and periodic vertex fields; a future GUI can inspect them without
owning the numerical calculation or mutating the selected source mesh.

## Proposed project model (not implemented)

An `Experiment` should be a versioned project manifest, not a large mutable
equivalent of the MATLAB handle class. It should reference these records:

| Record | Required information |
| --- | --- |
| Image series | Source URI, checksum, explicit `t,c,z,y,x` axis labels, dtype, channel IDs, voxel spacing, origin, physical unit |
| Timepoints | Explicit timestamp values and time unit, including nonuniform acquisition intervals |
| Coordinate frame | Raw/aligned/material frame ID, axis names, handedness, complete transform, units |
| Mesh artifact | Vertex/face arrays, correspondence ID, frame ID, topology report, checksum |
| Chart artifact | Mesh ID, per-vertex UV, units of each chart axis, periodicity, cut-to-glued index map |
| Field artifact | Scalar/vector/tensor data, sample association (face/vertex/pixel), basis, covariant/contravariant type, unit |
| Operation record | Input IDs/hashes, algorithm and version, parameters, output IDs, warnings, environment, timestamps |

Do not subtract time-adjacent vertex arrays unless correspondence is explicitly
established. Do not combine vectors from raw, aligned, and material frames
without transforming both their coordinates and bases. Do not interpret a
frame-index difference as elapsed time. APDV mirroring must be recorded because
it changes orientation-sensitive quantities.

For large volumes, evaluate chunked OME-Zarr adapters separately. Small mesh
snapshots currently use NPZ; this is not a claim that NPZ is a suitable whole
experiment format. Never serialize application objects with pickle as the
project storage format.

## GUI proposal

**Preferred first prototype: a napari plugin**, keeping all computation in the
headless package. Napari supplies image, point, vector, and surface layers and
a plugin mechanism; this is a design choice based on those capabilities, not
a validated integration in this repository. See its
[surface layer documentation](https://napari.org/stable/howtos/layers/surface.html)
and [plugin documentation](https://napari.org/stable/plugins/index.html).

The first GUI milestone should let a researcher:

1. Open a project and confirm image axes, voxel spacing, and time units.
2. Inspect the image volume and mesh together, with boundary/nonmanifold defects
   highlighted and an orientation preview.
3. Choose anterior/posterior/dorsal landmarks, inspect a centerline, and confirm
   cuts using reversible edits.
4. Inspect the unwrapped image beside the 3D surface, with linked selections
   and visible seam locations.
5. Run a validated measurement, inspect its quality report, and export both
   the result and the exact parameters used.

Implement the first two views before offering unvalidated mapping/strain
buttons. Keep unfinished operations out of the active workflow. An initial
surface-layer adapter must explicitly convert core XYZ coordinates to the
viewer's image-axis order; test a labeled asymmetric volume to catch mirrored
or transposed overlays. Per-face values cannot silently become per-vertex
colors: document any averaging used for display.

The GUI should submit immutable requests to background workers. Progress,
cancellation, errors, and result IDs return as messages. Qt objects remain on
the main thread. Results become visible only after a completed operation;
cancellation never promotes partial files to finished artifacts. Preview
parameters and committed analysis parameters must remain distinguishable.

A custom Qt/PyVista application remains an option if linked 2D/3D selection or
large-mesh performance proves inadequate in a napari prototype. That decision
should follow a real-dataset benchmark, not precede it.

## Reproducibility and execution (future work)

Each stage should declare inputs and outputs explicitly and operate on one
timepoint or a specified interval. Cache keys must include content hashes,
parameters, coordinate conventions, and algorithm version, not just a filename.
Write into a temporary artifact directory and atomically promote the completed
record; journal failures/cancellations. Handle migrations explicitly when a
manifest schema changes. The current NPZ writer only prevents overwrite; it
does not yet implement this transactional experiment store.

Parallelize independent timepoints only after single-process parity is stable.
Establish memory/runtime baselines on one representative midgut or heart
dataset before choosing process counts, array chunking, or a GPU backend.
No cloud service is required for the proposed desktop workflow.
