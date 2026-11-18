# Landmark frames and mesh alignment (0.5)

This release adds the numerical stages of `computeAPDVCoords` and
`alignMeshesAPDV`. Inputs are explicit XYZ landmarks and registered meshes.
These operations do not require an image volume or an ImSAnE experiment.
Landmark detection, endpoint displacement, temporal smoothing, and rendering
are separate, unimplemented stages of the original alignment workflow.

## Define a frame

```python
from tubular import TubULAR

tubi = TubULAR(time_points=[0, 3], opts={"resolution": 0.5, "spaceUnits": "um"})
anterior = [2, -3, 4]
posterior = [8, 1, 7]
dorsal = [4, 2, 10]
a, p, d, rotation, translation = tubi.computeAPDVCoords(anterior, posterior, dorsal)
```

Landmarks must each be a finite length-three real array in the **same raw XYZ
coordinates as the mesh**. The chosen origin maps to zero. The positive X axis
points from origin to posterior; positive Z points toward the component of
the dorsal vector perpendicular to X. Y completes a right-handed frame.

`origin=None` defaults to the anterior point. Supply `origin=[x, y, z]` to
use an explicitly selected start point instead. This changes both the AP axis
and the translation; it does not move an endpoint inside the mesh. The first
three returned arrays are copies of the supplied landmarks, before alignment.

The position transform remains
`(xyz @ rotation.T + translation) * resolution`, followed by optional reflection
of Y. `resolution` and `flipy` can be keyword arguments to `computeAPDVCoords`;
omitting them retains the object's configured values. Validation completes
before replacing the frame. The operation preserves time selection and mesh
caches; existing aligned meshes are snapshots, not automatically recomputed.

Scale is explicit. With landmarks and vertices already in physical units, use
`resolution=1`. With isotropic voxel coordinates, supply the physical length
per voxel. For anisotropic voxel coordinates, first convert **both** vertices
and landmarks to physical XYZ using all three spacings, then use scale one.
No metadata object supplies a hidden resampling or unit conversion.

For numerical use outside an experiment:

```python
from tubular.alignment import landmark_frame
rotation, translation = landmark_frame(anterior, posterior, dorsal)
```

Coincident origin/posterior points and a dorsal point on the AP axis are
rejected. Dorsal vectors with an angle to that axis whose absolute sine is
at most `1e-12` are rejected as numerically ambiguous. The frame uses normalized
cross products and handles AP parallel or antiparallel to any coordinate axis.
The original Rodrigues expression divides by zero for AP exactly along -X;
Python returns the uniquely specified proper frame for noncollinear landmarks.

## Align a series

Register each raw mesh using `tubi.register_mesh(timepoint, mesh_or_source)`.
Then preview or register the aligned results:

```python
preview = tubi.alignMeshesAPDV()
result = tubi.alignMeshesAPDV(time_points=[0, 3], register=True)
tubi.setTime(3)
mesh = tubi.getCurrentAlignedMesh()
```

`alignMeshesAPDV(*, time_points=None, register=False, preserve_orientation=True)`
reads the `rawMesh` stage and applies one configured frame to the requested
acquisition times. All times must exist and be strictly increasing. Raw unit
labels must agree across the series; output meshes carry `tubi.spaceUnits`.
The label alone does not determine or validate the supplied scale conversion.

The result is a Python dictionary, replacing MATLAB's six-output workflow:

| Key | Contents |
| --- | --- |
| `meshes` | Tuple of aligned, validated `Mesh` snapshots in timepoint order |
| `timepoints`, `times`, `time_unit` | Acquisition coordinates and physical elapsed times |
| `rot`, `trans`, `resolution`, `flipy` | Exact transform used for this batch |
| `preserve_orientation`, `face_corner_order` | Reflection policy and zero-based per-face corner mapping |
| `source_length_unit`, `length_unit` | Input and output coordinate labels |

Preview leaves all mesh sources, selection, and current mesh caches unchanged.
With `register=True`, sources for the requested `alignedMesh` timepoints are
replaced only after the whole batch succeeds. A selected aligned cache is
invalidated; unrelated caches and current time are preserved. File-backed
sources are read again for each batch. No files are written.

Changing the APDV frame does **not** transform an already aligned snapshot a
second time. To update alignment, rerun this operation from the raw sources.
For bounds, use the existing `measureXYZLims` operation. For permanent storage,
save the returned meshes explicitly through the I/O module and retain the
transform metadata with your analysis record.

## Reflection and correspondence

A reflection reverses signed orientation if triangle corner order stays fixed.
By default, Python swaps face corners 1 and 2 whenever `flipy=True`. Thus an
outward mesh stays outward and an inward mesh stays inward; the algorithm does
not infer which convention the input uses. Normals are computed from the
resulting geometry instead of importing MATLAB's separately flipped normal
arrays. Closed-mesh signed volume retains its sign and scales by resolution
cubed. Area scales by resolution squared.

Vertex and face row IDs stay fixed. UV values stay attached to their original
vertices. `face_corner_order` is `[0, 2, 1]` for a reflection with orientation
preservation, and `[0, 1, 2]` otherwise. Apply it to the corner axis of any
separately stored per-face-corner field. Vertex and scalar face fields require
no index permutation; vectors and tensors require the appropriate basis
transform. No arbitrary nearby vertices or seam copies are merged.

Use `preserve_orientation=False` to keep the original face arrays exactly, as
MATLAB does. With reflection this changes signed volume and the winding-derived
normal orientation. Both choices return the same transformed vertex positions.

The pure helper is
`alignment.align_mesh(mesh, rotation, translation, resolution=1, flipy=False,
preserve_orientation=True, length_unit=None)`. Its output unit defaults to the
input label; specify the new label when the explicit scale converts units.

## Reference comparisons and limits

`scripts/generate_alignment_reference.m` calls the unchanged original
`computeAPDVCoords` on four synthetic landmark triples using temporary input
files, including coordinate-axis and dorsal-antiparallel cases. It calls the
original `xyz2APDV` for vertex transformations with and without reflection;
that is the same arithmetic used inside `alignMeshesAPDV`.
It does **not** run the original full mesh-alignment, endpoint, or plotting
pipeline as an equivalence test. The fixture generator's unavoidable MATLAB
frame-preview figures are created only in temporary storage and removed.

The committed fixture and its generator/source hashes are recorded in
`tests/data/alignment_provenance.json`. The original class declaration exposes
only the three landmark outputs, although the external method declares five.
Its rotation and translation files use five significant digits. The comparison
therefore rounds Python's frame to that same precision (with `atol=1e-14` for
near-zero arithmetic residue), instead of claiming full-precision frame parity.
Vertex transformations are tested separately with full-precision proper input
rotations, using `atol=1e-13, rtol=1e-12`. Face mappings are exact.
Independent tests check orthonormality and handedness, landmark axis placement,
rotation/translation/scale covariance, normal/area/volume behavior, chart
interpolation, copy isolation, and transaction-like series registration.

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_alignment_reference('/path/to/tubular', ...
    '/path/to/pyTubULAR/tests/data/matlab_alignment_reference.mat');
```

These checks do not establish correct anatomical landmark choices on real
data, repair poor mesh geometry, or validate the complete MATLAB workflow.
The runnable [alignment example](../examples/landmark_alignment.py) uses only
native Python arrays and meshes.
