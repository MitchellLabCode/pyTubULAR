# Surface curvature (0.6)

`TubULAR.measureCurvatures` now estimates curvature on supplied meshes using
the estimator called by the original MATLAB method. It does not require a
UV chart or an ImSAnE experiment.

```python
series = tubi.measureCurvatures(mesh_kind="spcutMeshSmRSC", time_points=[0, 3])
first = series["curvatures"][0]
mean = first.mean
gaussian = first.gaussian
principal_values = first.principal_curvatures
```

Both arguments are keyword-only. The default stage is `spcutMeshSmRSC`, and
the default times are all acquired timepoints. The dictionary includes
`timepoints`, physical `times`, `time_unit`, `length_unit`, `mesh_kind`, and a
tuple of `CurvatureResult` objects. Meshes may have different vertex counts;
there is no assumption of temporal vertex correspondence. Units must agree.
The operation preserves selection and caches and writes no files.

Use the same pure computation on a single mesh:

```python
from tubular.curvature import estimate_curvature
result = estimate_curvature(mesh)
```

## Fields and conventions

| Field | Shape | Meaning |
| --- | --- | --- |
| `principal_curvatures` | N × 2 | Eigenvalues of the estimated normal derivative, ordered by decreasing absolute magnitude |
| `principal_directions` | N × 2 × 3 | Unit tangent eigenvectors in XYZ; signs are arbitrary |
| `mean` | N | TubULAR's convention: `-(k1 + k2)/2` |
| `gaussian` | N | `k1*k2` |
| `normals` | N × 3 | Normalized, Max-weighted vertex normals following face winding |
| `shape_operator` | N × 3 × 3 | Normal derivative in the ambient basis, zero on the estimated normal direction |
| `vertex_areas` | N | Sum of incident mixed corner areas |
| `corner_areas` | F × 3 | Area assigned to each triangle corner, in face/corner order |
| `boundary_vertices` | N | Boolean mask of vertices incident to boundary edges |
| `length_unit` | string | Label inherited from the supplied mesh |

All arrays are read-only snapshots. Vertex and face ordering are unchanged.
Mean, principal values, and shape operator have inverse-length units;
Gaussian curvature has inverse-square-length units; areas have squared units.
For an outward sphere of radius R, the principal values are positive 1/R,
TubULAR's mean is **negative** -1/R, and Gaussian curvature is positive 1/R².
Reversing winding reverses normals, principal values, shape operator, and mean,
while preserving Gaussian curvature. The code never infers outwardness.

Principal directions are axes, not signed arrows. Their signs may differ from
MATLAB; at repeated eigenvalues, including an ideal sphere or flat region,
individual directions have no unique meaning. Use the shape operator to
compare tensors without depending on that arbitrary choice of basis.

The method uses Max-weighted normals and mixed areas from the MATLAB helper,
least-squares face tensors, and transport into vertex tangent frames. Batched
QR solves replace MATLAB's rectangular backslash. Cross-product areas replace
Heron's formula for better numerical conditioning. Input coordinates are
temporarily normalized for calculation, then result units are restored.
This is distinct from `geometry.fundamental_forms`, which uses angle-averaged,
unnormalized normals and returns chart-based face tensors.

## Boundaries and failure handling

The estimator requires a consistently oriented manifold. It rejects canceled
vertex normals, singular tangent frames or face fits, and nearly antiparallel
normal transport. It does not repair faces, smooth coordinates, or determine
whether a surface intersects itself.

Open boundaries are allowed and flagged. Their estimates use truncated
neighborhoods and can be biased. A duplicated seam is also a boundary until
explicitly glued; this routine does not infer periodicity or endcaps. For a
known rectilinear cylinder, use `grid.glue_rectilinear_seam` and its returned
correspondence map before computing periodic circumferential curvature.

The full MATLAB plotting/saving pipeline, curvature derivatives, mesh
smoothing, and real-tissue accuracy benchmarks remain unimplemented.

## Reference evidence and attribution

The algorithm follows Rusinkiewicz's
[Estimating Curvatures and Their Derivatives on Triangle Meshes](https://pixl.cs.princeton.edu/gfx/pubs/Rusinkiewicz_2004_ECA/index.php).
This port adapts Itzik Ben Shabat's BSD-licensed MATLAB implementation with
Noah Mitchell's modifications. The complete notice is preserved in
[`licenses/curvature-BSD.txt`](../licenses/curvature-BSD.txt) and included in
both wheel and source distributions.

The dependency is the existing `MitchellLabCode/matlab` checkout at
`1877f4bf1c3fcedc34839c1c20f311ee7a826820`, under `geometry/curvature`.
The MATLAB validation harness makes **one recorded correction in a temporary
copy** of `getPrincipalCurvatures.m`: the active loop bound
`1:size(FV.vertices)` becomes `1:size(FV.vertices,1)`. Current MATLAB rejects
the original vector-valued colon bound. No numerical expression or original
repository file is changed.

`tests/data/curvature_provenance.json` records the source, dependency, generator,
temporary corrected helper, and fixture hashes. The reference includes an
open saddle patch and a nonuniform ellipsoid with both windings. Normals,
corner/vertex areas, principal values, H, K, ambient tensors, and direction
projectors agree within `atol=1e-12, rtol=1e-10`. Direct eigenvector signs are
not compared. Independent tests cover spheres, plane triangle types, cylinder
interiors, scale/rotation/winding transformations, tensor eigenvectors, and
failure cases. These tests establish the documented estimator, not accuracy
on every noisy or poorly sampled biological surface.

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_curvature_reference('/path/to/matlab/geometry/curvature', ...
    '/path/to/pyTubULAR/tests/data/matlab_curvature_reference.mat');
```
