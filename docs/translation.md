# Translation contract

Reference: [npmitchell/tubular at 9bd49db](https://github.com/npmitchell/tubular/tree/9bd49dbffd38b199c32552b8cc089f611f584679).
This is a deliberate translation of selected routines, not a mechanical rewrite
of every MATLAB file. Source defects and ambiguous conventions are recorded
rather than carried forward silently.

The Python `TubULAR` class is now available. Its method-by-method scope and
intentional differences are described in [class_api.md](class_api.md) and
[method_coverage.md](method_coverage.md). The inventory distinguishes adapted
methods from pending ports; it is not a claim of complete MATLAB compatibility.

## Implemented source map

The 0.4 curve operations port `utility/curve_functions/writheLevitt.m` and
`writheGaussIntegral.m` and expose their supported numerical workflow through
`measureWrithe` and `measureLength`. The Gauss helper's duplicated numerator
and extra normalization are available through an explicit legacy flag;
the default uses the standard Gauss factor. Curve trimming, time derivatives,
and stored length origins have documented corrections. See
[centerline_writhe.md](centerline_writhe.md) for exact conventions and limits.

The 0.3 class additions map `uv2APDV` to the original
`utility/axisymmetric_pullbacks/interpolate2Dpts_3Dmesh.m` barycentric operation;
`surface.locate_chart` also exposes this operation for arbitrary vertex/face
fields. `interpolateOntoPullbackXY` retains the separate tiled rectilinear-grid
algorithm. Bounds and centerline methods preserve the numerical transforms
while keeping file output outside the calculation. See the class guide for
supported options, outside-point sentinels, and intentional cache differences.

All paths below are relative to the original TubULAR repository.

| MATLAB source | Python API | Equivalence and deliberate differences |
| --- | --- | --- |
| `utility/mesh_handling/meshArea.m` | `geometry.surface_area` | Same triangle area; rejects invalid geometry rather than ignoring NaNs |
| `utility/mesh_handling/meshVolumeArea.m` | `geometry.signed_volume` | Same signed volume on closed oriented surfaces; stable shifted-origin tetrahedral sum; rejects open/nonmanifold/inconsistently oriented meshes |
| MATLAB `triangulation.faceNormal` | `geometry.face_normals` | Right-hand unit normal; does not infer outwardness |
| `utility/mesh_handling/faceNormals.m` | `geometry.face_normals(convention='legacy_face_normals')` | Original helper has the opposite sign to MATLAB's built-in |
| `utility/mesh_handling/meshAveragingOperators.m` | `geometry.averaging_operators` | Uniform V2F and angle-weighted F2V; stable atan2 corner angles |
| `utility/mesh_handling/coordinate_transformations/jacobian2Dto3DMesh.m` | `geometry.jacobian_2d_to_3d` | Equivalent per-face affine map via batched linear solves |
| `utility/mesh_handling/coordinate_transformations/jacobian3Dto2DMesh.m` | `geometry.jacobian_3d_to_2d` | Equivalent tangent inverse via pseudoinverse; not an inverse for normal motion |
| `utility/geometry/metric/inducedMetric.m` | `geometry.induced_metric` | Equivalent `J.T @ J` on each face |
| `utility/geometry/metric/constructFundamentalForms.m` | `geometry.fundamental_forms` | Same angle-averaged vertex normals, **without renormalization**; same symmetrized negative normal derivative |
| `utility/mesh_handling/meshDEC/resolveTangentNormalVector.m` | `geometry.resolve_tangent_normal` | Same projection, explicit sample-face indexing with corrected validation |
| `utility/mesh_handling/rectilinearMesh/defineFacesRectilinearGrid.m` | `grid.rectilinear_faces` | Exact vertex indexing and face ordering, zero-based; auxiliary MATLAB faceIDgrid not implemented |
| `utility/mesh_handling/rectilinearMesh/glueRectCylinderCutMeshSeam.m` | `grid.glue_rectilinear_seam` | Same index mapping, with seam-position validation; intentionally drops the now-discontinuous UV chart |

The averaging, Jacobian, induced metric, and fundamental-form routines are
derived from work by Dillon J. Cislo, with TubULAR contributions by Noah P.
Mitchell. `meshVolumeArea` credits K. Suresh and N.P. Mitchell. The source files
and their hashes are recorded in the reference provenance file.

## Conventions that must remain explicit

- Python mesh vertices are `(N,3)` XYZ; faces are `(F,3)` integer indices starting
  at zero. MATLAB imports subtract one only after validating integer-valued
  one-based indices. No transposed-array heuristics.
- A chart is `(N,2)` UV, not an image. Image arrays use `(z,y,x)`; a pullback image
  uses `(v,u)` rows/columns. Voxel `[0,0,0]` denotes its center at `origin_xyz`.
- A rectangular grid uses `index = j*n_u+i`; U varies fastest. This matches
  MATLAB `ndgrid` flattened in column-major order, not arbitrary NumPy reshaping.
  Face order is also preserved because downstream fields may depend on it.
- Fundamental forms are dense `(F,2,2)` arrays instead of MATLAB cell arrays.
  Batched matrix products use `@`/`einsum`; ordinary `*` is elementwise.
- `b = -sym(dX.T @ dn)`. Reversing face winding reverses `b`, signed normal
  velocity, and signed volume, but preserves area and metric.
- A seam consists of distinct vertices on a cut mesh. Gluing returns an explicit
  cut-to-glued map and never merges arbitrary nearby points. A glued cylinder is
  still open at its two ends, so its enclosed volume is undefined here.
- The fundamental-form routine uses the neighborhoods in the supplied mesh.
  Duplicated seam vertices have separate neighborhoods; periodic curvature
  estimation requires additional work and must not be inferred from this API.
- Vectors passed to `push_vectors` are contravariant chart components. Covectors
  and tensors need their own transformation rules; do not reuse that function.
- Length units label the existing coordinates. Area uses squared units and volume
  cubed units; no scaling happens on import/export. MATLAB export does not store
  the unit label, so it must be supplied again on reimport.

## New behavior with independent analytic tests

Topology validation, NPZ storage, and trilinear chart sampling are new services,
not claims of exact compatibility with every corresponding MATLAB workflow.
Pullbacks preserve the supplied mesh triangles, including holes. They do not
re-Delaunay-triangulate chart vertices, silently fill missing pixels, or wrap
periodic boundaries. Sampling is a single surface point per pixel, **not**
TexturePatch's normal-stack projection or the full `generateCurrentPullbacks`
workflow. Integer input images produce floating point samples.

## Findings to resolve before later ports

- `faceNormals.m` and `triangulation.faceNormal` use opposing signs. Every
  downstream curvature/normal-velocity calculation needs a sign audit.
- `constructFundamentalForms.m` calls its averaged vertex normals unit normals,
  but does not normalize them. Renormalization would change scientific outputs.
- The docstring descriptions of V2F/F2V in `meshAveragingOperators.m` are swapped;
  the actual matrix dimensions and calculations determine the Python semantics.
- `meshVolumeArea.m` uses a dimension test on `pts` while deciding the orientation
  of `tri`, and accepts open surfaces. Python requires canonical shapes and a
  closed mesh instead of reproducing those ambiguous cases.
- `measureMetricStrainRate.m` has a `QS` argument but references `tubi` in its
  initial options. Its time stepping and correspondence assumptions require
  review before a translation can be regarded as scientifically equivalent.
- `resolveTangentNormalVector.m` has optional-argument branches that can leave
  normals undefined or apply the wrong length assertion. Python handles the
  basic and explicitly indexed cases directly.
