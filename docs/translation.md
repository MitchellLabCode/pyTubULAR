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

The 0.10 extension adds reference-based geometric phase alignment through
`generateCurrentSPCutMesh`, with separate `fit_hoop_phase` and `fit_sphi_phase`
diagnostics. The non-resampled `phiOffsetsFromPrevMesh` objective has an
unchanged MATLAB reference; global search, unique seam weighting, axial
correspondence and unwrapping are explicit Python choices. See
[temporal_phase.md](temporal_phase.md).

The 0.9 operations adapt custom `sliceMeshEndcaps` vertex removal and the
geometric stages of `generateCurrentSPCutMesh`, and add shortest edge seams.
The single-pass `measurePIV2d` core translates pinned PIVlab numerical kernels.
That 0.9 slice did not include temporal phase fitting or multipass PIV. See
[sphi_parameterization.md](sphi_parameterization.md) and [piv.md](piv.md) for
reference scope, deliberate changes, package/license audit and remaining work.

The 0.8 resampling operation adapts `generateCurrentUVCutMesh` and
`tileAnnularCutMesh`. Default sampling uses the existing source triangles,
matching `interpolate2Dpts_3Dmesh`, and preserves source winding. Explicit
`matlab_scattered` interpolation and `preserve_orientation=False` retain the
original algorithm/face-order choices within documented Delaunay tie limits.
Both return vertex sampling weights; neither extrapolates or writes files.
See [uv_grid_resampling.md](uv_grid_resampling.md) for measured differences.

The 0.7 parameterization operation adapts `generateCurrentCutMesh` with a
caller-supplied seam, preserving the `cylinderCutMesh` custom-path correspondence
and `flattenAnnulus` Dirichlet boundary constraints. Local vertex-fan cutting
replaces the global path-adjacent flood, and output checks reject invalid charts
instead of attempting remeshing. No automatic seam search or aspect optimization
is included. See [annular_parameterization.md](annular_parameterization.md).

The 0.6 surface operations adapt `measureCurvatures` through the BSD-licensed
`GetCurvatures` dependency and its normal/area/tensor helpers. See
[surface_curvature.md](surface_curvature.md) for the mean-curvature sign and the
one-line temporary reference-harness correction required by current MATLAB.
`measureBeltramiCoefficient` and `measureBeltramiCoefficientPullbackToPullback`
adapt the `bc_metric.m` measurement stage on explicitly corresponding vertices.
Python returns raw coefficients and the reference coefficient separately and
rejects reversed planar maps. See [beltrami.md](beltrami.md).

The 0.5 alignment operations adapt `computeAPDVCoords` to supplied XYZ landmarks
and `alignMeshesAPDV` to mesh snapshots. Direct orthonormal-frame construction
avoids the original -X Rodrigues singularity. Reflection preserves mesh
orientation by default, with an explicit option to retain original face order.
The `xp` metadata adapter is removed; ImSAnE compatibility is not required.
See [landmark_alignment.md](landmark_alignment.md) for numerical and API scope.

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
| `utility/axisymmetric_pullbacks/tileAnnularCutMesh.m` | `resampling.tile_annular_mesh` | Same sewn vertex/face order; adds explicit source maps and seam/chart validation |
| `@TubULAR/generateCurrentUVCutMesh.m` | `resampling.resample_annulus`, `TubULAR.generateCurrentUVCutMesh` | Original-triangle default; explicit Delaunay comparison mode, exact seam closure, supplied normals, no extrapolation; optional legacy face order |
| `utility/axisymmetric_pullbacks/cylinderCutMesh.m` (custom path) | `parameterization.cut_annulus` | Local fan splitting, unchanged face/corner order, explicit vertex maps; errors rather than repairs |
| `utility/axisymmetric_pullbacks/flattenAnnulus.m` (Dirichlet) | `parameterization.flatten_annulus` | Same cotangent constraints and 0.01 negative-weight clamp; adds chart validation and explicit lateral scaling |
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
