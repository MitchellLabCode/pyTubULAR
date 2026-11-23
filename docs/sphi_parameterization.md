# Segmented mesh to surface (s,phi)

Version 0.9 adds a connected, headless path from a **triangulated** closed
segmented surface to an `(s,phi)` mesh. Endcap vertex sets and reference seam
landmarks are supplied explicitly. Voxel-label isosurface extraction, automatic
landmarks, mesh repair, texture-based phase fitting and GUI selection are not
included. Version 0.10 adds [geometric alignment to an explicit reference](temporal_phase.md).
Run `python examples/segmented_mesh_to_sphi.py` for a complete synthetic example.

```python
from tubular import TubULAR

tubi = TubULAR(time_points=[0], opts={"nU":101, "nV":201})
tubi.register_mesh(0, segmented_mesh, kind="rawMesh")  # Mesh; XYZ already physical
tubi.setTime(0)
opened = tubi.sliceMeshEndcaps(anterior_vertex_ids, posterior_vertex_ids, register=True)
cut = tubi.generateCurrentCutMesh(seam_endpoints=[start_id, end_id], register=True)
sp = tubi.generateCurrentSPCutMesh(cut.path_pairs, register=True)
coordinates = sp.sphi             # same array as sp.mesh.uv
```

All IDs are zero-based. Cap IDs refer to the input segmentation; seam endpoints
refer to `opened.mesh`. Convert a retained source ID with `opened.source_to_open`
(`-1` means removed). The start endpoint fixes the s=0 end. The two landmark
positions set the reference seam; no anatomical dorsal direction is inferred.
For an already-open annulus, register `cylinderMeshClean` directly and start at
`generateCurrentCutMesh`. An explicit `seam_path` still works; supply exactly one
of a path or endpoint pair.

Cap removal retains faces that touch neither selected cap and compacts indices.
It requires a connected, consistently oriented, closed genus-zero input and a
connected annulus with two distinct output boundaries. It returns source vertex
and face IDs, reverse vertex IDs and ordered anterior/posterior boundary loops.
It does not cut intersected triangles. Shortest-seam search uses Euclidean edge
lengths and excludes every boundary vertex except the specified endpoints. It is
a graph path, not a continuous geodesic; ties can change under vertex reindexing.
Neither operation discards disconnected components to hide a bad selection.

## Coordinate definition and approximation

The input cut chart can have a curved seam. The s-phi operation first samples
equal U/V spacing, computes circumferentially averaged longitudinal chord
lengths, and inverts the cumulative length to choose nonuniform U samples.
At those U values it samples hoops and resamples each hoop by 3D polygonal
arclength. Mean corresponding-point separations between these uniform hoops
define `ring_s`. Centers and radii use these hoops, excluding their duplicate
endpoint. These are surface path lengths, not centerline arclengths; the separate
`centerline_s` measures the hoop-center polyline.

The first grid's original geometry is then relabeled
`(ring_s(U), V - phi_offsets(U))` and sampled onto uniformly spaced s and phi.
The output has U-fastest vertex order (`j*n_u+i`) and an exact duplicate seam.
Original winding is retained. `s` has the units of the supplied XYZ coordinates;
`phi` is **turns**, with period 1, matching MATLAB. `phi_radians` multiplies by
2 pi. Phi is not generally a polar angle or proportional to hoop arclength.
No APDV transform, reflection or length-unit conversion is implicit.

Offsets default to zero, fixing phase to the input chart. Supplied `phi_offsets`
must contain one finite value per intermediate U ring, in turns, evaluated at
the returned `u_samples`. Version 0.10 alternatively accepts
`reference=previous_result` for geometric phase fitting; see
[temporal_phase.md](temporal_phase.md) for axial correspondence
and ambiguity handling. Texture registration and iterative phase smoothing
remain pending. Neither path establishes material tracking between frames.

Both interpolation stages use original triangles of their respective inputs.
The second stage interpolates a newly connected grid, so its points need not
lie on one triangle of the original segmentation. Sampling density controls
this approximation; check resolution convergence and overlays on real data.
There is no 3D self-intersection test or automatic resolution selection.
Degenerate triangles, inverted chart triangles, invalid topology and undefined
hoop lengths are errors. No extrapolation or silent geometric repair occurs.

## Field correspondence

`sp.first_sampling` maps the source cut vertices to the intermediate grid;
`sp.final_sampling` maps intermediate vertices to the final grid. Each exposes
source vertex triples and barycentric weights. `sp.interpolate(cut_field)`
composes them. Periodic fields must agree exactly on paired source seam vertices.
For a scalar field originally attached to the segmented mesh:

```python
cut_field = source_field[opened.vertex_sources][cut.cut_to_uncut]
final_field = sp.interpolate(cut_field)
```

Interpolation transfers scalar or vector components, not tensor bases. Keep the
result objects: class registration stores their meshes, not the correspondence
or diagnostics. All methods finish validation before replacing a registered
mesh. Grid overrides do not change `tubi.nU/nV`. Core calculations write no files.

## Reference scope and deliberate changes

The source is TubULAR `9bd49db`, particularly `generateCurrentSPCutMesh`,
`sliceMeshEndcaps`, `equidistantSampling1D`, `resampleCurvReplaceNaNs`,
`srFromDVCurves`, `interpolate2Dpts_3Dmesh` and the annular tiler.
`scripts/generate_sphi_reference.m` executes unchanged helpers on three cut
charts with both averaging conventions (six cases), including nonplanar
geometry, varying phase offsets and reversed winding. The stored final XYZ
comparison differs by at most approximately `1.9e-15` on the tested platform.
Custom endcap removal also has an original-helper reference. Source, fixture,
input fixture and generator hashes are in `tests/data/sphi_provenance.json`.

This is **stage-level validation**, not execution of the full MATLAB class:

- `ringpathsGridSampling`'s optional hoop-length output indexes the wrong axis
  and assumes equal nU/nV. The harness requests only its first output and uses
  the unchanged interpolator for XYZ. Python computes correct hoop-axis lengths.
- The class's `radii_from_avgpts` block indexes a flattened XYZ array as a 3D
  grid. Python exposes `hoop_radii` computed on the correctly shaped uniform
  hoops, validated against `srFromDVCurves`.
- Python's default `seam_weight="unique"` counts the periodic endpoint once
  when averaging longitudinal steps. `seam_weight="matlab"` retains the original
  duplicate-endpoint weighting for comparison. This distinction measurably
  changes s on asymmetric geometry.
- Exact endpoint interpolation replaces epsilon boundary nudges. Original
  triangle interpolation and winding are retained. Centerline cropping,
  1000-point resampled centerlines, automatic aspect optimization, normals,
  temporal phase fitting, image matching, plots and disk caches are omitted from
  this 0.9 reference comparison. The 0.10 geometric phase fitting is validated
  separately in [temporal_phase.md](temporal_phase.md).
- Cap compaction uses vertex IDs rather than coordinate equality. Coincident
  but distinct vertices are never merged by accident.

Independent tests cover analytic cylinder length, a known quarter-turn phase
shift, rigid motion, physical scaling, winding, field transfer, exact closure,
shortest-path length, invalid inputs and registration failure. Biological-data
equivalence is still an outstanding scientific milestone.
