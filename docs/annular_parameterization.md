# Annular surface parameterization (0.7)

`TubULAR.generateCurrentCutMesh` now constructs a UV chart from a registered
`cylinderMeshClean` and an explicit zero-based seam path. This is a numerical
slice of the original method, with no file workflow or automatic seam selection.
The source need not already have a chart. The method returns geometry, vertex
correspondence, seam pairs and solver diagnostics. See the runnable
[example](../examples/annular_parameterization.py).

```python
tubi.setTime(timepoint)
result = tubi.generateCurrentCutMesh(
    seam_vertex_ids, aspect_ratio=1.0, negative_weights="clamp", register=True,
)
chart = tubi.getCurrentCutMesh()  # result.mesh
cut_vertex_field = source_vertex_field[result.cut_to_uncut]
```

With the default `register=False`, this returns a preview without replacing
the stored cut mesh. Registration occurs only after all validation succeeds.
Keep the result: registration stores its `Mesh`, while seam and correspondence
metadata remain on the result. Existing mesh NPZ/MAT storage does not yet save
the complete parameterization result. No APDV transformation, `flipy`, normal
shift or inferred aspect ratio is applied. `aspect_ratio` explicitly scales U.

## Geometry and constraints

The pure operations in `tubular.parameterization` are:

- `cut_annulus(mesh, seam_path) -> SeamCut`: requires one connected, consistently
  oriented manifold annulus with two boundary loops. The simple edge path must
  connect different loops without touching the boundary in its interior.
- `flatten_annulus(cut_mesh, path_pairs, ...) -> AnnularEmbedding`: requires a
  disk with complete paired seam chains and exactly coincident 3D seam copies.
  Existing UV values are ignored. Imported approximate seam copies must be
  reconciled explicitly before use.

Cutting appends a copy of every seam vertex, keeping all original face rows,
face corner order, physical coordinates and units. `cut_to_uncut[cut_mesh.f]`
reproduces the source faces exactly. `uncut_to_cut` supplies every original
vertex's one or two copies. Face fields retain their original indexing; vertex
fields transfer through `cut_to_uncut`. Input UV coordinates are discarded
when cutting. Cutting uses local incident face fans, avoiding the original
global flood through faces adjacent to a path that may run close to itself.

The embedding follows Dillon Cislo's `flattenAnnulus.m` Dirichlet branch in
TubULAR, inspired by [Orbifold Tutte Embeddings](https://noamaig.github.io/euclidean_orbifolds/).
It solves the cotangent Laplacian with linear equality constraints:

- The two true boundary rings lie at U=0 and U=1 before lateral scaling. Their
  V values are fixed by normalized 3D boundary edge arclength.
- Corresponding seam vertices have identical U and differ by one in V:
  `uv[path_pairs[:,0]] - uv[path_pairs[:,1]] == [0,1]`.
- Interior seam positions are otherwise free. The seam curves need not lie
  on V=0 and V=1; the cut polygon need not be a rectangle. Treat this as a
  periodic fundamental domain. Explicit periodic tiling and regular-grid
  resampling are now available in [the 0.8 UV-grid operation](uv_grid_resampling.md).

The first seam column retains the left side of the directed path; the second
contains appended right-side copies. The MATLAB boundary convention normally
gives **negative UV signed areas** relative to preserved face winding. This is
reported as `embedding.orientation=-1`, and is not a foldover. Python does not
silently reverse faces or reflect the chart. All triangles must have the same
nonzero signed-area sign; existing UV-to-3D queries accept either orientation.

The default negative-weight policy reproduces MATLAB: negative **assembled
edge** cotangent weights become 0.01, then diagonal entries are recomputed.
`embedding.clamped_edges` reports how many undirected edges changed. This
changes the energy and is not a promise of an exactly conformal chart.
`negative_weights="error"` instead rejects those meshes. As in MATLAB, the
sign decision is strict; roundoff near an exactly zero cotangent may change
whether an edge is clamped. No Delaunay remeshing is performed. Cotangents use
locally scaled edge vectors for numerical stability; no third-party geometry
implementation is bundled or added as a dependency.

## Validation and limits

The sparse constrained solve must have finite output, maximum absolute system
residual at most 1e-6 (the MATLAB threshold), and constraint residual at most
1e-10. These diagnostics refer to the unit-width solve. Degenerate triangles,
mixed UV winding, boundary crossings, nonadjacent boundary contacts, and
boundary backtracking are rejected before a result is registered. Boundary
intersection checks are quadratic in boundary size in the worst case, with
bounding-box filtering. Very large biological meshes still need performance
benchmarks; there is no blanket accuracy or injectivity guarantee for arbitrary
ill-conditioned inputs. Physical 3D self-intersections are not detected here.

The unchanged MATLAB `cylinderCutMesh` custom-path branch and `flattenAnnulus`
Dirichlet branch were run in MATLAB R2025b on four synthetic cases: a straight
cylinder, a curved tapered tube, a diagonal seam, and reversed winding/path.
Python matches cut vertices, face connectivity and correspondence exactly and
UV values to a maximum observed absolute difference below 9e-16. The test uses
rtol=1e-11 and atol=1e-12. The fixture generator, hashes and pinned source revision
are in `scripts/generate_parameterization_reference.m` and
`tests/data/parameterization_provenance.json`. No source patches were needed.

Independent checks cover analytic cylinder coordinates, a zero-Beltrami chart
at the polygonal cylinder's correct aspect ratio, seam translation, topology,
field transfer, rigid motion/scaling, vertex/face renumbering and invalid inputs.
A twice-wound planar fan tests that local orientation alone cannot bypass the
boundary-overlap rejection. Class tests cover preview, successful registration,
failed registration and time selection without file writes.

Automatic seam search/tracking, aspect-ratio minimization, MVC, Ricci flow,
`generateCurrentSPCutMesh`, normal displacement and
temporal phase stabilization remain pending. The full original
`generateCurrentCutMesh` file workflow is not a parity target of this release.
Representative biological datasets and overlays are still needed before
declaring the topology/parameterization milestones scientifically complete.
