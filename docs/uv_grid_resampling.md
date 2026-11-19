# Periodic UV-grid resampling (0.8)

`TubULAR.generateCurrentUVCutMesh` now turns an existing periodic cut chart into
a regular grid, preserving explicit sampling correspondence. This is the next
operation after `generateCurrentCutMesh`; it does not solve a new conformal map.
The [runnable example](../examples/uv_grid_resampling.py) goes from an uncharted
annulus through cutting, flattening, resampling and scalar-field transfer.

```python
cut = tubi.generateCurrentCutMesh(seam_vertex_ids, register=True)
grid = tubi.generateCurrentUVCutMesh(cut.path_pairs, register=True)
mesh = tubi.getCurrentUVCutMesh()  # grid.mesh
cut_field = original_vertex_field[cut.cut_to_uncut]
grid_field = grid.interpolate(cut_field)
```

The selected `cutMesh` is the input. Pass its explicit seam pairs; those pairs
are not inferred from coincident points or stored in a bare `Mesh`. Registration
defaults to false, and `uvcutMesh` is replaced only after every calculation and
check succeeds. Keep the `UVGrid` result for sampling metadata; registering its
mesh or exporting a mesh snapshot does not persist the whole result.

Grid sizes default to the object's `nU`, `nV`, with optional `n_u`, `n_v` overrides
for this call. Overrides do not change the object's defaults; use the returned
`grid.n_u` and `grid.n_v` when interpreting its arrays. At least two longitudinal
and four circumferential vertices are required, including the duplicated seam.
Vertex order is U-fastest: `index = j*n_u+i`. The output UV coordinates span the
unit square. Input U spans zero to the positive width of its boundary rings;
input V is sampled from zero to the positive period determined by the seam
pairs. Both chart axes are normalized, with no change to physical XYZ units.

## The numerical choice is explicit

The default `interpolation="mesh"` tiles and samples the supplied triangles
barycentrically. Sampled points lie on those original triangles. The new grid
triangles connect the samples, so the resulting surface remains an approximation
whose quality depends on resolution. It does not reproduce every original face
or preserve area exactly. Grid triangle degeneracy is rejected; physical 3D
self-intersection detection and an automatic resolution criterion remain absent.

MATLAB's `generateCurrentUVCutMesh` instead uses
[`scatteredInterpolant`](https://www.mathworks.com/help/matlab/ref/scatteredinterpolant.html)
on three tiled copies of the vertices, rebuilding connectivity. A nonplanar
quadrilateral can produce different interpolated XYZ when its diagonal changes.
One synthetic reference case differs by about 0.05954 coordinate units between
the original-triangle and scattered approaches. This is a documented mathematical
change in the Python default, not a claim of exact legacy-output equivalence.

Set `interpolation="matlab_scattered"` to use a linear Delaunay interpolant on
the same three-tile point cloud. Python uses SciPy's
[`Delaunay`](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.spatial.Delaunay.html),
already covered by the existing SciPy dependency. Co-circular points admit
different valid diagonals: MATLAB and SciPy can choose differently, changing
non-affine fields even when XYZ agrees. The regular reference case explicitly
checks both analytic diagonal choices; its non-affine field is not claimed to
match MATLAB bit for bit. Neither Python mode performs nearest extrapolation.
Unsupported coverage raises an error instead of substituting a nearby value.

By default, `preserve_orientation=True` chooses the grid face winding to retain
the source chart's orientation. The negative UV orientation returned by
`generateCurrentCutMesh` therefore reverses the standard positive-UV grid corner
order. With `preserve_orientation=False`, face rows/corners follow MATLAB's
`defineFacesRectilinearGrid` exactly. That explicit choice may reverse physical
normals relative to the source. No hidden `flipy` or APDV transform occurs.

## Seam handling and field correspondence

`tile_annular_mesh(mesh, path_pairs, tile_count=(1,1))` exposes the sewn tiling
operation in `tubular.resampling`. It retains MATLAB's vertex/face order and
returns source vertex IDs, face IDs and V shifts for every copy. The two counts
refer to the direction of the signed seam shift (first column minus second).
Swapping the pair columns reverses that direction. Geometry must form a valid
cut annulus, with exactly coincident XYZ pairs and one uniform vertical chart
translation to relative tolerance 1e-10. Seam welding uses the existing copy's
UV at roundoff-level mismatches; it never merges undeclared nearby vertices.

Default mesh interpolation adapts the number of tiles to the seam's V span and
removes a whole-period chart-origin offset before allocating copies. Wide seams
can still require many tiles. The MATLAB-style option retains its fixed three
tiles. Input and sewn charts undergo orientation and boundary-intersection
checks. The final V=1 samples reuse V=0 weights and geometry exactly, recording
an additional period in their provenance. The output seam pairs are first row
(V=0), then last row (V=1), as in MATLAB's UV-grid output.

The result contains:

- `source_vertices`, `barycentric`: three original cut-mesh vertex indices and
  their weights for each output vertex. `grid.interpolate(values)` uses this map.
- `corner_v_shifts`, `source_query_uv`: periodic translations and query locations
  in the original chart's coordinates. These allow reconstructing the sampled
  UV from its source corners as well as the sampled XYZ.
- `source_faces`: original face IDs for default mesh sampling. It is `None` for
  scattered interpolation, whose triangles need not be original source faces.
- `path_pairs`, `source_path_pairs`, `n_u`, `n_v`: output/input seam metadata and
  grid dimensions. Arrays and mesh snapshots are read-only.

Field transfer accepts finite scalar or vector **vertex** fields whose paired
seam values agree exactly. Discontinuous values are rejected rather than
averaged implicitly. Face fields need an explicit policy at shared edges and
are not accepted by this vertex-field API. Coordinate vectors and tensors also
need basis transformations; interpolating their components alone is insufficient.

Optional `normals=` supplies periodic unit XYZ vectors at the source vertices.
They are interpolated and normalized, with cancellation rejected. No estimator
is chosen automatically. The returned normals describe the interpolated source
normal field, not normals recomputed from the grid faces; this distinction also
matters if the caller chooses legacy face winding. APDV copies, normal shifts,
plotting and file output are outside this method's Python scope.

## Validation evidence

The unchanged MATLAB class method ran in R2025b on three synthetic cases with
identity APDV transform: a regular cylinder, a nonplanar tube with a curved seam,
and reversed face winding/seam ordering. The unchanged tiling and barycentric
interpolation helpers supplied separate references for the default strategy.
Tiled connectivity/XYZ match exactly, and sampled XYZ, normals and unambiguous
fields agree within 4e-15 in this run (test rtol=1e-11, atol=1e-12). Source hashes,
fixture hashes, the Delaunay tie limitation and regeneration details are in
`tests/data/uvgrid_provenance.json` and `scripts/generate_uvgrid_reference.m`.
No MATLAB source patches were needed; class output files were temporary.

Independent tests cover polygonal-cylinder interpolation, unit normals, exact
seam closure/gluing, vertex-field transfer, source-coordinate reconstruction,
multiple-period curved seams, large integer phase offsets, rigid motion,
renumbering, invalid inputs and class preview/registration failures. Real-data
resolution/area convergence, biological overlays, s-phi construction, Ricci flow
and temporal phase stabilization remain future work.
