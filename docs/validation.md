# Validation evidence and limits

The initial implementation was checked on macOS arm64 with Python 3.9.6,
NumPy 2.0.2, SciPy 1.13.1, and MATLAB R2025b Update 2. The repository contains
62 passing tests at the numerical-foundation checkpoint. The class milestone
extended this to 100 passing tests. The geometry-method milestone extends it to
122 passing tests. The centerline/writhe milestone extends it to 157 passing
tests. Native alignment extended it to 190 passing tests. The surface curvature
and Beltrami milestone extends it to 218 passing tests. The annular cutting and
parameterization slice extends it to **256 passing tests** in the same local
environment. The CI workflow additionally requests
Linux, Windows, and macOS jobs on several Python versions; a configured job
is not evidence of a successful run. Consult GitHub Actions for those results.

## Independent reference generation

The 0.7 annular parameterization fixture runs the unchanged `cylinderCutMesh`
custom-path branch and `flattenAnnulus` Dirichlet branch on four synthetic
annuli. Connectivity and correspondence agree exactly; observed UV differences
are below 9e-16. Independent tests cover analytic UV coordinates, conformal
cylinder scaling, seam constraints, topology, invariance, field transfer,
invalid paths, chart overlaps and class registration. See
[parameterization evidence and limits](annular_parameterization.md). This is
kernel validation, not validation of automatic seam selection or the complete
MATLAB class workflow on biological data.

`scripts/generate_matlab_reference.m` calls original functions in the MATLAB
TubULAR checkout. It does not restate the Python algorithms in MATLAB. The
fixture uses a nonuniform embedded curved patch with a nontrivial chart,
plus an outward-oriented translated tetrahedron for enclosed volume.
The source commit, SHA-256 hashes of the actual MATLAB source files, MATLAB
version, fixture hash, and generation command are recorded in
`tests/data/provenance.json`.

Run the committed comparisons without MATLAB:

```sh
python -m pytest tests/test_matlab_parity.py
python scripts/check_matlab_reference.py
```

The default threshold is `abs(error) <= 1e-12 + 1e-11*abs(reference)`.
Topology/index mappings use exact equality. The reference-data test fails if
the fixture is missing. The measured absolute differences are recorded in
[`matlab_comparison.json`](matlab_comparison.json).

To regenerate intentionally, first check out the recorded original commit and
verify the MATLAB file hashes. Then, from MATLAB:

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_matlab_reference('/path/to/tubular', ...
    '/path/to/pyTubULAR/tests/data/matlab_reference.mat');
```

Review changed expected outputs before updating provenance and accepting a new
fixture. The MATLAB binary itself is not needed for ordinary Python CI.

## TubULAR class validation

`scripts/generate_class_reference.m` queries the original MATLAB class metadata
and calls its static numerical helpers directly. The committed inventory has
202 entries: excluding the constructor and MATLAB-generated `empty` leaves
200 original method names, including publicly accessible hidden helpers.
Python checks that every name is present and that every pending method raises
`MethodNotPortedError` without changing selection or creating output files.

The coordinate fixtures cover non-square images, non-unit UV extents,
double/single covers, even/odd row counts, nematic averaging, and legacy
dorsal/ventral index groups. MATLAB arithmetic defects are tested through
explicit `legacy=True` modes; corrected defaults are separately checked with
inverse/differential identities. Dtype preservation is checked for 16-bit images.
Hashes of the class, generator, inventory, and fixture are recorded in
`tests/data/class_provenance.json`.

```sh
python -m pytest tests/test_experiment.py tests/test_class_coordinates.py
python examples/class_workflow.py
```

To regenerate the class references from the recorded, unchanged source:

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_class_reference('/path/to/tubular', '/path/to/pyTubULAR/tests/data');
```

The generator reads names and flags rather than full method argument metadata:
MATLAB's `InputNames` inspection encounters an existing missing-`end` parse
error in `generateCurrentCutMesh.m` in this source snapshot. No original source
was edited to obtain the fixtures. This is not validation of that method.

Additional Python tests check known APDV position/vector transforms, mirroring
and inverses, explicit scalar-resolution handling, time labels and gaps,
cache invalidation across every mesh stage, separate-instance state, named MAT
structs, NPZ patterns, channel/MIP axes, file reloads, area/volume scaling, units,
and class-driven geometry and pullback sampling. APDV expectations in those
initial class tests are analytic. The next milestone also initializes a
synthetic original MATLAB experiment as described below.

These tests establish the documented adapted behavior, not equivalence of all
200 methods. In particular, MATLAB preprocessing, mesh generation, plotting,
filtering, and persistence side effects have not been replicated by the class.

## Geometry-method references (0.3)

`scripts/generate_surface_reference.m` constructs an actual original MATLAB
`TubULAR` instance, calls `uv2APDV`, `interpolateOntoPullbackXY`,
`measureXYZLims`, `getXYZLims`, and `getCurrentClineDVhoop`, and saves their
outputs. The original initializer needs explicit `lambda_err=0` because that
field is absent from its default smoothing struct. Supplying it is sufficient;
no MATLAB source is edited. All experiment inputs and analysis outputs are
created in a temporary directory and removed after fixture generation.

References include a nontrivial chart, off-surface queries, non-unit U scaling,
tiled nonconstant fields, out-of-grid extrapolation, rotated/reflected/translated
meshes at two timepoints, physical padding, and a reflected centerline. The
original interpolation class ignores `iMethod`; a direct MATLAB
`griddedInterpolant` call supplies independent checks of the newly honored
nearest option and its upper-node midpoint tie rule. SciPy's default nearest
tie rule differs, so Python selects the matching nodes explicitly.

```matlab
addpath('/path/to/pyTubULAR/scripts');
generate_surface_reference('/path/to/tubular', ...
    '/path/to/pyTubULAR/tests/data/matlab_surface_reference.mat');
```

The MATLAB version and SHA-256 hashes for source files, generator, and fixture
are preserved in `tests/data/surface_provenance.json`. Run all comparisons with:

```sh
python -m pytest tests/test_surface_methods.py
python examples/surface_queries.py
```

Independent tests cover affine scalar/vector interpolation, exact shared edges,
reversed winding, UV scale changes, chart holes, overlapping faces even with
identical XYZ, empty/outside queries, source immutability, field-axis conventions,
nearest ties, changed sources/frames, unit mismatches, independent centerline
cache copies, repeated reflection, and malformed centerline reloads. Surface
face IDs outside the chart deliberately use -1 and NaN weights in Python;
MATLAB uses NaN IDs and zero weights. Shared-edge face identity is not required
to match MATLAB's triangulation tie choice.

The default interpolation is bilinear on the legacy full tiled grid, whereas
the separate `surface.locate_chart` operation is piecewise linear on the
supplied triangles. Neither is a substitute for local ARAP flattening, seam
construction, or missing chart generation. Validation remains synthetic.

## Analytic and failure-case coverage

The 0.6 milestone compares vertex normals, mixed areas, curvature tensors,
principal values and direction projectors on an open patch and a closed
ellipsoid with both windings. Its original MATLAB principal helper requires a
documented one-line loop-bound correction in a temporary copy; all other
curvature calculations are unchanged, and source repositories are untouched.
Independent checks include sphere and cylinder values, flat triangle types,
winding/scale/rigid-motion invariants, and singular-normal rejection. See
[surface_curvature.md](surface_curvature.md) and the hashed provenance record.

Beltrami references call unchanged `bc_metric.m` with 2D and 3D targets.
Analytic stretch/shear, chart-axis phase, target similarity invariance,
orientation-reversal rejection, and explicit time/vertex correspondence are
checked separately. The reference tests do not execute the original tracking,
Ricci-flow, filtering, or plotting pipelines. See [beltrami.md](beltrami.md).

The 0.5 milestone runs the original `computeAPDVCoords` on four supplied
landmark triples. The reference uses its five-significant-digit saved frame,
since the class declaration does not expose the external method's rotation
and translation outputs. Full-precision proper rotations separately exercise
the original `xyz2APDV` transform. Independent checks cover frame axes and
handedness (including antiparallel AP), rigid-motion covariance, scaling,
reflection/winding, normal and signed-volume conventions, UV correspondence,
and atomic batch registration. See [alignment validation](landmark_alignment.md)
for exact comparison scope and `tests/data/alignment_provenance.json` for hashes.

The 0.4 centerline milestone adds independent Gauss-Legendre integration over
polygon segment pairs, geometric invariance tests, physical-time derivatives
with missing acquisition frames, and rejection of crossings including nearly
parallel ones. `generate_writhe_reference.m` runs the unchanged original Levitt
and Gauss kernels and the numerical `aux_compute_writhe` helper. See the
[writhe validation details](centerline_writhe.md#validation) for fixture hashes,
the required original `linesegments` dependency, and unsupported output fields.

- An oblique affine plane supplies exact Jacobians, metric, surface area, and
  zero second fundamental form.
- Coordinate reparameterization verifies the tensor transformation law.
- Rigid motions preserve metric/area; uniform scaling gives the correct area
  and volume powers; reversing orientation flips signed quantities.
- Known tangential/normal vectors verify projection and round trips.
- Constant fields remain constant through face/vertex averaging.
- Rectilinear connectivity has an explicit expected ordering. A cut cylinder
  becomes an annulus after gluing, with its geometry and area preserved.
- Polygonal-cylinder areas approach the known smooth-cylinder area with
  refinement. This does not establish periodic curvature accuracy.
- Mesh validation covers invalid indices, duplicate/degenerate faces,
  nonfinite/complex coordinates, unreferenced vertices, inconsistent winding,
  nonmanifold edges, and bow-tie vertices.
- MATLAB I/O covers top-level arrays, named structs, a single triangle, and
  invalid one-based indices. NPZ I/O checks schema/conventions and forbids pickle.
- Linear intensity fields provide exact trilinear-interpolation expectations,
  including anisotropic spacing, nonzero origin, and integer source images.
- Pullbacks preserve holes, detect sampled overlaps, handle reversed face
  winding, distinguish missing coverage from out-of-volume samples, and cancel.
- Command-line demo/inspection and no-overwrite behavior are exercised.

## What these tests do not establish

These are small deterministic reference cases and mathematical properties,
not proof of an error-free translation. They do not establish equivalence on
real tissue datasets, every MATLAB option, extremely ill-conditioned or
extreme-magnitude coordinates, mesh self-intersections, periodic curvature,
arbitrary MATLAB experiment files, or the full acquisition-to-strain pipeline.
No full GUI or performance benchmark has been validated.

`pullback_image` detects overlaps at sampled locations, not all intersections
of the UV mesh. `signed_volume` is algebraic and assumes a meaningful shell
orientation; it does not determine whether nested shells are cavities. The
reference fixture is deliberately small enough for CI and must be expanded
with real-data cases before scientific replacement of the MATLAB pipeline.
