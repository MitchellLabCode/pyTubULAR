# Validation evidence and limits

The initial implementation was checked on macOS arm64 with Python 3.9.6,
NumPy 2.0.2, SciPy 1.13.1, and MATLAB R2025b Update 2. The repository contains
62 passing tests at this checkpoint. The CI workflow additionally requests
Linux, Windows, and macOS jobs on several Python versions; a configured job
is not evidence of a successful run. Consult GitHub Actions for those results.

## Independent reference generation

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

## Analytic and failure-case coverage

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
